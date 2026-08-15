/*
 *  fmm.c
 *    routines related to fmm-style calculations
 *
 *    for piecewise constant elements
 *
 *  Author: Johannes Tausch
 *  Modified by J. Chen
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include "gkGlobal.h"
#include "gk.h"
#include "gpu_backend.h"

#define STOREM2L 0
#define SETUPONLY 0

/* blas: matrix times vector */
void dgemv_(char *tr, int *m, int *n, double *alpha, double *A, int *lda,
          double *x, int *incx, double *beta, double *y, int *incy);

void kernelKER4( double *x, double *y);
double *panelIA0(panel *pnlX, panel *pnlY );

double *calcMoments0(ssystem *sys, int order, cube *cb, int job);
void transM2M(ssystem *sys, cube *cbIn, cube *cbOut);
void transM2MWs(ssystem *sys, cube *cbIn, cube *cbOut, TransWorkspace *ws);
void transL2LWs(ssystem *sys, cube *cbIn, cube *cbOut, TransWorkspace *ws);
void transM2L(ssystem *sys, double *G0, double *Gk, cube *cbIn, cube *cbOut );
void transL2L(ssystem *sys, cube *cbIn, cube *cbOut );
void kernelDC0( double r, int p, double *G );
void kernelDS0( double r, int p, double *G );
void kernelDG0DGk( double r, int p, double *G0, double *Gk );
void setupDerivs(int order, double *x );

double curvature(double *x, double *h20, double *h11, double *h02);
double paramEllip( panel *pnl, double x, double y, double *r, double *nrm );
void dumpStats(ssystem *sys);
int nrCommonVtx( panel *p, panel *q, int *idxX, int *idxY );


/*
 * Threaded upward (M2M) and downward (L2L) passes.
 *
 * Both were linked-list walks on one thread: 44.3 s and 33.1 s of the 1,297 s
 * virus solve. Within a level the work is independent -- M2M has each parent
 * accumulating only into its own moments, L2L has each parent writing only to
 * its own children -- so cubes at a level can be split across threads. Levels
 * stay sequential, since each depends on the one before.
 *
 * transM2M/transL2L shared file-scope scratch, so the threads call the
 * workspace-taking variants with one buffer set each. Every thread writes a
 * disjoint set of cubes, so the result does not depend on the thread count.
 */
typedef struct {
  ssystem *sys;
  int lev;
  int begin;
  int end;
  int downward;             /* 0 = M2M (upward), 1 = L2L (downward) */
  TransWorkspace *ws;
} TransLevelTask;

static int transThreadCount(int nTasks) {
  const char *env = getenv("FABIPB_SETUP_THREADS");
  long hc;
  int threads;

  if (env != NULL && env[0] != '\0') {
    threads = atoi(env);
  } else {
    hc = sysconf(_SC_NPROCESSORS_ONLN);
    threads = (hc > 0) ? (int)hc : 1;
  }
  if (threads < 1) threads = 1;
  if (threads > nTasks) threads = nTasks;
  if (threads > 64) threads = 64;
  return threads;
}

static void freeTransWorkspace(TransWorkspace *ws) {
  free(ws->fcn1); free(ws->fcn2); free(ws->fcn3);
  free(ws->convR); free(ws->conv1); free(ws->conv2);
  memset(ws, 0, sizeof(*ws));
}

static int allocTransWorkspace(ssystem *sys, TransWorkspace *ws) {
  int order = sys->maxOrder;
  int nMoments = sys->nMom[order];

  ws->order = order;
  ws->nMoments = nMoments;
  ws->fcn1 = (double *)calloc((size_t)order + 1U, sizeof(double));
  ws->fcn2 = (double *)calloc((size_t)order + 1U, sizeof(double));
  ws->fcn3 = (double *)calloc((size_t)order + 1U, sizeof(double));
  ws->convR = (double *)calloc((size_t)nMoments, sizeof(double));
  ws->conv1 = (double *)calloc((size_t)nMoments, sizeof(double));
  ws->conv2 = (double *)calloc((size_t)nMoments, sizeof(double));
  return (ws->fcn1 && ws->fcn2 && ws->fcn3 && ws->convR && ws->conv1 && ws->conv2);
}

static void *transLevelWorker(void *arg) {
  TransLevelTask *task = (TransLevelTask *)arg;
  ssystem *sys = task->sys;
  int i, nKid;

  for (i = task->begin; i < task->end; i++) {
    cube *cb = sys->fmmCubeByIdx[i];
    for (nKid = 0; nKid < cb->nKids; nKid++) {
      if (task->downward) {
        transL2LWs(sys, cb, cb->kids[nKid], task->ws);
      } else {
        transM2MWs(sys, cb->kids[nKid], cb, task->ws);
      }
    }
  }
  return NULL;
}

static void transLevelSerial(ssystem *sys, int from, int to, int downward) {
  int i, nKid;

  for (i = from; i < to; i++) {
    cube *cb = sys->fmmCubeByIdx[i];
    for (nKid = 0; nKid < cb->nKids; nKid++) {
      if (downward) transL2L(sys, cb, cb->kids[nKid]);
      else          transM2M(sys, cb->kids[nKid], cb);
    }
  }
}

/*
 * Runs one level of the upward or downward pass.
 *
 * Threading only pays when a level holds enough cubes to outweigh creating and
 * joining the threads: each level is a separate fork/join and there are only a
 * handful of levels, so a small tree is faster left alone. Measured on a
 * 420k-panel mesh, where the busiest level is a few hundred cubes and a whole
 * pass costs ~25 ms, threading it made the pass three times slower. The virus
 * has 1.24M leaf cubes and passes costing hundreds of ms, which is the case
 * this is for.
 *
 * Workspaces are allocated once and reused, since allocating six buffers per
 * thread per level per matvec was itself a large part of the overhead.
 */
#define TRANS_MIN_CUBES_FOR_THREADS 8192

static TransWorkspace *gTransWs = NULL;
static int gTransWsCount = 0;

static void runTransLevel(ssystem *sys, int lev, int downward) {
  int start = sys->fmmLevelStart[lev];
  int count = sys->fmmLevelCount[lev];
  int nThreads, t, created = 0;
  pthread_t *threads;
  TransLevelTask *tasks;

  if (count <= 0) {
    return;
  }
  nThreads = transThreadCount(count);
  if (nThreads <= 1 || count < TRANS_MIN_CUBES_FOR_THREADS) {
    transLevelSerial(sys, start, start + count, downward);
    return;
  }

  if (gTransWsCount < nThreads) {
    TransWorkspace *grown = (TransWorkspace *)calloc((size_t)nThreads, sizeof(TransWorkspace));
    if (grown == NULL) {
      transLevelSerial(sys, start, start + count, downward);
      return;
    }
    for (t = 0; t < gTransWsCount; t++) grown[t] = gTransWs[t];
    for (t = gTransWsCount; t < nThreads; t++) {
      if (!allocTransWorkspace(sys, &grown[t])) {
        freeTransWorkspace(&grown[t]);
        break;
      }
    }
    if (t < nThreads) {
      int u;
      for (u = gTransWsCount; u < t; u++) freeTransWorkspace(&grown[u]);
      free(grown);
      transLevelSerial(sys, start, start + count, downward);
      return;
    }
    free(gTransWs);
    gTransWs = grown;
    gTransWsCount = nThreads;
  }

  threads = (pthread_t *)calloc((size_t)nThreads, sizeof(pthread_t));
  tasks = (TransLevelTask *)calloc((size_t)nThreads, sizeof(TransLevelTask));
  if (threads == NULL || tasks == NULL) {
    free(threads); free(tasks);
    transLevelSerial(sys, start, start + count, downward);
    return;
  }

  for (t = 0; t < nThreads; t++) {
    tasks[t].sys = sys;
    tasks[t].lev = lev;
    tasks[t].downward = downward;
    tasks[t].ws = &gTransWs[t];
    tasks[t].begin = start + (int)(((long long)count * t) / nThreads);
    tasks[t].end = start + (int)(((long long)count * (t + 1)) / nThreads);
    if (pthread_create(&threads[t], NULL, transLevelWorker, &tasks[t]) != 0) break;
    created++;
  }
  for (t = 0; t < created; t++) {
    pthread_join(threads[t], NULL);
  }
  if (created < nThreads) {
    int from = (created > 0) ? tasks[created - 1].end : start;
    transLevelSerial(sys, from, start + count, downward);
  }
  free(tasks);
  free(threads);
}

static double wall_seconds_local(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

static void buildApplyLayout(ssystem *sys) {
  cube *cb;
  int depth = sys->depth;
  int idx;
  int pairCount = 0;

  for (cb = sys->cubeList[depth]; cb != NULL; cb = cb->next) {
    pairCount += cb->nNbrs;
  }

  sys->nNearPairsFlat = pairCount;
  CALLOC(sys->nearPairSrc, pairCount, int);
  CALLOC(sys->nearPairDst, pairCount, int);

  for (idx = 0, cb = sys->cubeList[depth]; cb != NULL; cb = cb->next, idx++) {
    cb->leafFlatIdx = idx;
    sys->leafPanelStart[idx] = cb->pnls->idx;
    sys->leafPanelCount[idx] = cb->nPnls;
  }

  idx = 0;
  for (cb = sys->cubeList[depth]; cb != NULL; cb = cb->next) {
    int srcIdx = cb->leafFlatIdx;
    int inbr;
    for (inbr = 0; inbr < cb->nNbrs; inbr++, idx++) {
      sys->nearPairSrc[idx] = cb->nbrs[inbr]->leafFlatIdx;
      sys->nearPairDst[idx] = srcIdx;
    }
  }
}

/*
 * Index the panels, and copy their areas into a contiguous array.
 *
 * MtVmain and the diagonal preconditioner both need nothing from a panel but
 * its area, yet each used to walk the whole linked list to get it -- chasing
 * 10.2M pointers through 320-byte structs, roughly 3 GB of traffic, twice per
 * GMRES iteration. On the full virus that was 90 s in MtVmain and another 90 s
 * in psolve, about 14% of the solve, purely to read one double per panel.
 * A 78 MB array of areas replaces both walks and lets those loops be indexed
 * (and so parallelised) instead of pointer-chased.
 */
static void buildPanelIndex(ssystem *sys) {
  int idx;
  panel *pnl;

  CALLOC(sys->panelByIdx, sys->nPnls, panel *);
  CALLOC(sys->panelArea, sys->nPnls, double);
  for (idx = 0, pnl = sys->pnlLst; pnl != NULL; pnl = pnl->nextC, idx++) {
    ASSERT(idx < sys->nPnls);
    sys->panelByIdx[idx] = pnl;
    sys->panelArea[idx] = pnl->area;
  }
  ASSERT(idx == sys->nPnls);
}

static void buildFmmCubeLayout(ssystem *sys) {
  cube *cb;
  int depth = sys->depth;
  int height = sys->height;
  int lev, idx = 0;

  sys->nFmmCubesFlat = 0;
  for (lev = depth; lev >= height; lev--) {
    for (cb = sys->cubeList[lev]; cb != NULL; cb = cb->next) {
      sys->nFmmCubesFlat++;
    }
  }

  CALLOC(sys->fmmCubeByIdx, sys->nFmmCubesFlat, cube *);
  /* Cubes are laid out one level at a time, so each level owns a contiguous
   * run. Recording where each run starts turns the upward and downward passes
   * from linked-list walks into indexed loops, which can then be split across
   * threads. */
  CALLOC(sys->fmmLevelStart, depth + 1, int);
  CALLOC(sys->fmmLevelCount, depth + 1, int);
  for (lev = depth; lev >= height; lev--) {
    sys->fmmLevelStart[lev] = idx;
    for (cb = sys->cubeList[lev]; cb != NULL; cb = cb->next, idx++) {
      cb->flatIdx = idx;
      sys->fmmCubeByIdx[idx] = cb;
    }
    sys->fmmLevelCount[lev] = idx - sys->fmmLevelStart[lev];
  }
  ASSERT(idx == sys->nFmmCubesFlat);
}

static void buildM2LPairList(ssystem *sys) {
  cube *cb;
  int depth = sys->depth;
  int height = sys->height;
  int lev;
  int idx = 0;

  sys->nM2LPairsFlat = 0;
  for (lev = depth; lev >= height; lev--) {
    for (cb = sys->cubeList[lev]; cb != NULL; cb = cb->next) {
      int count = cb->n2Nbrs - cb->nNbrs;
      sys->nM2LPairsFlat += count;
    }
  }

  CALLOC(sys->m2lPairSrc, sys->nM2LPairsFlat, int);
  CALLOC(sys->m2lPairDst, sys->nM2LPairsFlat, int);
  CALLOC(sys->m2lPairOrder, sys->nM2LPairsFlat, int);

  for (lev = depth; lev >= height; lev--) {
    int order = sys->ordM2L[lev];
    int iNbr;
    for (cb = sys->cubeList[lev]; cb != NULL; cb = cb->next) {
      int dstIdx = cb->flatIdx;
      for (iNbr = cb->n2Nbrs - 1; iNbr >= cb->nNbrs; iNbr--, idx++) {
        int srcIdx = cb->nbrs[iNbr]->flatIdx;
        ASSERT(srcIdx >= 0);
        ASSERT(dstIdx >= 0);
        sys->m2lPairSrc[idx] = srcIdx;
        sys->m2lPairDst[idx] = dstIdx;
        sys->m2lPairOrder[idx] = order;
      }
    }
  }
  ASSERT(idx == sys->nM2LPairsFlat);
}

static void buildM2LDstGroups(ssystem *sys) {
  int idx;
  int currentDst = -1;
  int groupIdx = -1;

  sys->nM2LDstGroups = 0;
  for (idx = 0; idx < sys->nM2LPairsFlat; idx++) {
    if (sys->m2lPairDst[idx] != currentDst) {
      currentDst = sys->m2lPairDst[idx];
      sys->nM2LDstGroups++;
    }
  }

  CALLOC(sys->m2lDstGroupStart, sys->nM2LDstGroups, int);
  CALLOC(sys->m2lDstGroupCount, sys->nM2LDstGroups, int);

  currentDst = -1;
  for (idx = 0; idx < sys->nM2LPairsFlat; idx++) {
    int dstIdx = sys->m2lPairDst[idx];
    if (dstIdx != currentDst) {
      groupIdx++;
      currentDst = dstIdx;
      sys->m2lDstGroupStart[groupIdx] = idx;
      sys->m2lDstGroupCount[groupIdx] = 0;
    }
    sys->m2lDstGroupCount[groupIdx]++;
  }

  ASSERT(groupIdx + 1 == sys->nM2LDstGroups || sys->nM2LDstGroups == 0);
}

extern double **dG0;     /* workspace for setupDerivs */
extern double **dGk;     /* workspace for setupDerivs */
extern int normErr;
extern void (*kernel)(double *x, double *y);
extern void (*kernelD)(double r, int p, double *G0, double *Gk);
extern void (*kernelDC)(double r, int p, double *G);
extern void (*kernelDS)(double r, int p, double *G);
extern double kappa;
extern double epsilon;

double **Gp0, **Gpk;            /* translation matrices */
double **Q2PK1, **Q2PK2, **Q2PK3, **Q2PK4;
double **Q2M0, **Q2M1;
double **L2P0, **L2P1;

/* preconditioning variables */
extern double **matrixA, *rhs;
extern int *ipiv;


/*
 * setup everything related to FMM-style matrix-vector multiply
 * note that the orders used in the FMM are given by sys->ordM2L[]
 */
void setupFMM(ssystem *sys) {
  cube *cb, *cb1, *kid;
  int depth=sys->depth, height=sys->height;
  int lev, idx, i, j, k;
  int inbr, nNbrs;
  int order, nMoments, nCubesL;
  double r[3];
  double t0, t1;
  double tApplyLayout, tPanelIndex, tCubeLayout, tM2LPair, tM2LGroup;


  for ( nCubesL=0, cb=sys->cubeList[depth]; cb != NULL; cb=cb->next ) {
    nCubesL++;
  }
  sys->nLeafCubesFlat = nCubesL;
  CALLOC(sys->leafPanelStart, nCubesL, int);
  CALLOC(sys->leafPanelCount, nCubesL, int);
  CALLOC(Q2M0, nCubesL, double*);
  CALLOC(Q2M1, nCubesL, double*);
  CALLOC(L2P0, nCubesL, double*);
  CALLOC(L2P1, nCubesL, double*);

  order=sys->ordMom[depth];

  /* moments depend on which layer operator we have */
  t0 = wall_seconds_local();
  for ( idx=0, cb=sys->cubeList[depth]; cb!=NULL; cb=cb->next, idx++ ) {
    L2P0[idx] = Q2M0[idx] = calcMoments0(sys, order, cb, 0);
    L2P1[idx] = Q2M1[idx] = calcMoments0(sys, order, cb, 1);
  }
  t1 = wall_seconds_local();
  setupFmmLeafTime += (t1 - t0);


  t0 = wall_seconds_local();
  for ( lev=sys->depth; lev>=sys->height; lev-- ) {
    order = sys->ordMom[lev];
    nMoments  = sys->nMom[order];
    for ( cb=sys->cubeList[lev]; cb != NULL; cb=cb->next ) {
      CALLOC(cb->mom_pot, nMoments, double);
      CALLOC(cb->mom_dpdn, nMoments, double);
      CALLOC(cb->lec_k1, nMoments, double);
      CALLOC(cb->lec_k2, nMoments, double);
      CALLOC(cb->lec_k3, nMoments, double);
      CALLOC(cb->lec_k4, nMoments, double);
    }
  }
  t1 = wall_seconds_local();
  setupFmmCubeAllocTime += (t1 - t0);

  kernelDC = kernelDC0;
  kernelDS = kernelDS0;
  kernelD  = kernelDG0DGk;

  tApplyLayout = tPanelIndex = tCubeLayout = tM2LPair = tM2LGroup = 0.0;

  t0 = wall_seconds_local();
  buildApplyLayout(sys);
  t1 = wall_seconds_local();
  tApplyLayout = (t1 - t0);
  setupFmmApplyLayoutTime += tApplyLayout;

  t0 = wall_seconds_local();
  buildPanelIndex(sys);
  t1 = wall_seconds_local();
  tPanelIndex = (t1 - t0);
  setupFmmPanelIndexTime += tPanelIndex;

  t0 = wall_seconds_local();
  buildFmmCubeLayout(sys);
  t1 = wall_seconds_local();
  tCubeLayout = (t1 - t0);
  setupFmmCubeLayoutTime += tCubeLayout;

  t0 = wall_seconds_local();
  buildM2LPairList(sys);
  t1 = wall_seconds_local();
  tM2LPair = (t1 - t0);
  setupFmmM2LPairTime += tM2LPair;

  t0 = wall_seconds_local();
  buildM2LDstGroups(sys);
  t1 = wall_seconds_local();
  tM2LGroup = (t1 - t0);
  setupFmmM2LGroupTime += tM2LGroup;
  setupFmmLayoutTime += tApplyLayout + tPanelIndex + tCubeLayout + tM2LPair + tM2LGroup;
  if (sys->benchmarkMode > 0) {
    printf("Flattened apply layout: leaf-cubes=%d near-pairs=%d\n",
           sys->nLeafCubesFlat, sys->nNearPairsFlat);
    printf("Flattened interaction layout: m2l-pairs=%d dst-groups=%d\n",
           sys->nM2LPairsFlat, sys->nM2LDstGroups);
    printf("Flattened FMM cube layout: cubes=%d\n", sys->nFmmCubesFlat);
  }

} /* setupFMM */


/*
 * Add the nearfield.
 * Pcw constant case. The nearfield coefficients are computed
 * by at Here.
 * Same parameters as applyFMM().
 */
static void applyNearfield1CPU(ssystem *sys, double *alpha, double *sgm, double *pot) {
  int nPnls = sys->nPnls;
  int pairIdx;

  /* set up kernel */
  kernel = kernelKER4;

  for (pairIdx = 0; pairIdx < sys->nNearPairsFlat; pairIdx++) {
    int srcLeaf = sys->nearPairSrc[pairIdx];
    int dstLeaf = sys->nearPairDst[pairIdx];
    int srcStart = sys->leafPanelStart[srcLeaf];
    int srcCount = sys->leafPanelCount[srcLeaf];
    int dstStart = sys->leafPanelStart[dstLeaf];
    int dstCount = sys->leafPanelCount[dstLeaf];
    int i, j;

    for (i = 0; i < dstCount; i++) {
      int dstPanelIdx = dstStart + i;
      panel *pnlX = sys->panelByIdx[dstPanelIdx];
      double *KER;
      double *y_pot = &(pot[dstPanelIdx]);
      double *y_dpdn = &(pot[dstPanelIdx + nPnls]);

      for (j = 0; j < srcCount; j++) {
        int srcPanelIdx = srcStart + j;
        panel *pnlY = sys->panelByIdx[srcPanelIdx];
        double x_pot = sgm[srcPanelIdx];
        double x_dpdn = sgm[srcPanelIdx + nPnls];

        KER = panelIA0(pnlX, pnlY);
        y_pot[0] += (KER[0] * x_dpdn + KER[1] * x_pot) * (*alpha);
        y_dpdn[0] += (KER[2] * x_dpdn + KER[3] * x_pot) * (*alpha);
      }
    }
  }
} /* applyNearfield1CPU */

void applyNearfield1(ssystem *sys, double *alpha, double *sgm, double *beta, double *pot) {
  static int warnedNoGpuBackend = 0;
  static int warnedGpuApplyFailure = 0;

  if (sys->gpuMode > 0) {
    if (gpuNearfieldApply(sys, *alpha, sgm, pot)) {
      return;
    }
    if (!gpuBackendAvailable()) {
      if (!warnedNoGpuBackend) {
        printf("GPU backend requested but unavailable; using CPU nearfield path.\n");
        warnedNoGpuBackend = 1;
      }
    } else if (!warnedGpuApplyFailure) {
      printf("GPU nearfield path failed: %s; using CPU fallback.\n",
             gpuNearfieldLastError());
      warnedGpuApplyFailure = 1;
    }
  }

  applyNearfield1CPU(sys, alpha, sgm, pot);
  (void)beta;
} /* applyNearfield1 */


/*
 * FMM-style matrix-vector multiply for panels
 * Parameters
 *    sgm      input density (function on panels)
 *    pot      output potential (function on panels)
 */
void applyFMM(ssystem *sys, double *alpha, double *sgm, double *beta, double *pot) {
  cube *cb, *cb1;
  double *x, *y, *lec;
  double r[3], *self;
  int depth=sys->depth, height=sys->height, nPnls=sys->nPnls;
  int nKid, nKid1, iPnl, idx, nMom, order;
  int i, k, lev, n, n1, inc = 1;
  double time1, time2;
  static int warnedNoGpuM2L = 0;
  static int warnedNoGpuQ2M = 0;
  static int warnedNoGpuL2P = 0;

  /* zero out mom's and lec's */
  for ( lev=depth; lev>=height; lev-- ) {
    nMom  = sys->nMom[sys->ordMom[lev]];
    for ( cb=sys->cubeList[lev]; cb != NULL; cb=cb->next ) {
      for ( k=0; k<nMom;  k++ ) {
        cb->mom_pot[k] = 0.0;
        cb->mom_dpdn[k] = 0.0;
        cb->lec_k1[k] = 0.0;
        cb->lec_k2[k] = 0.0;
        cb->lec_k3[k] = 0.0;
        cb->lec_k4[k] = 0.0;
      }
    }
  }

  /* Q2M transformations */
  time1 = wall_seconds_local();
  nMom = sys->nMom[sys->ordMom[depth]];
  if (!(sys->gpuMode > 0 && sys->gpuQ2MMode > 0 && gpuQ2MApply(sys, sgm))) {
    if (sys->gpuMode > 0 && sys->gpuQ2MMode > 0 && gpuBackendAvailable() && !warnedNoGpuQ2M) {
      printf("GPU Q2M path unavailable; using CPU fallback.\n");
      warnedNoGpuQ2M = 1;
    }
    for ( idx=0, cb=sys->cubeList[depth]; cb != NULL; cb=cb->next, idx++ ) {
      x = &(sgm[cb->pnls->idx]);
      n = cb->nPnls;
      y = cb->mom_pot;
      dgemv_(&nChr, &nMom, &n, &one, Q2M1[idx], &nMom, x, &inc, &one, y, &inc);
      x = &(sgm[cb->pnls->idx+nPnls]);
      y = cb->mom_dpdn;
      dgemv_(&nChr, &nMom, &n, &one, Q2M0[idx], &nMom, x, &inc, &one, y, &inc);
    }
  }
  time2 = wall_seconds_local();
  fmmQ2MTime += (time2 - time1);

  /* upward pass */
  time1 = wall_seconds_local();
  for ( lev=depth-1; lev>=height; lev-- ) {
    runTransLevel(sys, lev, 0);
  }
  time2 = wall_seconds_local();
  fmmM2MTime += (time2 - time1);

  /* Interaction phase */
  time1 = wall_seconds_local();
  if (!(sys->gpuMode > 0 && gpuM2LApply(sys))) {
    if (sys->gpuMode > 0 && gpuBackendAvailable() && !warnedNoGpuM2L) {
      printf("GPU M2L path unavailable; using CPU fallback.\n");
      warnedNoGpuM2L = 1;
    }
  for (idx = 0; idx < sys->nM2LPairsFlat; idx++) {
#if STOREM2L
    transM2L(sys, Gp0[idx], Gpk[idx],
             sys->fmmCubeByIdx[sys->m2lPairSrc[idx]],
             sys->fmmCubeByIdx[sys->m2lPairDst[idx]]);
#else
    cb1 = sys->fmmCubeByIdx[sys->m2lPairSrc[idx]];
    cb = sys->fmmCubeByIdx[sys->m2lPairDst[idx]];
    order = sys->m2lPairOrder[idx];
    for (k = 0; k < 3; k++) r[k] = cb->x[k] - cb1->x[k];
    setupDerivs(order, r);
    transM2L(sys, dG0[0], dGk[0], cb1, cb);
#endif
  }
  }
  time2 = wall_seconds_local();
  fmmM2LTime += (time2 - time1);

  /* downward pass */
  time1 = wall_seconds_local();
  for ( lev=height; lev<depth; lev++ ) {
    runTransLevel(sys, lev, 1);
  }
  time2 = wall_seconds_local();
  fmmL2LTime += (time2 - time1);

  /* L2P transformations */
  time1 = wall_seconds_local();
  nMom = sys->nMom[sys->ordMom[depth]];
  if (!(sys->gpuMode > 0 && gpuL2PApply(sys, *alpha, *beta, pot))) {
    if (sys->gpuMode > 0 && gpuBackendAvailable() && !warnedNoGpuL2P) {
      printf("GPU L2P path unavailable; using CPU fallback.\n");
      warnedNoGpuL2P = 1;
    }
    for ( idx=0, cb=sys->cubeList[depth]; cb != NULL; cb=cb->next, idx++ ) {
      y = &(pot[cb->pnls->idx]);
      n = cb->nPnls;
      x = cb->lec_k1;
      dgemv_(&hChr, &nMom, &n, alpha, L2P0[idx], &nMom, x, &inc, beta, y, &inc);
      x = cb->lec_k2;
      dgemv_(&hChr, &nMom, &n, alpha, L2P0[idx], &nMom, x, &inc, &one, y, &inc);

      y = &(pot[cb->pnls->idx+nPnls]);
      x = cb->lec_k3;
      dgemv_(&hChr, &nMom, &n, alpha, L2P1[idx], &nMom, x, &inc, beta, y, &inc);
      x = cb->lec_k4;
      dgemv_(&hChr, &nMom, &n, alpha, L2P1[idx], &nMom, x, &inc, &one, y, &inc);
    }
  }
  time2 = wall_seconds_local();
  fmmL2PTime += (time2 - time1);

  time1 = wall_seconds_local();
  applyNearfield1(sys, alpha, sgm, beta, pot);
  time2 = wall_seconds_local();
  fmmNearTime += (time2 - time1);

} /* applyFMM */
