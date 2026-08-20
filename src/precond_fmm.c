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
#include "fabipb_system.h"

extern double **Q2PK1, **Q2PK2, **Q2PK3, **Q2PK4;
extern void (*kernel)();
double *panelIA0(panel *pnlX, panel *pnlY );
int nrCommonVtx(panel *p, panel *q, int *idxX, int *idxY);
void kernelKER4( double *x, double *y);

/* lapack: LU solver */
void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
void dgetrs_(char* C, int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);
void dgesdd_(char* C, int* N, int* M, double* A, int* LDA, double* s, double* u, int * LDU,
  double* VT, int* LDVT, double* work, int* lwork, int* iwork, int* info);

/* those variables are allocated at setupNearfield0 at fmm.c */
double *matrixA, *rhs;
int *ipiv;
double **pcBlocks;
double **pcLUBlocks;
int **pcIpivBlocks;
int *pcBlockSize;
int nPrecondBlocks;

extern double epsilon;
extern ssystem *sys;

int nlevel;

typedef struct {
  ssystem *sys;
  cube **cubes;
  int begin;
  int end;
  double scale1;
  double scale2;
  int buildLU;
  /*
   * Precomputed disjoint-pair kernel values for this batch, or NULL to compute
   * them here. Pair (i,j) of block idx lives at pairOffset[idx] + i*HMsize + j.
   */
  const double *k0, *k1, *k2, *k3;
  const long long *pairOffset;
} PrecondSetupTask;

typedef struct {
  cube **cubes;
  int begin;
  int end;
  int nPnls;
  double *sgm;
  double *pot;
  double *rhsLocal;
  double assembleTime;
  double solveTime;
  double scatterTime;
  int info;
  int failedIdx;
} PrecondApplyTask;

typedef struct {
  double *pot;
  const double *sgm;
  const double *area;
  int nPnls;
  int begin;
  int end;
  double scale1;
  double scale2;
} DiagPrecondTask;

static void *diagPrecondWorker(void *arg) {
  DiagPrecondTask *task = (DiagPrecondTask *)arg;
  int nPnls = task->nPnls;
  int i;

  for (i = task->begin; i < task->end; i++) {
    task->pot[i] = task->sgm[i] / (task->scale1 * task->area[i]);
    task->pot[nPnls + i] = task->sgm[nPnls + i] / (task->scale2 * task->area[i]);
  }
  return NULL;
}

static double wall_seconds_pc(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

static int setupThreadCountPc(int nTasks) {
  const char *env = getenv("FABIPB_SETUP_THREADS");
  int threads;

  if (env != NULL) {
    threads = atoi(env);
  } else {
    threads = fabipb_online_cpu_count();
  }
  if (threads < 1) {
    threads = 1;
  }
  if (threads > nTasks) {
    threads = nTasks;
  }
  if (threads > 32) {
    threads = 32;
  }
  return threads;
}

static int applyThreadCountPc(int nTasks) {
  const char *env = getenv("FABIPB_PRECOND_APPLY_THREADS");
  int threads;

  if (env != NULL) {
    threads = atoi(env);
  } else {
    threads = fabipb_online_cpu_count();
  }
  if (threads < 1) {
    threads = 1;
  }
  if (threads > nTasks) {
    threads = nTasks;
  }
  if (threads > 32) {
    threads = 32;
  }
  return threads;
}

static int useGpuPrecondDisjoint(const ssystem *sys) {
  const char *env = getenv("FABIPB_GPU_PRECOND_BUILD_DISJOINT");
  return (sys != NULL && sys->gpuMode > 0 && sys->maxQuadOrder == 1 &&
          !(env != NULL && atoi(env) == 0));
}

static void *precondSetupWorker(void *arg) {
  PrecondSetupTask *task = (PrecondSetupTask *)arg;
  int idx;
  long long localDisjoint = 0;
  long long localOneCommon = 0;
  long long localTwoCommon = 0;
  long long localTwoCommonRev = 0;
  long long localSelf = 0;

  for (idx = task->begin; idx < task->end; idx++) {
    cube *cb = task->cubes[idx];
    int HMsize = cb->nPnls;
    int Msize = 2 * HMsize;
    panel *pnlX, *pnlY;
    int i, j;
    int useGpuDisjoint = (task->k0 == NULL) && useGpuPrecondDisjoint(task->sys);
    panel **blockPanels = NULL;
    int *disjointDst = NULL;
    int *disjointSrc = NULL;
    int nDisjoint = 0;

    if (useGpuDisjoint) {
      blockPanels = (panel **)malloc((size_t)HMsize * sizeof(panel *));
      disjointDst = (int *)malloc((size_t)HMsize * (size_t)HMsize * sizeof(int));
      disjointSrc = (int *)malloc((size_t)HMsize * (size_t)HMsize * sizeof(int));
      if (blockPanels == NULL || disjointDst == NULL || disjointSrc == NULL) {
        free(blockPanels);
        free(disjointDst);
        free(disjointSrc);
        blockPanels = NULL;
        disjointDst = NULL;
        disjointSrc = NULL;
        useGpuDisjoint = 0;
      }
    }

    for (i = 0, pnlY = cb->pnls; i < HMsize; i++, pnlY = pnlY->nextC) {
      if (blockPanels != NULL) {
        blockPanels[i] = pnlY;
      }
    }
    for (i = 0, pnlY = cb->pnls; i < HMsize; i++, pnlY = pnlY->nextC) {
      for (j = 0, pnlX = cb->pnls; j < HMsize; j++, pnlX = pnlX->nextC) {
        int idxX[3], idxY[3];
        int nVtx = nrCommonVtx(pnlX, pnlY, idxX, idxY);
        if (task->sys->benchmarkMode > 0) {
          if (nVtx == 0) localDisjoint++;
          else if (nVtx == 1) localOneCommon++;
          else if (nVtx == 2) localTwoCommon++;
          else if (nVtx == -2) localTwoCommonRev++;
          else if (nVtx == 3) localSelf++;
        }
        if (task->k0 != NULL && nVtx == 0) {
          /* Already computed on the device for the whole batch. */
          long long at = task->pairOffset[idx] + (long long)i * HMsize + j;
          pcBlocks[idx][i*Msize+j]                 = -task->k1[at];
          pcBlocks[idx][i*Msize+j+HMsize]          = -task->k0[at];
          pcBlocks[idx][(i+HMsize)*Msize+j]        = -task->k3[at];
          pcBlocks[idx][(i+HMsize)*Msize+j+HMsize] = -task->k2[at];
        } else if (useGpuDisjoint && nVtx == 0) {
          disjointDst[nDisjoint] = i;
          disjointSrc[nDisjoint] = j;
          nDisjoint++;
        } else {
          double *KER = panelIA0(pnlX, pnlY);
          pcBlocks[idx][i*Msize+j]                 = -KER[1];
          pcBlocks[idx][i*Msize+j+HMsize]          = -KER[0];
          pcBlocks[idx][(i+HMsize)*Msize+j]        = -KER[3];
          pcBlocks[idx][(i+HMsize)*Msize+j+HMsize] = -KER[2];
        }
      }
      pcBlocks[idx][i*Msize+i] += task->scale1 * pnlY->area;
      pcBlocks[idx][(i+HMsize)*Msize+i+HMsize] += task->scale2 * pnlY->area;
    }

    if (useGpuDisjoint && nDisjoint > 0) {
      if (!gpuBuildPrecondDisjointBlock(blockPanels, HMsize, disjointDst, disjointSrc,
                                        nDisjoint, pcBlocks[idx])) {
        for (i = 0; i < nDisjoint; i++) {
          pnlY = blockPanels[disjointDst[i]];
          pnlX = blockPanels[disjointSrc[i]];
          {
            double *KER = panelIA0(pnlX, pnlY);
            int dst = disjointDst[i];
            int src = disjointSrc[i];
            pcBlocks[idx][dst*Msize+src]                 = -KER[1];
            pcBlocks[idx][dst*Msize+src+HMsize]          = -KER[0];
            pcBlocks[idx][(dst+HMsize)*Msize+src]        = -KER[3];
            pcBlocks[idx][(dst+HMsize)*Msize+src+HMsize] = -KER[2];
          }
        }
      }
    }
    free(blockPanels);
    free(disjointDst);
    free(disjointSrc);

    if (task->buildLU) {
      int info;
      memcpy(pcLUBlocks[idx], pcBlocks[idx], (size_t)Msize * (size_t)Msize * sizeof(double));
      dgetrf_(&Msize, &Msize, pcLUBlocks[idx], &Msize, pcIpivBlocks[idx], &info);
      if (info != 0) {
        fprintf(stderr, "Error: dgetrf failed in cached LU setup for leaf %d (info=%d)\n",
                idx, info);
        exit(1);
      }
    }
  }
  if (task->sys->benchmarkMode > 0) {
    __sync_fetch_and_add(&pcCaseDisjointCount, localDisjoint);
    __sync_fetch_and_add(&pcCaseOneCommonCount, localOneCommon);
    __sync_fetch_and_add(&pcCaseTwoCommonCount, localTwoCommon);
    __sync_fetch_and_add(&pcCaseTwoCommonRevCount, localTwoCommonRev);
    __sync_fetch_and_add(&pcCaseSelfCount, localSelf);
  }
  return NULL;
}

/*
 * Runs precondSetupWorker over [begin,end) with the given thread count.
 * kArrays/pairOffset may be NULL, in which case each worker computes its own
 * disjoint pairs as before.
 */
static void runPrecondSetupRange(ssystem *sys, cube **cubes, int begin, int end,
                                 double scale1, double scale2, int buildLU,
                                 const double *k0, const double *k1,
                                 const double *k2, const double *k3,
                                 const long long *pairOffset) {
  int nBlocks = end - begin;
  int nThreads = setupThreadCountPc(nBlocks);
  int t;

  if (nBlocks <= 0) return;
  if (nThreads <= 1) {
    PrecondSetupTask task;
    memset(&task, 0, sizeof(task));
    task.sys = sys; task.cubes = cubes;
    task.begin = begin; task.end = end;
    task.scale1 = scale1; task.scale2 = scale2; task.buildLU = buildLU;
    task.k0 = k0; task.k1 = k1; task.k2 = k2; task.k3 = k3;
    task.pairOffset = pairOffset;
    precondSetupWorker(&task);
    return;
  }
  {
    pthread_t *threads = (pthread_t *)calloc((size_t)nThreads, sizeof(pthread_t));
    PrecondSetupTask *tasks =
        (PrecondSetupTask *)calloc((size_t)nThreads, sizeof(PrecondSetupTask));
    int created = 0;
    ASSERT(threads != NULL);
    ASSERT(tasks != NULL);
    for (t = 0; t < nThreads; t++) {
      tasks[t].sys = sys; tasks[t].cubes = cubes;
      tasks[t].begin = begin + (int)(((long long)nBlocks * t) / nThreads);
      tasks[t].end   = begin + (int)(((long long)nBlocks * (t + 1)) / nThreads);
      tasks[t].scale1 = scale1; tasks[t].scale2 = scale2;
      tasks[t].buildLU = buildLU;
      tasks[t].k0 = k0; tasks[t].k1 = k1; tasks[t].k2 = k2; tasks[t].k3 = k3;
      tasks[t].pairOffset = pairOffset;
    }
    for (t = 0; t < nThreads - 1; t++) {
      if (pthread_create(&threads[created], NULL, precondSetupWorker, &tasks[t]) != 0) break;
      created++;
    }
    if (created < nThreads - 1) {
      /* Fold the un-launched ranges into this thread's task. */
      tasks[created].end = end;
    }
    precondSetupWorker(&tasks[created]);
    for (t = 0; t < created; t++) pthread_join(threads[t], NULL);
    free(tasks);
    free(threads);
  }
}

typedef struct {
  cube **cubes;
  const long long *pairOffset;
  int *pairSrc;
  int *pairDst;
  int begin;
  int end;
} PrecondFillTask;

static void *precondFillWorker(void *arg) {
  PrecondFillTask *task = (PrecondFillTask *)arg;
  int idx;

  for (idx = task->begin; idx < task->end; idx++) {
    cube *cb = task->cubes[idx];
    int HMsize = cb->nPnls;
    long long base = task->pairOffset[idx];
    panel *pnlY;
    int a, b;

    for (a = 0, pnlY = cb->pnls; a < HMsize; a++, pnlY = pnlY->nextC) {
      panel *pnlX;
      long long row = base + (long long)a * HMsize;
      for (b = 0, pnlX = cb->pnls; b < HMsize; b++, pnlX = pnlX->nextC) {
        task->pairDst[row + b] = pnlY->idx;
        task->pairSrc[row + b] = pnlX->idx;
      }
    }
  }
  return NULL;
}

/* Fills [begin,end) across the setup pool; segments are disjoint by offset. */
static void runPrecondFill(PrecondFillTask *proto, int begin, int end) {
  int nBlocks = end - begin;
  int nThreads = setupThreadCountPc(nBlocks);
  int t, created = 0;
  pthread_t *threads;
  PrecondFillTask *tasks;

  if (nBlocks <= 0) return;
  if (nThreads <= 1) {
    proto->begin = begin; proto->end = end;
    precondFillWorker(proto);
    return;
  }
  threads = (pthread_t *)calloc((size_t)nThreads, sizeof(pthread_t));
  tasks = (PrecondFillTask *)calloc((size_t)nThreads, sizeof(PrecondFillTask));
  ASSERT(threads != NULL);
  ASSERT(tasks != NULL);
  for (t = 0; t < nThreads; t++) {
    tasks[t] = *proto;
    tasks[t].begin = begin + (int)(((long long)nBlocks * t) / nThreads);
    tasks[t].end   = begin + (int)(((long long)nBlocks * (t + 1)) / nThreads);
  }
  for (t = 0; t < nThreads - 1; t++) {
    if (pthread_create(&threads[created], NULL, precondFillWorker, &tasks[t]) != 0) break;
    created++;
  }
  if (created < nThreads - 1) tasks[created].end = end;
  precondFillWorker(&tasks[created]);
  for (t = 0; t < created; t++) pthread_join(threads[t], NULL);
  free(tasks);
  free(threads);
}

/* Host budget for one batch, in pairs: 2 ints in, 4 doubles out per pair. */
static long long precondBatchPairCap(void) {
  const char *env = getenv("FABIPB_PRECOND_BATCH_MIB");
  double mib = 1024.0;
  long long perPair = (long long)(2 * sizeof(int) + 4 * sizeof(double));
  long long cap;

  if (env != NULL && env[0] != '\0') {
    char *end = NULL;
    double v = strtod(env, &end);
    if (end != env && *end == '\0' && v > 0.0) mib = v;
  }
  cap = (long long)((mib * 1024.0 * 1024.0) / (double)perPair);
  if (cap < 65536) cap = 65536;
  return cap;
}

/*
 * Build every block with the batched GPU path.
 *
 * Each block's pairs are a subset of the nearfield's -- children of one parent
 * cube are always neighbours -- so this reuses the nearfield kernel and one
 * resident copy of the panel geometry rather than the per-block path, which
 * took a global mutex and issued about ten synchronous CUDA calls per block.
 *
 * All HMsize^2 pairs of a block go to the device, not just the disjoint ones.
 * Sorting them out first would need either a second nrCommonVtx pass or a
 * compaction step, and the shared-vertex, shared-edge and self cases are only a
 * few percent of the pairs, so computing values that are then ignored is
 * cheaper than the bookkeeping to avoid it.
 *
 * Returns 0 if anything is unavailable, leaving the caller to run the original
 * path over the whole range.
 */
static int precondSetupBatchedGpu(ssystem *sys, cube **cubes, int nBlocks,
                                  double scale1, double scale2, int buildLU) {
  long long cap = precondBatchPairCap();
  long long *pairOffset = NULL;
  int *pairSrc = NULL, *pairDst = NULL;
  double *k[4] = {NULL, NULL, NULL, NULL};
  int b0, i, ok = 1;

  if (!useGpuPrecondDisjoint(sys) || nBlocks <= 0) return 0;
  /*
   * Below this the per-block path is faster and this one is declined.
   *
   * The per-block build issues its CUDA calls from the setup pool under a
   * mutex, so while one thread is inside the driver the other 71 are doing
   * nrCommonVtx, allocation and dgetrf. That overlap is worth more than the
   * launch overhead removed here until there are enough blocks for the
   * overhead to dominate. Measured setupPC, per-block against batched:
   *
   *      11,257 blocks   1.276 s | 1.401 s   (0.91x -- per-block wins)
   *      45,444 blocks   2.375 s | 0.701 s   (3.39x)
   *     161,005 blocks   1.191 s | 0.439 s   (2.71x)
   *   1,391,176 blocks 111.961 s | 78.932 s  (1.42x, capsid sdens=2)
   *
   * The crossover sits between the first two, so the threshold is placed
   * between them rather than at a measured point. Block counts here are not a
   * proxy for problem size: more blocks means smaller blocks, which is why the
   * per-block column is not monotonic.
   */
  {
    const char *env = getenv("FABIPB_PRECOND_BATCH_MIN_BLOCKS");
    long minBlocks = 32768;
    if (env != NULL && env[0] != '\0') {
      char *end = NULL;
      long v = strtol(env, &end, 10);
      if (end != env && *end == '\0' && v >= 0) minBlocks = v;
    }
    if ((long)nBlocks < minBlocks) return 0;
  }

  pairOffset = (long long *)calloc((size_t)nBlocks, sizeof(long long));
  pairSrc = (int *)malloc((size_t)cap * sizeof(int));
  pairDst = (int *)malloc((size_t)cap * sizeof(int));
  for (i = 0; i < 4; i++) k[i] = (double *)malloc((size_t)cap * sizeof(double));
  if (pairOffset == NULL || pairSrc == NULL || pairDst == NULL ||
      k[0] == NULL || k[1] == NULL || k[2] == NULL || k[3] == NULL) {
    ok = 0;
  }

  for (b0 = 0; ok && b0 < nBlocks; ) {
    long long used = 0;
    int b1 = b0;
    int idx;

    /* Grow the batch while it fits; always take at least one block. */
    while (b1 < nBlocks) {
      long long need = (long long)cubes[b1]->nPnls * (long long)cubes[b1]->nPnls;
      if (b1 > b0 && used + need > cap) break;
      pairOffset[b1] = used;
      used += need;
      b1++;
      if (used >= cap) break;
    }
    if (used > cap) { ok = 0; break; }

    /* Threaded: blocks write disjoint segments of the pair arrays. */
    {
      PrecondFillTask fill;
      fill.cubes = cubes;
      fill.pairOffset = pairOffset;
      fill.pairSrc = pairSrc;
      fill.pairDst = pairDst;
      runPrecondFill(&fill, b0, b1);
    }

    if (!gpuBuildPrecondPairsBatched(sys, pairSrc, pairDst, used,
                                     k[0], k[1], k[2], k[3])) {
      ok = 0;
      break;
    }
    runPrecondSetupRange(sys, cubes, b0, b1, scale1, scale2, buildLU,
                         k[0], k[1], k[2], k[3], pairOffset);
    b0 = b1;
  }

  free(pairOffset);
  free(pairSrc);
  free(pairDst);
  for (i = 0; i < 4; i++) free(k[i]);
  return ok;
}

static void *precondApplyLUWorker(void *arg) {
  PrecondApplyTask *task = (PrecondApplyTask *)arg;
  int idx;

  for (idx = task->begin; idx < task->end; idx++) {
    cube *cb = task->cubes[idx];
    int Msize = pcBlockSize[idx];
    int HMsize = cb->nPnls;
    double *rhsLocal = task->rhsLocal;
    double t0;
    int i;
    int info;

    t0 = wall_seconds_pc();
    for (i = 0; i < HMsize; i++) {
      rhsLocal[i] = task->sgm[cb->pnls->idx + i];
      rhsLocal[i + HMsize] = task->sgm[task->nPnls + cb->pnls->idx + i];
    }
    task->assembleTime += wall_seconds_pc() - t0;

    t0 = wall_seconds_pc();
    dgetrs_(&nChr, &Msize, &oneI, pcLUBlocks[idx], &Msize, pcIpivBlocks[idx], rhsLocal, &Msize, &info);
    if (info != 0) {
      task->info = info;
      task->failedIdx = idx;
      return NULL;
    }
    task->solveTime += wall_seconds_pc() - t0;

    t0 = wall_seconds_pc();
    for (i = 0; i < HMsize; i++) {
      task->pot[cb->pnls->idx + i] = rhsLocal[i];
      task->pot[task->nPnls + cb->pnls->idx + i] = rhsLocal[i + HMsize];
    }
    task->scatterTime += wall_seconds_pc() - t0;
  }

  return NULL;
}

void setupPreconditioning(ssystem *sys) {

  int maxnPnls=0, idx, ttlcube=0;
  cube *cb;
  double scale1, scale2;

  nlevel=sys->depth-1;
  //nlevel=sys->depth;
  kernel = kernelKER4;
  scale1 = (1.0+epsilon)/2.0;
  scale2 = (1.0+1.0/epsilon)/2.0;
  pcCaseDisjointCount = 0;
  pcCaseOneCommonCount = 0;
  pcCaseTwoCommonCount = 0;
  pcCaseTwoCommonRevCount = 0;
  pcCaseSelfCount = 0;
  pcBlockCount = 0;
  pcBlockSizeSum = 0;
  pcBlockSizeSqSum = 0;
  pcBlockSizeMin = 0;
  pcBlockSizeMax = 0;

  for ( idx=0, cb=sys->cubeList[nlevel]; cb!=NULL; cb=cb->next,idx++ ) {
    maxnPnls = cb->nPnls > maxnPnls ? cb->nPnls : maxnPnls;
    ttlcube += cb->nPnls;
    if (pcBlockCount == 0 || cb->nPnls < pcBlockSizeMin) {
      pcBlockSizeMin = cb->nPnls;
    }
    if (pcBlockCount == 0 || cb->nPnls > pcBlockSizeMax) {
      pcBlockSizeMax = cb->nPnls;
    }
    pcBlockCount++;
    pcBlockSizeSum += cb->nPnls;
    pcBlockSizeSqSum += (long long)cb->nPnls * (long long)cb->nPnls;
  }

  nPrecondBlocks = idx;
  maxnPnls *= 2;
  /*
   * The dense block is indexed as pcBlocks[idx][i*Msize+j] with int arithmetic
   * throughout, so a leaf holding more than 2^31/2 panels per side wraps the
   * index negative and writes outside the block. That needs ~23k panels in one
   * leaf cube, which only happens when a very large mesh is run on a shallow
   * tree (the 41M-panel production run peaked at 105 per leaf). Catch it here
   * with an actionable message rather than corrupting the heap.
   */
  if (maxnPnls > 46340) {
    fprintf(stderr,
            "Error: preconditioner leaf block is %d x %d, too large to index with "
            "32-bit arithmetic.\n"
            "       The tree is too shallow for this mesh: increase -t (current depth %d) "
            "or use -P=3 (diagonal preconditioner).\n",
            maxnPnls, maxnPnls, sys->depth);
    exit(1);
  }
  CALLOC_FULL(matrixA, (size_t)maxnPnls * (size_t)maxnPnls, double, OFF, ASOLVER);
  CALLOC_FULL(ipiv, maxnPnls, int, OFF, ASOLVER);
  CALLOC_FULL(rhs, maxnPnls, double, OFF, ASOLVER);
  if ((sys->precondCacheMode > 0 && sys->precondCacheMode != 3) ||
      sys->debugComparePrecond > 0) {
    cube **precondCubes;
    CALLOC_FULL(pcBlocks, nPrecondBlocks, double *, OFF, ASOLVER);
    CALLOC_FULL(pcBlockSize, nPrecondBlocks, int, OFF, ASOLVER);
    if (sys->precondCacheMode > 1 || sys->debugComparePrecond > 0) {
      CALLOC_FULL(pcLUBlocks, nPrecondBlocks, double *, OFF, ASOLVER);
      CALLOC_FULL(pcIpivBlocks, nPrecondBlocks, int *, OFF, ASOLVER);
    }
    precondCubes = (cube **)calloc((size_t)nPrecondBlocks, sizeof(cube *));
    ASSERT(precondCubes != NULL);
    for (idx = 0, cb = sys->cubeList[nlevel]; cb != NULL; cb = cb->next, idx++) {
      int HMsize = cb->nPnls;
      int Msize = 2 * HMsize;

      precondCubes[idx] = cb;
      pcBlockSize[idx] = Msize;
      CALLOC_FULL(pcBlocks[idx], (size_t)Msize * (size_t)Msize, double, OFF, ASOLVER);
      if (pcLUBlocks != NULL) {
        CALLOC_FULL(pcLUBlocks[idx], (size_t)Msize * (size_t)Msize, double, OFF, ASOLVER);
        CALLOC_FULL(pcIpivBlocks[idx], Msize, int, OFF, ASOLVER);
      }
    }
    /*
     * Batched GPU build first: the block pairs duplicate nearfield pairs, so it
     * shares that kernel and one resident panel-geometry upload instead of the
     * per-block path's mutex and ~10 synchronous CUDA calls per block. Falls
     * back to the original per-block build if anything is unavailable.
     */
    if (!precondSetupBatchedGpu(sys, precondCubes, nPrecondBlocks,
                                scale1, scale2, (pcLUBlocks != NULL))) {
      runPrecondSetupRange(sys, precondCubes, 0, nPrecondBlocks,
                           scale1, scale2, (pcLUBlocks != NULL),
                           NULL, NULL, NULL, NULL, NULL);
    }
    free(precondCubes);
  }
  if (sys->benchmarkMode > 0) {
    printf("Maximum number of elements in finest cluster: %d\n", maxnPnls);
    printf("----------------------------\n");
  }
}


/*
 * Pure diagonal (Jacobi) preconditioner: divides each unknown by its own
 * diagonal system entry (scale1*area for the potential block, scale2*area
 * for the normal-derivative block -- see the identical scale1*pnl->area /
 * scale2*pnl->area terms added in MtVmain), with zero inter-panel coupling.
 * This mirrors TABI-PB's default `precondition_diagonal` (precondition.cpp),
 * adapted to root's own Galerkin diagonal, which is area-weighted where
 * TABI-PB's node-patch collocation diagonal is a bare constant.
 */
int PtVfmmDiagonal(double *pot, double *sgm) {
  int nPnls = sys->nPnls;
  double scale1 = (1.0 + epsilon) / 2.0;
  double scale2 = (1.0 + 1.0 / epsilon) / 2.0;
  const double *area = sys->panelArea;
  int i;

  /* Indexed over the contiguous area array rather than walking the panel list;
   * see buildPanelIndex() in fmm.c. Entries are independent, so the loop is
   * split across threads with disjoint output ranges -- the result does not
   * depend on the thread count. */
  {
    int nThreads = applyThreadCountPc(nPnls);
    if (nThreads <= 1) {
      for (i = 0; i < nPnls; i++) {
        pot[i] = sgm[i] / (scale1 * area[i]);
        pot[nPnls + i] = sgm[nPnls + i] / (scale2 * area[i]);
      }
    } else {
      pthread_t *threads = (pthread_t *)calloc((size_t)nThreads, sizeof(pthread_t));
      DiagPrecondTask *tasks =
          (DiagPrecondTask *)calloc((size_t)nThreads, sizeof(DiagPrecondTask));
      int t, created = 0;

      if (threads == NULL || tasks == NULL) {
        free(threads);
        free(tasks);
        for (i = 0; i < nPnls; i++) {
          pot[i] = sgm[i] / (scale1 * area[i]);
          pot[nPnls + i] = sgm[nPnls + i] / (scale2 * area[i]);
        }
        return 0;
      }
      for (t = 0; t < nThreads; t++) {
        tasks[t].pot = pot;
        tasks[t].sgm = sgm;
        tasks[t].area = area;
        tasks[t].nPnls = nPnls;
        tasks[t].scale1 = scale1;
        tasks[t].scale2 = scale2;
        tasks[t].begin = (int)(((long long)nPnls * t) / nThreads);
        tasks[t].end = (int)(((long long)nPnls * (t + 1)) / nThreads);
        if (pthread_create(&threads[t], NULL, diagPrecondWorker, &tasks[t]) != 0) {
          break;
        }
        created++;
      }
      for (t = 0; t < created; t++) {
        pthread_join(threads[t], NULL);
      }
      if (created < nThreads) {
        /* Cover whatever the unstarted threads would have done. */
        int from = (created > 0) ? tasks[created - 1].end : 0;
        for (i = from; i < nPnls; i++) {
          pot[i] = sgm[i] / (scale1 * area[i]);
          pot[nPnls + i] = sgm[nPnls + i] / (scale2 * area[i]);
        }
      }
      free(tasks);
      free(threads);
    }
  }

  return 0;
} /* PtVfmmDiagonal */

/*
 * Preconditioner by using the direct summation matrix
*/
int PtVfmm(double *pot, double *sgm) {
//void preconditioningFMM(ssystem *sys) {
  int i, j, idx, Msize, HMsize, inc;
  int nPnls=sys->nPnls;
  double scale1, scale2, *KER;
  cube *cb;
  panel *pnlX, *pnlY;
  double t0;

  scale1 = (1.0+epsilon)/2.0;
  scale2 = (1.0+1.0/epsilon)/2.0;

  for ( idx=0, cb=sys->cubeList[nlevel]; cb != NULL; cb=cb->next ) {
    Msize = 2*cb->nPnls;
    HMsize = cb->nPnls;

    t0 = wall_seconds_pc();
    for ( i=0, pnlY=cb->pnls; i<HMsize; i++, pnlY=pnlY->nextC ) {
      for ( j=0, pnlX=cb->pnls; j<HMsize; j++, pnlX=pnlX->nextC ) {
        KER = panelIA0(pnlX, pnlY);
        matrixA[i*Msize+j]                 = -KER[1];
        matrixA[i*Msize+j+HMsize]          = -KER[0];
        matrixA[(i+HMsize)*Msize+j]        = -KER[3];
        matrixA[(i+HMsize)*Msize+j+HMsize] = -KER[2];
      }
      matrixA[i*Msize+i] += scale1*pnlY->area;
      matrixA[(i+HMsize)*Msize+i+HMsize] += scale2*pnlY->area;

      rhs[i] = sgm[cb->pnls->idx+i];
      rhs[i+HMsize] = sgm[nPnls+cb->pnls->idx+i];
    }
    pcAssembleTime += wall_seconds_pc() - t0;

    t0 = wall_seconds_pc();
    /* inc doubles as the LAPACK INFO out-parameter here and was never
     * inspected: a singular leaf block left a zero pivot in U, and the dgetrs
     * below then divided by it, seeding inf/NaN into the preconditioned vector
     * and silently poisoning the Krylov basis. PtVfmmCachedLU already checks. */
    dgetrf_( &Msize, &Msize, matrixA, &Msize, ipiv, &inc );
    if (inc != 0) {
      fprintf(stderr, "Error: dgetrf failed in preconditioner apply for leaf %d (info=%d)\n",
              idx, inc);
      exit(1);
    }
    pcFactorTime += wall_seconds_pc() - t0;
    t0 = wall_seconds_pc();
    dgetrs_( &nChr, &Msize, &oneI, matrixA, &Msize, ipiv, rhs, &Msize, &inc );
    if (inc != 0) {
      fprintf(stderr, "Error: dgetrs failed in preconditioner apply for leaf %d (info=%d)\n",
              idx, inc);
      exit(1);
    }
    pcSolveTime += wall_seconds_pc() - t0;

    t0 = wall_seconds_pc();
    for ( i=0; i<HMsize; i++ ) {
      pot[cb->pnls->idx+i] = rhs[i];
      pot[nPnls+cb->pnls->idx+i] = rhs[i+HMsize];
    }
    idx += cb->nNbrs;

    for ( i=0; i<Msize; i++ ) {
      for ( j=0; j<Msize; j++ ) matrixA[i*Msize+j] = 0.0;
      rhs[i] = 0.0;
    }
    pcScatterTime += wall_seconds_pc() - t0;
  }

  //free(rhs);
  //free(ipiv);
  //for ( i=0; i<maxnPnls; i++ ) free(matrixA[i]);
  //free(matrixA);
  //exit(0);
  return 0;
}

int PtVfmmCached(double *pot, double *sgm) {
  int i, idx, Msize, HMsize, inc;
  int nPnls = sys->nPnls;
  cube *cb;
  double t0;

  ASSERT(pcBlocks != NULL);
  ASSERT(pcBlockSize != NULL);

  for (idx = 0, cb = sys->cubeList[nlevel]; cb != NULL; cb = cb->next, idx++) {
    Msize = pcBlockSize[idx];
    HMsize = cb->nPnls;

    t0 = wall_seconds_pc();
    memcpy(matrixA, pcBlocks[idx], (size_t)Msize * (size_t)Msize * sizeof(double));
    for (i = 0; i < HMsize; i++) {
      rhs[i] = sgm[cb->pnls->idx+i];
      rhs[i+HMsize] = sgm[nPnls+cb->pnls->idx+i];
    }
    pcAssembleTime += wall_seconds_pc() - t0;

    t0 = wall_seconds_pc();
    dgetrf_(&Msize, &Msize, matrixA, &Msize, ipiv, &inc);
    if (inc != 0) {
      fprintf(stderr, "Error: dgetrf failed in cached preconditioner apply for leaf %d (info=%d)\n",
              idx, inc);
      exit(1);
    }
    pcFactorTime += wall_seconds_pc() - t0;
    t0 = wall_seconds_pc();
    dgetrs_(&nChr, &Msize, &oneI, matrixA, &Msize, ipiv, rhs, &Msize, &inc);
    if (inc != 0) {
      fprintf(stderr, "Error: dgetrs failed in cached preconditioner apply for leaf %d (info=%d)\n",
              idx, inc);
      exit(1);
    }
    pcSolveTime += wall_seconds_pc() - t0;

    t0 = wall_seconds_pc();
    for (i = 0; i < HMsize; i++) {
      pot[cb->pnls->idx+i] = rhs[i];
      pot[nPnls+cb->pnls->idx+i] = rhs[i+HMsize];
    }
    for (i = 0; i < Msize; i++) {
      rhs[i] = 0.0;
    }
    pcScatterTime += wall_seconds_pc() - t0;
  }

  return 0;
}

int PtVfmmCachedLU(double *pot, double *sgm) {
  int i, idx, Msize, HMsize, info;
  int nPnls = sys->nPnls;
  cube *cb;
  double t0;

  ASSERT(pcLUBlocks != NULL);
  ASSERT(pcIpivBlocks != NULL);
  ASSERT(pcBlockSize != NULL);

  if (nPrecondBlocks > 1) {
    int nThreads = applyThreadCountPc(nPrecondBlocks);
    if (nThreads > 1) {
      cube **applyCubes = (cube **)calloc((size_t)nPrecondBlocks, sizeof(cube *));
      pthread_t *threads = (pthread_t *)calloc((size_t)nThreads, sizeof(pthread_t));
      PrecondApplyTask *tasks = (PrecondApplyTask *)calloc((size_t)nThreads, sizeof(PrecondApplyTask));
      double wallStart, wallEnd;
      ASSERT(applyCubes != NULL);
      ASSERT(threads != NULL);
      ASSERT(tasks != NULL);

      for (idx = 0, cb = sys->cubeList[nlevel]; cb != NULL; cb = cb->next, idx++) {
        applyCubes[idx] = cb;
      }
      wallStart = wall_seconds_pc();
      for (idx = 0; idx < nThreads; idx++) {
        int maxRhs = 2 * pcBlockSizeMax;
        tasks[idx].cubes = applyCubes;
        tasks[idx].begin = (int)(((long long)nPrecondBlocks * idx) / nThreads);
        tasks[idx].end = (int)(((long long)nPrecondBlocks * (idx + 1)) / nThreads);
        tasks[idx].nPnls = nPnls;
        tasks[idx].sgm = sgm;
        tasks[idx].pot = pot;
        tasks[idx].rhsLocal = (double *)calloc((size_t)maxRhs, sizeof(double));
        tasks[idx].assembleTime = 0.0;
        tasks[idx].solveTime = 0.0;
        tasks[idx].scatterTime = 0.0;
        tasks[idx].info = 0;
        tasks[idx].failedIdx = -1;
        ASSERT(tasks[idx].rhsLocal != NULL);
        pthread_create(&threads[idx], NULL, precondApplyLUWorker, &tasks[idx]);
      }
      for (idx = 0; idx < nThreads; idx++) {
        pthread_join(threads[idx], NULL);
        if (tasks[idx].info != 0) {
          fprintf(stderr, "Error: dgetrs failed in cached LU apply for leaf %d (info=%d)\n",
                  tasks[idx].failedIdx, tasks[idx].info);
          exit(1);
        }
        free(tasks[idx].rhsLocal);
      }
      wallEnd = wall_seconds_pc();
      pcSolveTime += wallEnd - wallStart;
      free(tasks);
      free(threads);
      free(applyCubes);
      return 0;
    }
  }

  for (idx = 0, cb = sys->cubeList[nlevel]; cb != NULL; cb = cb->next, idx++) {
    Msize = pcBlockSize[idx];
    HMsize = cb->nPnls;

    t0 = wall_seconds_pc();
    for (i = 0; i < HMsize; i++) {
      rhs[i] = sgm[cb->pnls->idx+i];
      rhs[i+HMsize] = sgm[nPnls+cb->pnls->idx+i];
    }
    pcAssembleTime += wall_seconds_pc() - t0;

    t0 = wall_seconds_pc();
    dgetrs_(&nChr, &Msize, &oneI, pcLUBlocks[idx], &Msize, pcIpivBlocks[idx], rhs, &Msize, &info);
    if (info != 0) {
      fprintf(stderr, "Error: dgetrs failed in cached LU apply for leaf %d (info=%d)\n",
              idx, info);
      exit(1);
    }
    pcSolveTime += wall_seconds_pc() - t0;

    t0 = wall_seconds_pc();
    for (i = 0; i < HMsize; i++) {
      pot[cb->pnls->idx+i] = rhs[i];
      pot[nPnls+cb->pnls->idx+i] = rhs[i+HMsize];
    }
    for (i = 0; i < Msize; i++) {
      rhs[i] = 0.0;
    }
    pcScatterTime += wall_seconds_pc() - t0;
  }

  return 0;
}
