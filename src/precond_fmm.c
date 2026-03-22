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
} PrecondSetupTask;

static double wall_seconds_pc(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

static int setupThreadCountPc(int nTasks) {
  const char *env = getenv("FABIPB_SETUP_THREADS");
  long hc;
  int threads;

  if (env != NULL) {
    threads = atoi(env);
  } else {
    hc = sysconf(_SC_NPROCESSORS_ONLN);
    threads = (hc > 0) ? (int)hc : 1;
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
          env != NULL && atoi(env) > 0);
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
    int useGpuDisjoint = useGpuPrecondDisjoint(task->sys);
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
        if (useGpuDisjoint && nVtx == 0) {
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

  for ( idx=0, cb=sys->cubeList[nlevel]; cb!=NULL; cb=cb->next,idx++ ) {
    maxnPnls = cb->nPnls > maxnPnls ? cb->nPnls : maxnPnls;
    ttlcube += cb->nPnls;
  }

  nPrecondBlocks = idx;
  maxnPnls *= 2;
  CALLOC_FULL(matrixA, maxnPnls * maxnPnls, double, OFF, ASOLVER);
  CALLOC_FULL(ipiv, maxnPnls, int, OFF, ASOLVER);
  CALLOC_FULL(rhs, maxnPnls, double, OFF, ASOLVER);
  if (sys->precondCacheMode > 0 || sys->debugComparePrecond > 0) {
    cube **precondCubes;
    int nThreads;
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
      CALLOC_FULL(pcBlocks[idx], Msize * Msize, double, OFF, ASOLVER);
      if (pcLUBlocks != NULL) {
        CALLOC_FULL(pcLUBlocks[idx], Msize * Msize, double, OFF, ASOLVER);
        CALLOC_FULL(pcIpivBlocks[idx], Msize, int, OFF, ASOLVER);
      }
    }
    nThreads = setupThreadCountPc(nPrecondBlocks);
    if (nThreads <= 1) {
      PrecondSetupTask task;
      task.sys = sys;
      task.cubes = precondCubes;
      task.begin = 0;
      task.end = nPrecondBlocks;
      task.scale1 = scale1;
      task.scale2 = scale2;
      task.buildLU = (pcLUBlocks != NULL);
      precondSetupWorker(&task);
    } else {
      pthread_t *threads = (pthread_t *)calloc((size_t)nThreads, sizeof(pthread_t));
      PrecondSetupTask *tasks = (PrecondSetupTask *)calloc((size_t)nThreads, sizeof(PrecondSetupTask));
      ASSERT(threads != NULL);
      ASSERT(tasks != NULL);
      for (idx = 0; idx < nThreads; idx++) {
        tasks[idx].sys = sys;
        tasks[idx].cubes = precondCubes;
        tasks[idx].begin = (nPrecondBlocks * idx) / nThreads;
        tasks[idx].end = (nPrecondBlocks * (idx + 1)) / nThreads;
        tasks[idx].scale1 = scale1;
        tasks[idx].scale2 = scale2;
        tasks[idx].buildLU = (pcLUBlocks != NULL);
        pthread_create(&threads[idx], NULL, precondSetupWorker, &tasks[idx]);
      }
      for (idx = 0; idx < nThreads; idx++) {
        pthread_join(threads[idx], NULL);
      }
      free(tasks);
      free(threads);
    }
    free(precondCubes);
  }
  if (sys->benchmarkMode > 0) {
    printf("Maximum number of elements in finest cluster: %d\n", maxnPnls);
    printf("----------------------------\n");
  }
}


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
    dgetrf_( &Msize, &Msize, matrixA, &Msize, ipiv, &inc );
    pcFactorTime += wall_seconds_pc() - t0;
    t0 = wall_seconds_pc();
    dgetrs_( &nChr, &Msize, &oneI, matrixA, &Msize, ipiv, rhs, &Msize, &inc );
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
    pcFactorTime += wall_seconds_pc() - t0;
    t0 = wall_seconds_pc();
    dgetrs_(&nChr, &Msize, &oneI, matrixA, &Msize, ipiv, rhs, &Msize, &inc);
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
