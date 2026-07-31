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

static int applyThreadCountPc(int nTasks) {
  const char *env = getenv("FABIPB_PRECOND_APPLY_THREADS");
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
          !(env != NULL && atoi(env) == 0));
}

static int precondDebugEnabled(void) {
  static int initialized = 0;
  static int enabled = 0;
  const char *env;

  if (!initialized) {
    env = getenv("FABIPB_PRECOND_DEBUG");
    enabled = (env != NULL && atoi(env) != 0);
    initialized = 1;
  }
  return enabled;
}

static int precondDebugLimit(void) {
  static int initialized = 0;
  static int limit = 8;
  const char *env;

  if (!initialized) {
    env = getenv("FABIPB_PRECOND_DEBUG_LIMIT");
    if (env != NULL && atoi(env) > 0) {
      limit = atoi(env);
    }
    initialized = 1;
  }
  return limit;
}

static int precondDebugTargetBlock(void) {
  static int initialized = 0;
  static int target = -1;
  const char *env;

  if (!initialized) {
    env = getenv("FABIPB_PRECOND_DEBUG_BLOCK");
    if (env != NULL) {
      target = atoi(env);
    }
    initialized = 1;
  }
  return target;
}

static int precondDebugBlockEnabled(int idx) {
  if (!precondDebugEnabled()) {
    return 0;
  }
  return idx < precondDebugLimit() || idx == precondDebugTargetBlock();
}

static int precondFirstApplyDebugEnabled(void) {
  static int initialized = 0;
  static int enabled = 0;
  const char *env;

  if (!initialized) {
    env = getenv("FABIPB_PRECOND_DEBUG_FIRST_APPLY");
    enabled = (env != NULL && atoi(env) != 0);
    initialized = 1;
  }
  return enabled;
}

static void printVectorPreview(const char *tag, int idx, const double *x, int n) {
  int i;
  int limit = (n < 6) ? n : 6;

  printf("PRECOND vec %s: block=%d n=%d", tag, idx, n);
  for (i = 0; i < limit; i++) {
    printf(" x[%d]=%e", i, x[i]);
  }
  printf("\n");
}

static double vecInfNorm(const double *x, int n) {
  int i;
  double vmax = 0.0;

  for (i = 0; i < n; i++) {
    double ax = fabs(x[i]);
    if (ax > vmax) {
      vmax = ax;
    }
  }
  return vmax;
}

static void printMatrixDiagnostics(const char *tag, int idx, int Msize, const double *amat) {
  int i, j;
  int maxRow = 0;
  int maxCol = 0;
  double diagMin = 0.0;
  double diagMax = 0.0;
  double rowSumInf = 0.0;
  double maxAbs = 0.0;
  double trace = 0.0;

  for (i = 0; i < Msize; i++) {
    double rowSum = 0.0;
    double di = fabs(amat[i * Msize + i]);
    if (i == 0 || di < diagMin) {
      diagMin = di;
    }
    if (di > diagMax) {
      diagMax = di;
    }
    trace += amat[i * Msize + i];
    for (j = 0; j < Msize; j++) {
      double aij = fabs(amat[i * Msize + j]);
      rowSum += aij;
      if (aij > maxAbs) {
        maxAbs = aij;
        maxRow = i;
        maxCol = j;
      }
    }
    if (rowSum > rowSumInf) {
      rowSumInf = rowSum;
    }
  }

  printf("PRECOND matrix %s: block=%d size=%d row_sum_inf=%e diag_abs_min=%e diag_abs_max=%e max_abs=%e max_pos=(%d,%d) trace=%e\n",
         tag, idx, Msize, rowSumInf, diagMin, diagMax, maxAbs, maxRow, maxCol, trace);
}

static void printMatrixSvdDiagnostics(const char *tag, int idx, int Msize, const double *amat) {
  char jobz = 'N';
  int m = Msize;
  int n = Msize;
  int lda = Msize;
  int ldu = 1;
  int ldvt = 1;
  int info = 0;
  int lwork = -1;
  int minmn = Msize;
  int *iwork;
  double wkopt = 0.0;
  double *acopy;
  double *s;
  double *u_dummy;
  double *vt_dummy;
  double *work;
  double sigmaMax;
  double sigmaMin;
  double condEst;

  acopy = (double *)malloc((size_t)Msize * (size_t)Msize * sizeof(double));
  s = (double *)malloc((size_t)minmn * sizeof(double));
  iwork = (int *)malloc((size_t)(8 * minmn) * sizeof(int));
  u_dummy = (double *)malloc(sizeof(double));
  vt_dummy = (double *)malloc(sizeof(double));
  if (acopy == NULL || s == NULL || iwork == NULL || u_dummy == NULL || vt_dummy == NULL) {
    free(acopy);
    free(s);
    free(iwork);
    free(u_dummy);
    free(vt_dummy);
    return;
  }
  memcpy(acopy, amat, (size_t)Msize * (size_t)Msize * sizeof(double));
  dgesdd_(&jobz, &m, &n, acopy, &lda, s, u_dummy, &ldu, vt_dummy, &ldvt, &wkopt, &lwork, iwork, &info);
  if (info != 0) {
    printf("PRECOND svd %s: block=%d size=%d info=%d\n", tag, idx, Msize, info);
    free(acopy);
    free(s);
    free(iwork);
    free(u_dummy);
    free(vt_dummy);
    return;
  }
  lwork = (int)wkopt;
  if (lwork < 1) {
    lwork = 1;
  }
  work = (double *)malloc((size_t)lwork * sizeof(double));
  if (work == NULL) {
    free(acopy);
    free(s);
    free(iwork);
    free(u_dummy);
    free(vt_dummy);
    return;
  }
  memcpy(acopy, amat, (size_t)Msize * (size_t)Msize * sizeof(double));
  dgesdd_(&jobz, &m, &n, acopy, &lda, s, u_dummy, &ldu, vt_dummy, &ldvt, work, &lwork, iwork, &info);
  if (info != 0) {
    printf("PRECOND svd %s: block=%d size=%d info=%d\n", tag, idx, Msize, info);
  } else {
    sigmaMax = s[0];
    sigmaMin = s[minmn - 1];
    condEst = (sigmaMin > 0.0) ? (sigmaMax / sigmaMin) : INFINITY;
    printf("PRECOND svd %s: block=%d size=%d sigma_max=%e sigma_min=%e cond2_est=%e\n",
           tag, idx, Msize, sigmaMax, sigmaMin, condEst);
  }
  free(work);
  free(acopy);
  free(s);
  free(iwork);
  free(u_dummy);
  free(vt_dummy);
}

static void printLuDiagnostics(const char *tag, int idx, int Msize, const double *lu, const int *ipiv, int info) {
  int i;
  int pivSwaps = 0;
  double udiagMin = 0.0;
  double udiagMax = 0.0;

  for (i = 0; i < Msize; i++) {
    double du = fabs(lu[i * Msize + i]);
    if (i == 0 || du < udiagMin) {
      udiagMin = du;
    }
    if (du > udiagMax) {
      udiagMax = du;
    }
    if (ipiv != NULL && ipiv[i] != i + 1) {
      pivSwaps++;
    }
  }

  printf("PRECOND LU %s: block=%d size=%d info=%d udiag_min=%e udiag_max=%e piv_swaps=%d\n",
         tag, idx, Msize, info, udiagMin, udiagMax, pivSwaps);
}

static void printSolveDiagnostics(const char *tag, int idx, int Msize, const double *amat, const double *rhsOrig, const double *sol) {
  int i, j;
  double rhsInf = vecInfNorm(rhsOrig, Msize);
  double solInf = vecInfNorm(sol, Msize);
  double residInf = 0.0;

  for (i = 0; i < Msize; i++) {
    double ax = 0.0;
    for (j = 0; j < Msize; j++) {
      ax += amat[i * Msize + j] * sol[j];
    }
    {
      double ri = fabs(ax - rhsOrig[i]);
      if (ri > residInf) {
        residInf = ri;
      }
    }
  }

  printf("PRECOND solve %s: block=%d size=%d rhs_inf=%e sol_inf=%e resid_inf=%e rel_resid_inf=%e\n",
         tag, idx, Msize, rhsInf, solInf, residInf,
         residInf / ((rhsInf > 0.0) ? rhsInf : 1.0));
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
      if (precondDebugBlockEnabled(idx)) {
        printMatrixDiagnostics("setup-cached-lu-raw", idx, Msize, pcBlocks[idx]);
        printMatrixSvdDiagnostics("setup-cached-lu-raw", idx, Msize, pcBlocks[idx]);
      }
      memcpy(pcLUBlocks[idx], pcBlocks[idx], (size_t)Msize * (size_t)Msize * sizeof(double));
      dgetrf_(&Msize, &Msize, pcLUBlocks[idx], &Msize, pcIpivBlocks[idx], &info);
      if (precondDebugBlockEnabled(idx)) {
        printLuDiagnostics("setup-cached-lu", idx, Msize, pcLUBlocks[idx], pcIpivBlocks[idx], info);
      }
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
  CALLOC_FULL(matrixA, maxnPnls * maxnPnls, double, OFF, ASOLVER);
  CALLOC_FULL(ipiv, maxnPnls, int, OFF, ASOLVER);
  CALLOC_FULL(rhs, maxnPnls, double, OFF, ASOLVER);
  if ((sys->precondCacheMode > 0 && sys->precondCacheMode != 3) ||
      sys->debugComparePrecond > 0) {
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
  panel *pnl;
  int i;

  for (pnl = sys->pnlLst, i = 0; pnl != NULL; pnl = pnl->nextC, i++) {
    pot[i] = sgm[i] / (scale1 * pnl->area);
    pot[nPnls + i] = sgm[nPnls + i] / (scale2 * pnl->area);
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
    double *debugA = NULL;
    double *debugRhs = NULL;
    int debugThisBlock;
    Msize = 2*cb->nPnls;
    HMsize = cb->nPnls;
    debugThisBlock = precondDebugBlockEnabled(idx);

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
    if (debugThisBlock) {
      debugA = (double *)malloc((size_t)Msize * (size_t)Msize * sizeof(double));
      debugRhs = (double *)malloc((size_t)Msize * sizeof(double));
      ASSERT(debugA != NULL);
      ASSERT(debugRhs != NULL);
      memcpy(debugA, matrixA, (size_t)Msize * (size_t)Msize * sizeof(double));
      memcpy(debugRhs, rhs, (size_t)Msize * sizeof(double));
    }

    t0 = wall_seconds_pc();
    dgetrf_( &Msize, &Msize, matrixA, &Msize, ipiv, &inc );
    if (debugThisBlock) {
      printLuDiagnostics("apply-original", idx, Msize, matrixA, ipiv, inc);
    }
    pcFactorTime += wall_seconds_pc() - t0;
    t0 = wall_seconds_pc();
    dgetrs_( &nChr, &Msize, &oneI, matrixA, &Msize, ipiv, rhs, &Msize, &inc );
    if (debugThisBlock) {
      printSolveDiagnostics("apply-original", idx, Msize, debugA, debugRhs, rhs);
      free(debugA);
      free(debugRhs);
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
    double *debugRhs = NULL;
    int debugThisBlock;
    Msize = pcBlockSize[idx];
    HMsize = cb->nPnls;
    debugThisBlock = precondDebugBlockEnabled(idx);

    t0 = wall_seconds_pc();
    memcpy(matrixA, pcBlocks[idx], (size_t)Msize * (size_t)Msize * sizeof(double));
    for (i = 0; i < HMsize; i++) {
      rhs[i] = sgm[cb->pnls->idx+i];
      rhs[i+HMsize] = sgm[nPnls+cb->pnls->idx+i];
    }
    pcAssembleTime += wall_seconds_pc() - t0;
    if (debugThisBlock) {
      debugRhs = (double *)malloc((size_t)Msize * sizeof(double));
      ASSERT(debugRhs != NULL);
      memcpy(debugRhs, rhs, (size_t)Msize * sizeof(double));
    }

    t0 = wall_seconds_pc();
    dgetrf_(&Msize, &Msize, matrixA, &Msize, ipiv, &inc);
    if (debugThisBlock) {
      printLuDiagnostics("apply-cached", idx, Msize, matrixA, ipiv, inc);
    }
    pcFactorTime += wall_seconds_pc() - t0;
    t0 = wall_seconds_pc();
    dgetrs_(&nChr, &Msize, &oneI, matrixA, &Msize, ipiv, rhs, &Msize, &inc);
    if (debugThisBlock) {
      printSolveDiagnostics("apply-cached", idx, Msize, pcBlocks[idx], debugRhs, rhs);
      free(debugRhs);
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
  static int firstApplyLogged = 0;
  int i, idx, Msize, HMsize, info;
  int nPnls = sys->nPnls;
  int debugFirstApply = precondFirstApplyDebugEnabled() && !firstApplyLogged;
  int firstApplyMaxBlock = -1;
  int firstApplyMaxLocal = -1;
  double firstApplyMaxAbs = 0.0;
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
        tasks[idx].begin = (nPrecondBlocks * idx) / nThreads;
        tasks[idx].end = (nPrecondBlocks * (idx + 1)) / nThreads;
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
    double *debugRhs = NULL;
    int debugThisBlock;
    Msize = pcBlockSize[idx];
    HMsize = cb->nPnls;
    debugThisBlock = precondDebugBlockEnabled(idx);

    t0 = wall_seconds_pc();
    for (i = 0; i < HMsize; i++) {
      rhs[i] = sgm[cb->pnls->idx+i];
      rhs[i+HMsize] = sgm[nPnls+cb->pnls->idx+i];
    }
    pcAssembleTime += wall_seconds_pc() - t0;
    if (debugThisBlock || (debugFirstApply && idx < precondDebugLimit())) {
      debugRhs = (double *)malloc((size_t)Msize * sizeof(double));
      ASSERT(debugRhs != NULL);
      memcpy(debugRhs, rhs, (size_t)Msize * sizeof(double));
    }
    if (debugThisBlock) {
      printLuDiagnostics("apply-cached-lu", idx, Msize, pcLUBlocks[idx], pcIpivBlocks[idx], 0);
    }

    t0 = wall_seconds_pc();
    dgetrs_(&nChr, &Msize, &oneI, pcLUBlocks[idx], &Msize, pcIpivBlocks[idx], rhs, &Msize, &info);
    if (debugThisBlock) {
      printSolveDiagnostics("apply-cached-lu", idx, Msize, pcBlocks[idx], debugRhs, rhs);
    }
    if (debugFirstApply) {
      int localMaxIdx = 0;
      double localMaxAbs = 0.0;
      for (i = 0; i < Msize; i++) {
        double av = fabs(rhs[i]);
        if (av > localMaxAbs) {
          localMaxAbs = av;
          localMaxIdx = i;
        }
      }
      if (idx < precondDebugLimit() && debugRhs != NULL) {
        printf("PRECOND first-apply block=%d rhs_inf=%e out_inf=%e max_abs=%e local_idx=%d\n",
               idx, vecInfNorm(debugRhs, Msize), vecInfNorm(rhs, Msize), localMaxAbs, localMaxIdx);
      }
      if (localMaxAbs > firstApplyMaxAbs) {
        if (debugFirstApply) {
          double rhsInf = (debugRhs != NULL) ? vecInfNorm(debugRhs, Msize) : -1.0;
          printf("PRECOND first-apply new-max block=%d size=%d rhs_inf=%e out_inf=%e max_abs=%e local_idx=%d\n",
                 idx, Msize, rhsInf, vecInfNorm(rhs, Msize), localMaxAbs, localMaxIdx);
        }
        firstApplyMaxAbs = localMaxAbs;
        firstApplyMaxBlock = idx;
        firstApplyMaxLocal = localMaxIdx;
      }
    }
    if (debugRhs != NULL) {
      free(debugRhs);
    }
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

  if (debugFirstApply) {
    printf("PRECOND first-apply summary: max_block=%d max_local_idx=%d max_abs=%e\n",
           firstApplyMaxBlock, firstApplyMaxLocal, firstApplyMaxAbs);
    firstApplyLogged = 1;
  }

  return 0;
}
