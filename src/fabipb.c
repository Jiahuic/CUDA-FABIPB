#define _POSIX_C_SOURCE 200809L
/*
 * fabipb.c: main driver
 * This program computes the boundary integral PB equation with fmm method
 * usage:
 *   fabipb [options] panelfile [options]
 *
 * Copyright: Jiahui Chen, Weihua Geng, Johannes Tausch
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "gkGlobal.h"
#include "gk.h"
#include "gmres.h"
#include "direct_backend.h"
#include "gpu_backend.h"

/* global variables */
int orderMom=0;
double kappa, epsilon, epsilon1=1.0, epsilon2=80.0;

ssystem *sys;

/* function pointers to kernel routines */
void (*kernel)(double *x, double *y);
void (*kernelD)(double r, int p, double *G0, double *Gk);
void (*kernelDC)(double r, int p, double *G);
void (*kernelDS)(double r, int p, double *G);
int (*MtV)(), (*PtV)();

/* routines used by the main routine */
panel *loadPanel(char *panelfile, char *density, int *numSing, ssystem *sys);
void gkInit(ssystem *sys, panel *pnlList, int order, int orderMom);
void setupFMM(ssystem *sys);
void applyFMM( ssystem *sys, double *alpha, double *sgm, double *beta, double *pot );
void setupPreconditioning(ssystem *sys);

double *panelRHS(int qOrder, panel *pnlX, double *chrY );

int MtVmain(double *alpha, double *sgm, double *beta, double *pot);
int PtVfmm(double *pot, double *sgm);
int PtVfmmCached(double *pot, double *sgm);
int PtVfmmCachedLU(double *pot, double *sgm);
int PtVmain(double *pot, double *sgm);

void applyTreecode( ssystem *sys, double *sgm, double *pot );

static void set_benchmark_thread_defaults(void) {
  const char *vars[] = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS"
  };
  size_t i;

  for (i = 0; i < sizeof(vars) / sizeof(vars[0]); i++) {
    if (getenv(vars[i]) == NULL) {
      setenv(vars[i], "1", 0);
    }
  }
}

static void buildPanelIndexDirect(ssystem *sys) {
  int idx = 0;
  panel *pnl;

  if (sys->panelByIdx != NULL) {
    return;
  }
  CALLOC(sys->panelByIdx, sys->nPnls, panel *);
  for (pnl = sys->pnlLst; pnl != NULL; pnl = pnl->nextC, idx++) {
    ASSERT(idx < sys->nPnls);
    sys->panelByIdx[idx] = pnl;
  }
}

static void compareApplyFMMOnce(ssystem *sys, double *sgm) {
  double *potCpu, *potGpu;
  double alpha = 1.0, beta = 0.0;
  double maxAbs = 0.0, l2Diff = 0.0, l2Ref = 0.0;
  double savedQ2M, savedM2M, savedM2L, savedL2L, savedL2P, savedNear;
  int oldGpuMode = sys->gpuMode;
  int maxIdx = -1;
  int i, n = 2 * sys->nPnls;

  if (oldGpuMode <= 0) {
    printf("applyFMM debug compare skipped: run with -g=1 to compare CPU and GPU paths.\n");
    return;
  }
  if (!gpuBackendAvailable()) {
    printf("applyFMM debug compare skipped: GPU backend unavailable.\n");
    return;
  }

  CALLOC(potCpu, n, double);
  CALLOC(potGpu, n, double);

  savedQ2M = fmmQ2MTime;
  savedM2M = fmmM2MTime;
  savedM2L = fmmM2LTime;
  savedL2L = fmmL2LTime;
  savedL2P = fmmL2PTime;
  savedNear = fmmNearTime;

  sys->gpuMode = 0;
  applyFMM(sys, &alpha, sgm, &beta, potCpu);
  sys->gpuMode = oldGpuMode;
  applyFMM(sys, &alpha, sgm, &beta, potGpu);

  fmmQ2MTime = savedQ2M;
  fmmM2MTime = savedM2M;
  fmmM2LTime = savedM2L;
  fmmL2LTime = savedL2L;
  fmmL2PTime = savedL2P;
  fmmNearTime = savedNear;

  for (i = 0; i < n; i++) {
    double diff = fabs(potCpu[i] - potGpu[i]);
    if (diff > maxAbs) {
      maxAbs = diff;
      maxIdx = i;
    }
    l2Diff += diff * diff;
    l2Ref += potCpu[i] * potCpu[i];
  }

  printf("applyFMM debug compare: max_abs=%e rel_l2=%e max_idx=%d cpu=%e gpu=%e\n",
         maxAbs,
         (l2Ref > 0.0) ? sqrt(l2Diff / l2Ref) : 0.0,
         maxIdx,
         (maxIdx >= 0) ? potCpu[maxIdx] : 0.0,
         (maxIdx >= 0) ? potGpu[maxIdx] : 0.0);

  free(potCpu);
  free(potGpu);
}

static void comparePrecondOnce(ssystem *sys, double *sgm) {
  double *potOrig, *potCached;
  double maxAbs = 0.0, l2Diff = 0.0, l2Ref = 0.0;
  const char *modeLabel = (sys->precondCacheMode > 1) ? "cached-lu" : "cached-blocks";
  int maxIdx = -1;
  int i, n = 2 * sys->nPnls;

  CALLOC(potOrig, n, double);
  CALLOC(potCached, n, double);

  PtVfmm(potOrig, sgm);
  if (sys->precondCacheMode > 1) {
    PtVfmmCachedLU(potCached, sgm);
  } else {
    PtVfmmCached(potCached, sgm);
  }

  for (i = 0; i < n; i++) {
    double diff = fabs(potOrig[i] - potCached[i]);
    if (diff > maxAbs) {
      maxAbs = diff;
      maxIdx = i;
    }
    l2Diff += diff * diff;
    l2Ref += potOrig[i] * potOrig[i];
  }

  printf("PtVfmm debug compare (%s): max_abs=%e rel_l2=%e max_idx=%d orig=%e test=%e\n",
         modeLabel,
         maxAbs,
         (l2Ref > 0.0) ? sqrt(l2Diff / l2Ref) : 0.0,
         maxIdx,
         (maxIdx >= 0) ? potOrig[maxIdx] : 0.0,
         (maxIdx >= 0) ? potCached[maxIdx] : 0.0);

  free(potOrig);
  free(potCached);
}

static double wall_seconds(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

static void print_usage(const char *prog) {
  printf("Usage: %s [options] <panel-base-or-pqr-path>\n", prog);
  printf("Core options:\n");
  printf("  -g=0|1    CPU only or request GPU (default: auto)\n");
  printf("  -m=0|1    reuse or regenerate MSMS mesh (default: 1)\n");
  printf("  -B=0|1    quiet default or benchmark/profiling output (default: 0)\n");
  printf("  -d=<val>  MSMS density used when -m=1 (default: 1)\n");
  printf("  -r=0|1|2  FMM, direct GPU, or direct CPU matvec (default: 0)\n");
  printf("  -Q=0|1    CPU-default or GPU-debug Q2M path (default: 0)\n");
  printf("  -G=0|1    interaction or destination-leaf GPU nearfield (default: 1)\n");
  printf("  -P=0|1|2  original, cached-block, or cached-LU preconditioner (default: 2)\n");
  printf("  -t=<lev>  tree depth\n");
  printf("  -H=<lev>  coarsest active FMM level (default: 2)\n");
  printf("  -q=<ord>  panel quadrature order\n");
  printf("  -k=<val>  Debye-Huckel kappa\n");
  printf("  -e1=<val> solvent epsilon 1\n");
  printf("  -e2=<val> solvent epsilon 2\n");
  printf("  -S=<val>  FMM separation ratio\n");
  printf("  -o=<val>  GMRES tolerance\n");
  printf("  -p=<val>  FMM order\n");
  printf("  -pm=<val> moment order override\n");
  printf("Development/debug options:\n");
  printf("  -c=1      compare one CPU/GPU applyFMM call before GMRES\n");
  printf("  -C=1      compare one original/cached preconditioner apply before GMRES\n");
  printf("  -h        show this help\n");
}
/*
 *  setup right hand side (exterior Neumann problem)
 */
void setupRHS(ssystem *sys, double *sgm) {
  int i, j;
  int nPnls = sys->nPnls, nChar = sys->nChar, qOrder=sys->maxQuadOrder;
  double *intgr, fac;
  panel *pnl;
  static int warnedGpuRHS = 0;

  fac = fourPiI/epsilon1;

  if (sys->gpuMode > 0 && gpuSetupRHS(sys, qOrder, fac, sgm)) {
    return;
  }
  if (sys->gpuMode > 0 && gpuBackendAvailable() && !warnedGpuRHS) {
    printf("GPU RHS path unavailable; using CPU setupRHS.\n");
    warnedGpuRHS = 1;
  }

  /* triangles order for Direct */
  for ( i=0, pnl=sys->pnlLst; pnl!=NULL; pnl=pnl->nextC, i++ ) {
    sgm[i] = 0.0; sgm[nPnls+i] = 0.0;
    for ( j=0; j<nChar; j++ ) {
      intgr=panelRHS(qOrder, pnl, &sys->pos[3*j]);
      sgm[i] += sys->chr[j]*intgr[0];
      sgm[i+nPnls] += sys->chr[j]*intgr[1];
    }
    sgm[i] *= fac;
    sgm[nPnls+i] *= fac;
  }
} /* setupRHS */



int main(int nargs, char *argv[]){
  char panelfile[80], density[80];
  int order=-1, image=0, refineLev=0, numSurfOne=1;
  int i, j, k, n, nPnls, nChar;
  int numItr=100, arnoldiSz=30, ldw, ldh;
  panel *inputLst, *pnl;
  cube *cb;
  double tolpar=1.0e-4, para=332.0716;
  double *sgm, *pot, *GMRES_work, *GMRES_h, ptl;
  static int info;

  double start_t, end_t;
  double stage_t0, loadPanel_t, gkInit_t, setupFMM_t_local;
  double setupPC_t, setupRHS_t, gmres_t, treecode_t;

  CALLOC(sys, 1, ssystem);
  set_benchmark_thread_defaults();
  sys->height = 2;
  sys->maxSepRatio = 0.8;
  sys->maxQuadOrder = 1;
  sys->nKerl = 4;
  sys->depth = 5;
  sys->mesh_flag = 1;
  sys->benchmarkMode = 0;
  sys->gpuMode = -1;
  sys->debugCompareApply = 0;
  sys->debugComparePrecond = 0;
  sys->matvecMode = 0;
  sys->gpuQ2MMode = 0;
  sys->gpuNearfieldMode = 1;
  sys->precondCacheMode = 2;
  sprintf(density,"1");
  double bulk_strength = 0.15;
  //kappa = sqrt(8.430325455*bulk_strength/epsilon2);
  kappa = 0.1257;

  /* parse the command line */
  panelfile[0] = 0;
  for ( i=1; i<nargs; i++ )
    if ( argv[i][0] == '-' )
      switch ( argv[i][1] ) {
        case 'h':
          print_usage(argv[0]);
          return 0;
        case 'S': sys->maxSepRatio = atof( argv[i]+3 );
          break;
        case 'o': tolpar = atof( argv[i]+3 );
          break;
        case 'p':
          if ( argv[i][2] == '=' ) order = atoi( argv[i]+3 );
          if ( argv[i][2] == 'm' ) orderMom = atoi( argv[i]+4 );
          break;
        case 'q':
          sys->maxQuadOrder = atoi( argv[i]+3 );
          break;
        case 't': sys->depth = atoi( argv[i]+3 );
          break;
        case 'H': sys->height = atoi( argv[i]+3 );
          break;
        case 'd': strcpy(density,argv[i]+3);
          break;
        case 'e':
          if ( argv[i][4] == '1' ) epsilon1 = atof( argv[i]+6 );
          if ( argv[i][4] == '2' ) epsilon2 = atof( argv[i]+6 );
          break;
        case 'k': kappa = atof( argv[i]+3 );
          break;
        case 'm': sys->mesh_flag = atoi( argv[i]+3 );
          break;
        case 'B': sys->benchmarkMode = atoi( argv[i]+3 );
          break;
        case 'g': sys->gpuMode = atoi( argv[i]+3 );
          break;
        case 'c': sys->debugCompareApply = atoi( argv[i]+3 );
          break;
        case 'C': sys->debugComparePrecond = atoi( argv[i]+3 );
          break;
        case 'r': sys->matvecMode = atoi( argv[i]+3 );
          break;
        case 'Q': sys->gpuQ2MMode = atoi( argv[i]+3 );
          break;
        case 'G': sys->gpuNearfieldMode = atoi( argv[i]+3 );
          break;
        case 'P': sys->precondCacheMode = atoi( argv[i]+3 );
          break;
      }
    else {
      strcpy(panelfile,argv[i]);
    }

  epsilon = epsilon2/epsilon1;
  if ( panelfile[0] == 0 ) {
    printf("\n Name of the panel file > ");
    if ( scanf("%s",panelfile) < 1 ) {
      printf("PDB name input failed\n");
      exit(0);
    }
  }
  if ( sys->depth < 0 ) {
    printf("Select tree depth > ");
    if ( scanf("%d", &sys->depth) < 1 ) {
      printf("PDB density input failed\n");
      exit(0);
    }
    if( sys->depth < 1 ) {
      printf("Bad tree depth: %d\n", sys->depth );
      exit(0);
    }
  }
  if ( sys->height < 0 ) {
    printf("Bad FMM height: %d\n", sys->height );
    exit(0);
  }
  if ( sys->height > sys->depth ) {
    printf("Bad FMM level range: height=%d depth=%d\n", sys->height, sys->depth );
    exit(0);
  }
  //printf("PDB id: %s, MSMS density: %s\n", panelfile, density);
  if (sys->gpuMode < 0) {
    sys->gpuMode = gpuBackendAvailable() ? 1 : 0;
  }

  if (sys->benchmarkMode > 0) {
    printf("----------------------------\n");
    if (sys->matvecMode == 0) {
      printf("FMM variables: nLev=%d height=%d ord=%d SepRat=%lg qOrd=%d\n",
        sys->depth, sys->height, order, sys->maxSepRatio, sys->maxQuadOrder );
    } else {
      printf("Direct baseline mode enabled (no FMM matvec)\n");
    }
    printf("GMRES variables: tol=%1.e arnoldiSz=%d maxIt=%d\n",
      tolpar, arnoldiSz, numItr);
    printf("kappa=%f, eps1=%f, eps2=%f\n", kappa, epsilon1, epsilon2);
    printf("GPU mode=%d (0=CPU, 1=GPU)\n", sys->gpuMode);
    printf("Matvec mode=%d (0=FMM, 1=direct GPU baseline, 2=direct CPU baseline)\n",
           sys->matvecMode);
    if (sys->matvecMode == 0) {
      printf("GPU Q2M mode=%d (0=CPU default, 1=GPU debug)\n", sys->gpuQ2MMode);
      printf("GPU nearfield mode=%d (0=interaction, 1=destination-leaf)\n", sys->gpuNearfieldMode);
    }
    printf("Preconditioner mode=%d (0=original, 1=cached-blocks, 2=cached-LU)\n", sys->precondCacheMode);
    if (sys->debugCompareApply > 0 || sys->debugComparePrecond > 0) {
      printf("Debug compare flags: apply=%d precond=%d\n",
             sys->debugCompareApply, sys->debugComparePrecond);
    }
  }
  //printf("----------------------------\n");


  /*
   * get panels by msms from pqr
   * or use the panel on sphere test example
   */
  start_t = wall_seconds();
  stage_t0 = start_t;
  inputLst = loadPanel(panelfile, density, &nPnls, sys);
  loadPanel_t = wall_seconds() - stage_t0;
  sys->pnlOLst = inputLst;

  stage_t0 = wall_seconds();
  gkInit(sys, inputLst, order, orderMom);
  gkInit_t = wall_seconds() - stage_t0;

  CALLOC(sgm, 2*nPnls, double);
  CALLOC(pot, 2*nPnls, double);

  stage_t0 = wall_seconds();
  setupFMM(sys);
  setupFMM_t_local = wall_seconds() - stage_t0;
  if (sys->matvecMode != 0) {
    /* Direct matvec modes still use treecode/FMM-side data for postprocessing. */
    buildPanelIndexDirect(sys);
  }
  stage_t0 = wall_seconds();
  setupPreconditioning(sys);
  setupPC_t = wall_seconds() - stage_t0;

  stage_t0 = wall_seconds();
  setupRHS(sys, sgm);
  setupRHS_t = wall_seconds() - stage_t0;
  if (sys->debugCompareApply > 0) {
    compareApplyFMMOnce(sys, sgm);
  }
  if (sys->debugComparePrecond > 0) {
    comparePrecondOnce(sys, sgm);
  }
  for ( i=0; i<2*sys->nPnls; i++ ) pot[i] = sgm[i];

  MtV = MtVmain;
  PtV = PtVmain;
  ldw = 2*nPnls;
  ldh = arnoldiSz+1;

  CALLOC(GMRES_work, ldw*(arnoldiSz+4), double);
  CALLOC(GMRES_h, ldh*(arnoldiSz+2), double);

  resetFmmMatvecStats();
  resetGmresStats();
  stage_t0 = wall_seconds();
  gmres(ldw, pot, sgm, arnoldiSz, GMRES_work, ldw, GMRES_h, ldh,
        &numItr, &tolpar, MtV, PtV, &info);
  gmres_t = wall_seconds() - stage_t0;

  stage_t0 = wall_seconds();
  applyTreecode( sys, sgm, &ptl );
  treecode_t = wall_seconds() - stage_t0;
  ptl *= twoPi*para;
  end_t = wall_seconds() - start_t;
  printf("ttl time: %f, gmres-its=%d\n", end_t, numItr);
  printf("solvation energy: %f\n", ptl);
  if (sys->benchmarkMode > 0) {
    printf("Top-level stage times (s): loadPanel=%.6f gkInit=%.6f setupFMM=%.6f setupPC=%.6f setupRHS=%.6f gmres=%.6f treecode=%.6f\n",
           loadPanel_t, gkInit_t, setupFMM_t_local, setupPC_t, setupRHS_t, gmres_t, treecode_t);
  }
  printSetupFmmStats();
  printPrecondSetupStats();
  printGmresStats(gmres_t);
  if (sys->benchmarkMode > 0 && sys->matvecMode == 0) {
    printFmmMatvecStats();
  } else if (sys->benchmarkMode > 0) {
    printf("Direct baseline run: FMM stage stats omitted.\n");
    if (sys->gpuMode > 0 && sys->matvecMode == 1) {
      printDirectMatvecStats();
    }
  }

}



/*
 * Matrix times Vector, subroutine of the iterative solver
 * the vector sgm and the result pot are ordered contiguously within cubes
 */
int MtVmain(double *alpha, double *sgm, double *beta, double *pot) {
  int i, lev, inc=1, nPnls = sys->nPnls;
  cube *cb;
  panel *pnl;
  double scale1, scale2, inv_beta;
  double callStart, callEnd, applyStart, applyEnd;
  static int warnedDirectGpu = 0;
  static int warnedDirectCpu = 0;

  scale1 = (1.0+epsilon)/2.0*(*alpha);
  scale2 = (1.0+1.0/epsilon)/2.0*(*alpha);

  callStart = wall_seconds();
  inv_beta = -(*beta);
  applyStart = wall_seconds();
  if (sys->matvecMode == 1) {
    if (!gpuDirectApply(sys, *alpha, inv_beta, sgm, pot)) {
      if (!warnedDirectGpu) {
        printf("Direct GPU matvec unavailable; using FMM path.\n");
        warnedDirectGpu = 1;
      }
      applyFMM(sys, alpha, sgm, &inv_beta, pot);
    }
  } else if (sys->matvecMode == 2) {
    if (!cpuDirectApply(sys, *alpha, inv_beta, sgm, pot)) {
      if (!warnedDirectCpu) {
        printf("Direct CPU matvec unavailable; using FMM path.\n");
        warnedDirectCpu = 1;
      }
      applyFMM(sys, alpha, sgm, &inv_beta, pot);
    }
  } else {
    applyFMM(sys, alpha, sgm, &inv_beta, pot);
  }
  applyEnd = wall_seconds();
  for (  i=0, pnl=sys->pnlLst; pnl!=NULL; pnl=pnl->nextC, i++ ) {
    pot[i] = (scale1*pnl->area*sgm[i]-pot[i]);
    pot[i+nPnls] = scale2*pnl->area*sgm[i+nPnls]-pot[i+nPnls];
  }
  callEnd = wall_seconds();

  mtvCalls++;
  mtvApplyFMMTime += (applyEnd - applyStart);
  mtvTotalTime += (callEnd - callStart);

  return 0;
} /* MtVmain */

int PtVmain(double *pot, double *sgm) {
  if (sys->precondCacheMode > 1) {
    return PtVfmmCachedLU(pot, sgm);
  }
  if (sys->precondCacheMode > 0) {
    return PtVfmmCached(pot, sgm);
  }
  return PtVfmm(pot, sgm);
}
