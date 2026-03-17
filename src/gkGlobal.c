#include <math.h>
#include <stdio.h>
#include <time.h>
#include "gkGlobal.h"
#include "gk.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double gkInitTime, setupFMMTime, setupRHSTime, gmresTime;
double solveTimeNoPC, solveTimePC;
double setupQ2PTime, setupQ2MTime, setupM2LTime;
double setupFmmLeafTime, setupFmmCubeAllocTime, setupFmmLayoutTime;
double setupFmmApplyLayoutTime, setupFmmPanelIndexTime, setupFmmCubeLayoutTime;
double setupFmmM2LPairTime, setupFmmM2LGroupTime;
double fmmQ2MTime, fmmM2MTime, fmmM2LTime, fmmL2LTime, fmmL2PTime, fmmNearTime;
double fmmNearGpuBuildTime, fmmNearGpuH2DTime, fmmNearGpuKernelTime, fmmNearGpuD2HTime;
double fmmNearGpuMetaTime, fmmNearGpuCoeffTime, fmmNearGpuUploadTime;
double directGpuBuildTime, directGpuCoeffTime, directGpuStoreTime, directGpuH2DTime, directGpuKernelTime, directGpuD2HTime;
double mtvApplyFMMTime, mtvTotalTime;
double gmresMatvecTime, gmresPsolveTime, gmresBasisTime, gmresUpdateTime, gmresResidualTime;
double pcAssembleTime, pcFactorTime, pcSolveTime, pcScatterTime;
long mtvCalls;
long gmresMatvecCalls, gmresPsolveCalls;

/* memory counters */
long memcount=0;    /* total memory */
long memPVE=0;      /* panels, vertices, edges */
long memCUBES=0;    /* cube tree */
long memQ2P=0;      /* Q2P transformations */
long memQ2M=0;      /* Q2M/L2P transformations */
long memM2L=0;      /* M2L transformations */
long memSOLVER=0;   /* allocated by solver */
long memMISC=0;     /* all other stuff */

/* matrix entry counters */
long numKernRealEval=0, numKernCplxEval=0;

/* misc constants */
double fourPi = 4*M_PI, fourPiI = 1.0/(4*M_PI);
double twoPi = 2*M_PI, twoPiI = 1.0/(2*M_PI);
double zero=0.0, one=1.0;
int oneI = 1;

char hChr='T';             /* transpose */
char nChr='N';             /* normal */
extern ssystem *sys;

void resetFmmMatvecStats(void) {
  fmmQ2MTime = 0.0;
  fmmM2MTime = 0.0;
  fmmM2LTime = 0.0;
  fmmL2LTime = 0.0;
  fmmL2PTime = 0.0;
  fmmNearTime = 0.0;
  fmmNearGpuBuildTime = 0.0;
  fmmNearGpuH2DTime = 0.0;
  fmmNearGpuKernelTime = 0.0;
  fmmNearGpuD2HTime = 0.0;
  fmmNearGpuMetaTime = 0.0;
  fmmNearGpuCoeffTime = 0.0;
  fmmNearGpuUploadTime = 0.0;
  directGpuBuildTime = 0.0;
  directGpuCoeffTime = 0.0;
  directGpuStoreTime = 0.0;
  directGpuH2DTime = 0.0;
  directGpuKernelTime = 0.0;
  directGpuD2HTime = 0.0;
  mtvApplyFMMTime = 0.0;
  mtvTotalTime = 0.0;
  mtvCalls = 0;
}

void printDirectMatvecStats(void) {
  double invCalls;
  double buildOther;

  if (sys == NULL || sys->benchmarkMode == 0) {
    return;
  }
  if (gmresMatvecCalls <= 0) {
    printf("Direct matvec stats: no calls recorded.\n");
    return;
  }

  invCalls = 1.0 / (double)gmresMatvecCalls;
  buildOther = directGpuBuildTime - directGpuCoeffTime;
  buildOther -= directGpuStoreTime;
  if (buildOther < 0.0) {
    buildOther = 0.0;
  }

  printf("GPU direct breakdown (s): build=%.6f calc=%.6f store=%.6f h2d=%.6f kernel=%.6f d2h=%.6f\n",
         directGpuBuildTime, directGpuCoeffTime, directGpuStoreTime, directGpuH2DTime,
         directGpuKernelTime, directGpuD2HTime);
  printf("GPU direct avg/call (ms): build=%.3f calc=%.3f store=%.3f h2d=%.3f kernel=%.3f d2h=%.3f\n",
         1.0e3 * directGpuBuildTime * invCalls,
         1.0e3 * directGpuCoeffTime * invCalls,
         1.0e3 * directGpuStoreTime * invCalls,
         1.0e3 * directGpuH2DTime * invCalls,
         1.0e3 * directGpuKernelTime * invCalls,
         1.0e3 * directGpuD2HTime * invCalls);
  printf("GPU direct build breakdown (s): calc=%.6f store=%.6f other=%.6f\n",
         directGpuCoeffTime, directGpuStoreTime, buildOther);
}

void printSetupFmmStats(void) {
  double total = setupFmmLeafTime + setupFmmCubeAllocTime + setupFmmLayoutTime;
  double layoutTracked = setupFmmApplyLayoutTime + setupFmmPanelIndexTime +
                         setupFmmCubeLayoutTime + setupFmmM2LPairTime +
                         setupFmmM2LGroupTime;

  if (sys == NULL || sys->benchmarkMode == 0) {
    return;
  }
  if (total <= 0.0) {
    return;
  }

  printf("setupFMM breakdown (s): leaf-transforms=%.6f cube-alloc=%.6f layouts=%.6f tracked=%.6f\n",
         setupFmmLeafTime, setupFmmCubeAllocTime, setupFmmLayoutTime, total);
  if (setupFmmLayoutTime > 0.0) {
    printf("setupFMM layout breakdown (s): apply=%.6f panel-index=%.6f cubes=%.6f m2l-pairs=%.6f m2l-groups=%.6f tracked=%.6f\n",
           setupFmmApplyLayoutTime, setupFmmPanelIndexTime, setupFmmCubeLayoutTime,
           setupFmmM2LPairTime, setupFmmM2LGroupTime, layoutTracked);
  }
}

void resetGmresStats(void) {
  gmresMatvecTime = 0.0;
  gmresPsolveTime = 0.0;
  gmresBasisTime = 0.0;
  gmresUpdateTime = 0.0;
  gmresResidualTime = 0.0;
  pcAssembleTime = 0.0;
  pcFactorTime = 0.0;
  pcSolveTime = 0.0;
  pcScatterTime = 0.0;
  gmresMatvecCalls = 0;
  gmresPsolveCalls = 0;
}

void printFmmMatvecStats(void) {
  double stageTotal;
  double invCalls;

  if (sys == NULL || sys->benchmarkMode == 0) {
    return;
  }
  if (mtvCalls <= 0) {
    printf("FMM matvec stats: no calls recorded.\n");
    return;
  }

  stageTotal = fmmQ2MTime + fmmM2MTime + fmmM2LTime + fmmL2LTime + fmmL2PTime + fmmNearTime;
  invCalls = 1.0 / (double)mtvCalls;

  printf("FMM matvec stats: calls=%ld total=%.6f s applyFMM=%.6f s\n",
         mtvCalls, mtvTotalTime, mtvApplyFMMTime);
  printf("FMM stage totals (s): Q2M=%.6f M2M=%.6f M2L=%.6f L2L=%.6f L2P=%.6f Near=%.6f Sum=%.6f\n",
         fmmQ2MTime, fmmM2MTime, fmmM2LTime, fmmL2LTime, fmmL2PTime, fmmNearTime, stageTotal);
  printf("FMM stage avg/call (ms): Q2M=%.3f M2M=%.3f M2L=%.3f L2L=%.3f L2P=%.3f Near=%.3f\n",
         1.0e3 * fmmQ2MTime * invCalls, 1.0e3 * fmmM2MTime * invCalls,
         1.0e3 * fmmM2LTime * invCalls, 1.0e3 * fmmL2LTime * invCalls,
         1.0e3 * fmmL2PTime * invCalls, 1.0e3 * fmmNearTime * invCalls);
  if (fmmNearGpuBuildTime > 0.0 || fmmNearGpuH2DTime > 0.0 ||
      fmmNearGpuKernelTime > 0.0 || fmmNearGpuD2HTime > 0.0) {
    printf("GPU nearfield breakdown (s): build=%.6f h2d=%.6f kernel=%.6f d2h=%.6f\n",
           fmmNearGpuBuildTime, fmmNearGpuH2DTime,
           fmmNearGpuKernelTime, fmmNearGpuD2HTime);
    printf("GPU nearfield avg/call (ms): build=%.3f h2d=%.3f kernel=%.3f d2h=%.3f\n",
           1.0e3 * fmmNearGpuBuildTime * invCalls,
           1.0e3 * fmmNearGpuH2DTime * invCalls,
           1.0e3 * fmmNearGpuKernelTime * invCalls,
           1.0e3 * fmmNearGpuD2HTime * invCalls);
    printf("GPU nearfield build breakdown (s): meta=%.6f coeff=%.6f upload=%.6f other=%.6f\n",
           fmmNearGpuMetaTime, fmmNearGpuCoeffTime, fmmNearGpuUploadTime,
           fmmNearGpuBuildTime - (fmmNearGpuMetaTime + fmmNearGpuCoeffTime + fmmNearGpuUploadTime));
  }
}

void printGmresStats(double gmresWallTime) {
  double accounted;
  double other;

  if (sys == NULL || sys->benchmarkMode == 0) {
    return;
  }
  accounted = gmresMatvecTime + gmresPsolveTime + gmresBasisTime +
              gmresUpdateTime + gmresResidualTime;
  other = gmresWallTime - accounted;
  if (other < 0.0) {
    other = 0.0;
  }

  printf("GMRES breakdown (s): matvec=%.6f psolve=%.6f basis=%.6f update=%.6f residual=%.6f other=%.6f\n",
         gmresMatvecTime, gmresPsolveTime, gmresBasisTime,
         gmresUpdateTime, gmresResidualTime, other);
  printf("GMRES call counts: matvec=%ld psolve=%ld\n",
         gmresMatvecCalls, gmresPsolveCalls);
  if (gmresPsolveTime > 0.0) {
    printf("Preconditioner breakdown (s): assemble=%.6f factor=%.6f solve=%.6f scatter=%.6f other=%.6f\n",
           pcAssembleTime, pcFactorTime, pcSolveTime, pcScatterTime,
           gmresPsolveTime - (pcAssembleTime + pcFactorTime + pcSolveTime + pcScatterTime));
  }
}
