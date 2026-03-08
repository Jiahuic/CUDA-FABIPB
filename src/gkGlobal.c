#include <math.h>
#include <stdio.h>
#include <time.h>
#include "gkGlobal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double gkInitTime, setupFMMTime, setupRHSTime, gmresTime;
double solveTimeNoPC, solveTimePC;
double setupQ2PTime, setupQ2MTime, setupM2LTime;
double fmmQ2MTime, fmmM2MTime, fmmM2LTime, fmmL2LTime, fmmL2PTime, fmmNearTime;
double mtvApplyFMMTime, mtvTotalTime;
long mtvCalls;

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

void resetFmmMatvecStats(void) {
  fmmQ2MTime = 0.0;
  fmmM2MTime = 0.0;
  fmmM2LTime = 0.0;
  fmmL2LTime = 0.0;
  fmmL2PTime = 0.0;
  fmmNearTime = 0.0;
  mtvApplyFMMTime = 0.0;
  mtvTotalTime = 0.0;
  mtvCalls = 0;
}

void printFmmMatvecStats(void) {
  double stageTotal;
  double invCalls;

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
}
