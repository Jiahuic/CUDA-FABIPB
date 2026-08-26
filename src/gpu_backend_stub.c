#include "gpu_backend.h"
#include "gk.h"

int gpuBackendAvailable(void) {
  return 0;
}

const char *gpuNearfieldLastError(void) {
  return "CUDA backend not compiled";
}

const char *gpuM2LLastError(void) {
  return "CUDA backend not compiled";
}

int gpuNearfieldApply(ssystem *sys, double alpha, const double *sgm, double *pot) {
  (void)sys;
  (void)alpha;
  (void)sgm;
  (void)pot;
  return 0;
}

int gpuDirectApply(ssystem *sys, double alpha, double beta, const double *sgm, double *pot) {
  (void)sys;
  (void)alpha;
  (void)beta;
  (void)sgm;
  (void)pot;
  return 0;
}

int gpuM2LApply(ssystem *sys) {
  (void)sys;
  return 0;
}

int gpuQ2MApply(ssystem *sys, const double *sgm) {
  (void)sys;
  (void)sgm;
  return 0;
}

int gpuL2PApply(ssystem *sys, double alpha, double beta, double *pot) {
  (void)sys;
  (void)alpha;
  (void)beta;
  (void)pot;
  return 0;
}

void gpuReleaseMatvecCaches(void) {
}

void gpuReleaseChargeTreeCache(void) {
}

int gpuSetupRHS(ssystem *sys, int qOrder, double fac, double *sgm) {
  (void)sys;
  (void)qOrder;
  (void)fac;
  (void)sgm;
  return 0;
}

int gpuBuildPrecondDisjointBlock(panel **panels, int nPanels,
                                 const int *dstLocal, const int *srcLocal,
                                 int nPairs, double *block) {
  (void)panels;
  (void)nPanels;
  (void)dstLocal;
  (void)srcLocal;
  (void)nPairs;
  (void)block;
  return 0;
}

int gpuBuildPrecondPairsBatched(ssystem *sys,
                                const int *pairSrc, const int *pairDst,
                                long long nPairs,
                                double *k0, double *k1, double *k2, double *k3) {
  (void)sys;
  (void)pairSrc;
  (void)pairDst;
  (void)nPairs;
  (void)k0;
  (void)k1;
  (void)k2;
  (void)k3;
  return 0;
}

int gpuPanelChargeTreeEnergy(ssystem *sys, const double *sgm, double *pot) {
  (void)sys; (void)sgm; (void)pot;
  return 0;
}

int gpuRhsReuseCoulombDot(ssystem *sys, const double *rhs,
                          const double *sgm, double *dot) {
  (void)sys; (void)rhs; (void)sgm; (void)dot;
  return 0;
}

int gpuRhsReusePanelChargeTreeEnergy(ssystem *sys, const double *rhs,
                                     const double *sgm, double *pot) {
  (void)sys; (void)rhs; (void)sgm; (void)pot;
  return 0;
}

int gpuChargeTreeRHS(ssystem *sys, int qOrder, double fac, double *sgm) {
  (void)sys; (void)qOrder; (void)fac; (void)sgm;
  return 0;
}
