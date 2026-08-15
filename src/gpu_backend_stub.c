#include "gpu_backend.h"
#include "gk.h"

int gpuBackendAvailable(void) {
  return 0;
}

const char *gpuNearfieldLastError(void) {
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

int gpuPanelChargeTreeEnergy(ssystem *sys, const double *sgm, double *pot) {
  (void)sys; (void)sgm; (void)pot;
  return 0;
}
