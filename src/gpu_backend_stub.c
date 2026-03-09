#include "gpu_backend.h"
#include "gk.h"

int gpuBackendAvailable(void) {
  return 0;
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
