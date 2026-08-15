#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

struct panel;
struct ssystem;

#ifdef __cplusplus
extern "C" {
#endif

int gpuBackendAvailable(void);
const char *gpuNearfieldLastError(void);
int gpuNearfieldApply(struct ssystem *sys, double alpha, const double *sgm, double *pot);
int gpuDirectApply(struct ssystem *sys, double alpha, double beta, const double *sgm, double *pot);
int gpuM2LApply(struct ssystem *sys);
int gpuQ2MApply(struct ssystem *sys, const double *sgm);
int gpuL2PApply(struct ssystem *sys, double alpha, double beta, double *pot);
int gpuSetupRHS(struct ssystem *sys, int qOrder, double fac, double *sgm);
/*
 * Post-solve solvation energy by walking the charge tree on the device, one
 * warp per panel. Returns 0 if the GPU path is unavailable, leaving the caller
 * to fall back to the threaded CPU evaluator.
 */
int gpuPanelChargeTreeEnergy(struct ssystem *sys, const double *sgm, double *pot);
int gpuBuildPrecondDisjointBlock(struct panel **panels, int nPanels,
                                 const int *dstLocal, const int *srcLocal,
                                 int nPairs, double *block);

#ifdef __cplusplus
}
#endif

#endif
