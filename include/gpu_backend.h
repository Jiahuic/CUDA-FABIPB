#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

struct ssystem;

#ifdef __cplusplus
extern "C" {
#endif

int gpuBackendAvailable(void);
int gpuNearfieldApply(struct ssystem *sys, double alpha, const double *sgm, double *pot);
int gpuDirectApply(struct ssystem *sys, double alpha, double beta, const double *sgm, double *pot);
int gpuM2LApply(struct ssystem *sys);
int gpuQ2MApply(struct ssystem *sys, const double *sgm);
int gpuL2PApply(struct ssystem *sys, double alpha, double beta, double *pot);
int gpuSetupRHS(struct ssystem *sys, int qOrder, double fac, double *sgm);
int gpuBuildPrecondDisjointBlock(struct panel **panels, int nPanels,
                                 const int *dstLocal, const int *srcLocal,
                                 int nPairs, double *block);

#ifdef __cplusplus
}
#endif

#endif
