#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

struct ssystem;

#ifdef __cplusplus
extern "C" {
#endif

int gpuBackendAvailable(void);
int gpuNearfieldApply(struct ssystem *sys, double alpha, const double *sgm, double *pot);
int gpuDirectApply(struct ssystem *sys, double alpha, double beta, const double *sgm, double *pot);

#ifdef __cplusplus
}
#endif

#endif
