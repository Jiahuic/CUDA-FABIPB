#ifndef GPU_BACKEND_CUDA_INTERNAL_H
#define GPU_BACKEND_CUDA_INTERNAL_H

#include "gpu_backend.h"
#include "gk.h"
#include "gkGlobal.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <limits.h>
#include <cstring>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <thread>
#include <vector>
#include <mutex>

extern "C" double *panelIA0(panel *pnlX, panel *pnlY);
extern "C" int nrCommonVtx(panel *p, panel *q, int *idxX, int *idxY);
extern "C" int rhsChargeExpansionOrder(ssystem *sys, int level);
extern "C" void initRhsTreeWorkspace(ssystem *sys, RhsTreeWorkspace *ws);
extern "C" void freeRhsTreeWorkspace(RhsTreeWorkspace *ws);
extern "C" void setupDerivsWorkspace(ssystem *sys, RhsTreeWorkspace *ws,
                                     int order, const double *x);
extern "C" int ***idx3;
extern "C" void kernelKER4(double *x, double *y);
extern "C" void (*kernel)(double *x, double *y);
extern "C" void setupDerivs(int order, double *x);
extern "C" double **dG0;
extern "C" double **dGk;
extern "C" int *sgn3;
extern "C" double kappa;
extern "C" double epsilon;
extern "C" double **Q2M0;
extern "C" double **Q2M1;
extern "C" double **L2P0;
extern "C" double **L2P1;
extern "C" double **tLegA;
extern "C" double **wLegA;

struct NearPanelGeom {
  double vtx[3][3];
  double a0[3];
  double a1[3];
  double a2[3];
  double normal[3];
  double area;
};

#endif
