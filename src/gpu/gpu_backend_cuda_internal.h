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

/* One host-computed near-field coefficient with the interaction it belongs to.
 * Kept as a single record rather than five parallel arrays: appending to five
 * vectors from every build thread put ~160 write streams in flight and cost
 * more than the PCIe transfer the compaction saves. */
typedef struct {
  long long idx;
  double k0;
  double k1;
  double k2;
  double k3;
} NearSpecialCoeff;

struct NearPanelGeom {
  double vtx[3][3];
  double a0[3];
  double a1[3];
  double a2[3];
  double normal[3];
  double area;
};

struct NearfieldGpuCache {
  const ssystem *sys;
  int nPnls;
  int nearfieldMode;
  int streaming;
  long long nInteractions;
  long long chunkCapacity;
  long long caseDisjointCount;
  long long caseOneCommonCount;
  long long caseTwoCommonCount;
  long long caseTwoCommonRevCount;
  long long caseSelfCount;
  NearPanelGeom *h_panels;
  int *h_src;
  int *h_dst;
  int *h_pairSrcCount;
  long long *h_pairInteractionOffset;
  int *h_leafPanelStart;
  int *h_leafPanelCount;
  int *h_leafPairOffset;
  double *h_k0;
  double *h_k1;
  double *h_k2;
  double *h_k3;
  int *d_src;
  int *d_dst;
  int *d_pairSrcCount;
  long long *d_pairInteractionOffset;
  int *d_leafPanelStart;
  int *d_leafPanelCount;
  int *d_leafPairOffset;
  NearPanelGeom *d_panels;
  double *d_k0;
  double *d_k1;
  double *d_k2;
  double *d_k3;
  double *d_sgm;
  double *d_pot;
  int streamPairCount;
  long long *d_streamPairOffset;
  int *d_streamPairSrcStart;
  int *d_streamPairSrcCount;
  int *d_streamPairDstStart;
  int h_srcPinned;
  int h_dstPinned;
  int h_k0Pinned;
  int h_k1Pinned;
  int h_k2Pinned;
  int h_k3Pinned;
  /* Sparse coefficient upload for the resident build (see T5 in
   * gpu_nearfield_cuda.inc). When the on-device disjoint builder is enabled it
   * writes every disjoint interaction itself, so only the geometrically
   * special pairs -- around 5% of interactions -- have to cross PCIe. These
   * hold that compacted set: an interaction index plus its four coefficients. */
  int sparseCoeffUpload;
  long long nSpecialCoeff;
  NearSpecialCoeff *d_special;

  int specialCacheEnabled;
  int specialCacheValid;
  long long specialCacheChunks;
  std::vector<long long> *specialOffset;
  std::vector<int> *cacheSrc;
  std::vector<int> *cacheDst;
  std::vector<double> *cacheK0;
  std::vector<double> *cacheK1;
  std::vector<double> *cacheK2;
  std::vector<double> *cacheK3;
};

struct M2LGpuCache {
  const ssystem *sys;
  int streaming;
  int nCubes;
  int nPairs;
  int nGroups;
  int maxOrder;
  int maxIdxDim;
  long long totalPairCoeff;
  int totalCubeCoeff;
  int *h_pairSrc;
  int *h_pairCoeffOffset;
  int *h_groupStart;
  int *h_groupCount;
  int *h_groupDst;
  int *h_groupOrder;
  int *h_cubeCoeffOffset;
  int *h_cubeNMom;
  int *h_idxI1;
  int *h_idxI2;
  int *h_idxI3;
  int *h_idx3Flat;
  double *h_g0;
  double *h_gk;
  double *h_momPot;
  double *h_momDpdn;
  double *h_lec1;
  double *h_lec2;
  double *h_lec3;
  double *h_lec4;
  int *d_pairSrc;
  int *d_pairCoeffOffset;
  int *d_groupStart;
  int *d_groupCount;
  int *d_groupDst;
  int *d_groupOrder;
  int *d_cubeCoeffOffset;
  int *d_cubeNMom;
  int *d_idxI1;
  int *d_idxI2;
  int *d_idxI3;
  int *d_idx3Flat;
  int *d_sgn3;
  double *d_g0;
  double *d_gk;
  double *d_momPot;
  double *d_momDpdn;
  double *d_lec1;
  double *d_lec2;
  double *d_lec3;
  double *d_lec4;
  int streamPairCapacity;
  int streamGroupCapacity;
  int streamCoeffCapacity;
};

struct LeafTransformGpuCache {
  const ssystem *sys;
  int nLeaves;
  int nMom;
  long long totalMatrixEntries;
  int *h_leafMatrixOffset;
  double *h_q2m0;
  double *h_q2m1;
  /* No l2p arrays: L2P0/L2P1 alias Q2M0/Q2M1 (fmm.c), so the leaf transform
   * cache stores one copy and l2pLeafKernel is handed the q2m pointers. */
  double *h_sgm;
  double *h_pot;
  double *h_momPot;
  double *h_momDpdn;
  double *h_lec1;
  double *h_lec2;
  double *h_lec3;
  double *h_lec4;
  int *d_leafPanelStart;
  int *d_leafPanelCount;
  int *d_leafMatrixOffset;
  double *d_q2m0;
  double *d_q2m1;
  double *d_sgm;
  double *d_pot;
  double *d_momPot;
  double *d_momDpdn;
  double *d_lec1;
  double *d_lec2;
  double *d_lec3;
  double *d_lec4;
};

struct DirectGpuCache {
  const ssystem *sys;
  int nPnls;
  long long nInteractions;
  int mode;
  int blockDstCount;
  double *h_k0;
  double *h_k1;
  double *h_k2;
  double *h_k3;
  double *d_k0;
  double *d_k1;
  double *d_k2;
  double *d_k3;
  double *d_sgm;
  double *d_pot;
};

struct PrecondGpuCache {
  int panelCapacity;
  int pairCapacity;
  NearPanelGeom *h_panels;
  int *h_src;
  int *h_dst;
  double *h_k0;
  double *h_k1;
  double *h_k2;
  double *h_k3;
  NearPanelGeom *d_panels;
  int *d_src;
  int *d_dst;
  double *d_k0;
  double *d_k1;
  double *d_k2;
  double *d_k3;
};

struct RhsPanelGeom {
  double v0[3];
  double a0[3];
  double a2[3];
  double normal[3];
  double area;
};

struct RhsGpuCache {
  const ssystem *sys;
  int nPnls;
  int nChar;
  int qOrder;
  RhsPanelGeom *h_panels;
  double *h_chrPos;
  double *h_chrVal;
  RhsPanelGeom *d_panels;
  double *d_chrPos;
  double *d_chrVal;
  double *d_sgm;
};

#endif
