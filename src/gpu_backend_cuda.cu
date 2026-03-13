#include "gpu_backend.h"
#include "gk.h"
#include "gkGlobal.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <thread>
#include <vector>

extern "C" double *panelIA0(panel *pnlX, panel *pnlY);
extern "C" void kernelKER4(double *x, double *y);
extern "C" void (*kernel)(double *x, double *y);
extern "C" void setupDerivs(int order, double *x);
extern "C" double **dG0;
extern "C" double **dGk;
extern "C" int *sgn3;
extern "C" double epsilon;
extern "C" double **Q2M0;
extern "C" double **Q2M1;
extern "C" double **L2P0;
extern "C" double **L2P1;
extern "C" double **tLegA;
extern "C" double **wLegA;

namespace {
double wall_seconds_cuda_local() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

struct NearfieldGpuCache {
  const ssystem *sys;
  int nPnls;
  int nearfieldMode;
  long long nInteractions;
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
  double *d_k0;
  double *d_k1;
  double *d_k2;
  double *d_k3;
  double *d_sgm;
  double *d_pot;
};

NearfieldGpuCache gNear = {};

struct M2LGpuCache {
  const ssystem *sys;
  int nCubes;
  int nPairs;
  int nGroups;
  int maxOrder;
  int maxIdxDim;
  long long totalPairCoeff;
  int totalCubeCoeff;
  int *h_pairCoeffOffset;
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
};

M2LGpuCache gM2L = {};

struct LeafTransformGpuCache {
  const ssystem *sys;
  int nLeaves;
  int nMom;
  long long totalMatrixEntries;
  int *h_leafMatrixOffset;
  double *h_q2m0;
  double *h_q2m1;
  double *h_l2p0;
  double *h_l2p1;
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
  double *d_l2p0;
  double *d_l2p1;
  double *d_sgm;
  double *d_pot;
  double *d_momPot;
  double *d_momDpdn;
  double *d_lec1;
  double *d_lec2;
  double *d_lec3;
  double *d_lec4;
};

LeafTransformGpuCache gLeaf = {};

struct DirectGpuCache {
  const ssystem *sys;
  int nPnls;
  long long nInteractions;
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

DirectGpuCache gDirect = {};

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

RhsGpuCache gRhs = {};

__constant__ double c_rhsTLeg[10];
__constant__ double c_rhsWLeg[10];

void freeNearfieldCache() {
  free(gNear.h_src);
  free(gNear.h_dst);
  free(gNear.h_pairSrcCount);
  free(gNear.h_pairInteractionOffset);
  free(gNear.h_leafPanelStart);
  free(gNear.h_leafPanelCount);
  free(gNear.h_leafPairOffset);
  free(gNear.h_k0);
  free(gNear.h_k1);
  free(gNear.h_k2);
  free(gNear.h_k3);
  gNear.h_src = NULL;
  gNear.h_dst = NULL;
  gNear.h_pairSrcCount = NULL;
  gNear.h_pairInteractionOffset = NULL;
  gNear.h_leafPanelStart = NULL;
  gNear.h_leafPanelCount = NULL;
  gNear.h_leafPairOffset = NULL;
  gNear.h_k0 = NULL;
  gNear.h_k1 = NULL;
  gNear.h_k2 = NULL;
  gNear.h_k3 = NULL;

  cudaFree(gNear.d_src);
  cudaFree(gNear.d_dst);
  cudaFree(gNear.d_pairSrcCount);
  cudaFree(gNear.d_pairInteractionOffset);
  cudaFree(gNear.d_leafPanelStart);
  cudaFree(gNear.d_leafPanelCount);
  cudaFree(gNear.d_leafPairOffset);
  cudaFree(gNear.d_k0);
  cudaFree(gNear.d_k1);
  cudaFree(gNear.d_k2);
  cudaFree(gNear.d_k3);
  cudaFree(gNear.d_sgm);
  cudaFree(gNear.d_pot);
  gNear.d_src = NULL;
  gNear.d_dst = NULL;
  gNear.d_pairSrcCount = NULL;
  gNear.d_pairInteractionOffset = NULL;
  gNear.d_leafPanelStart = NULL;
  gNear.d_leafPanelCount = NULL;
  gNear.d_leafPairOffset = NULL;
  gNear.d_k0 = NULL;
  gNear.d_k1 = NULL;
  gNear.d_k2 = NULL;
  gNear.d_k3 = NULL;
  gNear.d_sgm = NULL;
  gNear.d_pot = NULL;

  gNear.sys = NULL;
  gNear.nPnls = 0;
  gNear.nearfieldMode = 0;
  gNear.nInteractions = 0;
}

void freeM2LCache() {
  free(gM2L.h_pairCoeffOffset);
  free(gM2L.h_groupDst);
  free(gM2L.h_groupOrder);
  free(gM2L.h_cubeCoeffOffset);
  free(gM2L.h_cubeNMom);
  free(gM2L.h_idxI1);
  free(gM2L.h_idxI2);
  free(gM2L.h_idxI3);
  free(gM2L.h_idx3Flat);
  free(gM2L.h_g0);
  free(gM2L.h_gk);
  free(gM2L.h_momPot);
  free(gM2L.h_momDpdn);
  free(gM2L.h_lec1);
  free(gM2L.h_lec2);
  free(gM2L.h_lec3);
  free(gM2L.h_lec4);
  gM2L.h_pairCoeffOffset = NULL;
  gM2L.h_groupDst = NULL;
  gM2L.h_groupOrder = NULL;
  gM2L.h_cubeCoeffOffset = NULL;
  gM2L.h_cubeNMom = NULL;
  gM2L.h_idxI1 = NULL;
  gM2L.h_idxI2 = NULL;
  gM2L.h_idxI3 = NULL;
  gM2L.h_idx3Flat = NULL;
  gM2L.h_g0 = NULL;
  gM2L.h_gk = NULL;
  gM2L.h_momPot = NULL;
  gM2L.h_momDpdn = NULL;
  gM2L.h_lec1 = NULL;
  gM2L.h_lec2 = NULL;
  gM2L.h_lec3 = NULL;
  gM2L.h_lec4 = NULL;

  cudaFree(gM2L.d_pairSrc);
  cudaFree(gM2L.d_pairCoeffOffset);
  cudaFree(gM2L.d_groupStart);
  cudaFree(gM2L.d_groupCount);
  cudaFree(gM2L.d_groupDst);
  cudaFree(gM2L.d_groupOrder);
  cudaFree(gM2L.d_cubeCoeffOffset);
  cudaFree(gM2L.d_cubeNMom);
  cudaFree(gM2L.d_idxI1);
  cudaFree(gM2L.d_idxI2);
  cudaFree(gM2L.d_idxI3);
  cudaFree(gM2L.d_idx3Flat);
  cudaFree(gM2L.d_sgn3);
  cudaFree(gM2L.d_g0);
  cudaFree(gM2L.d_gk);
  cudaFree(gM2L.d_momPot);
  cudaFree(gM2L.d_momDpdn);
  cudaFree(gM2L.d_lec1);
  cudaFree(gM2L.d_lec2);
  cudaFree(gM2L.d_lec3);
  cudaFree(gM2L.d_lec4);
  gM2L.d_pairSrc = NULL;
  gM2L.d_pairCoeffOffset = NULL;
  gM2L.d_groupStart = NULL;
  gM2L.d_groupCount = NULL;
  gM2L.d_groupDst = NULL;
  gM2L.d_groupOrder = NULL;
  gM2L.d_cubeCoeffOffset = NULL;
  gM2L.d_cubeNMom = NULL;
  gM2L.d_idxI1 = NULL;
  gM2L.d_idxI2 = NULL;
  gM2L.d_idxI3 = NULL;
  gM2L.d_idx3Flat = NULL;
  gM2L.d_sgn3 = NULL;
  gM2L.d_g0 = NULL;
  gM2L.d_gk = NULL;
  gM2L.d_momPot = NULL;
  gM2L.d_momDpdn = NULL;
  gM2L.d_lec1 = NULL;
  gM2L.d_lec2 = NULL;
  gM2L.d_lec3 = NULL;
  gM2L.d_lec4 = NULL;

  gM2L.sys = NULL;
  gM2L.nCubes = 0;
  gM2L.nPairs = 0;
  gM2L.nGroups = 0;
  gM2L.maxOrder = 0;
  gM2L.maxIdxDim = 0;
  gM2L.totalPairCoeff = 0;
  gM2L.totalCubeCoeff = 0;
}

void freeLeafCache() {
  free(gLeaf.h_leafMatrixOffset);
  free(gLeaf.h_q2m0);
  free(gLeaf.h_q2m1);
  free(gLeaf.h_l2p0);
  free(gLeaf.h_l2p1);
  free(gLeaf.h_sgm);
  free(gLeaf.h_pot);
  free(gLeaf.h_momPot);
  free(gLeaf.h_momDpdn);
  free(gLeaf.h_lec1);
  free(gLeaf.h_lec2);
  free(gLeaf.h_lec3);
  free(gLeaf.h_lec4);
  gLeaf.h_leafMatrixOffset = NULL;
  gLeaf.h_q2m0 = NULL;
  gLeaf.h_q2m1 = NULL;
  gLeaf.h_l2p0 = NULL;
  gLeaf.h_l2p1 = NULL;
  gLeaf.h_sgm = NULL;
  gLeaf.h_pot = NULL;
  gLeaf.h_momPot = NULL;
  gLeaf.h_momDpdn = NULL;
  gLeaf.h_lec1 = NULL;
  gLeaf.h_lec2 = NULL;
  gLeaf.h_lec3 = NULL;
  gLeaf.h_lec4 = NULL;

  cudaFree(gLeaf.d_leafPanelStart);
  cudaFree(gLeaf.d_leafPanelCount);
  cudaFree(gLeaf.d_leafMatrixOffset);
  cudaFree(gLeaf.d_q2m0);
  cudaFree(gLeaf.d_q2m1);
  cudaFree(gLeaf.d_l2p0);
  cudaFree(gLeaf.d_l2p1);
  cudaFree(gLeaf.d_sgm);
  cudaFree(gLeaf.d_pot);
  cudaFree(gLeaf.d_momPot);
  cudaFree(gLeaf.d_momDpdn);
  cudaFree(gLeaf.d_lec1);
  cudaFree(gLeaf.d_lec2);
  cudaFree(gLeaf.d_lec3);
  cudaFree(gLeaf.d_lec4);
  gLeaf.d_leafPanelStart = NULL;
  gLeaf.d_leafPanelCount = NULL;
  gLeaf.d_leafMatrixOffset = NULL;
  gLeaf.d_q2m0 = NULL;
  gLeaf.d_q2m1 = NULL;
  gLeaf.d_l2p0 = NULL;
  gLeaf.d_l2p1 = NULL;
  gLeaf.d_sgm = NULL;
  gLeaf.d_pot = NULL;
  gLeaf.d_momPot = NULL;
  gLeaf.d_momDpdn = NULL;
  gLeaf.d_lec1 = NULL;
  gLeaf.d_lec2 = NULL;
  gLeaf.d_lec3 = NULL;
  gLeaf.d_lec4 = NULL;

  gLeaf.sys = NULL;
  gLeaf.nLeaves = 0;
  gLeaf.nMom = 0;
  gLeaf.totalMatrixEntries = 0;
}

void freeDirectCache() {
  free(gDirect.h_k0);
  free(gDirect.h_k1);
  free(gDirect.h_k2);
  free(gDirect.h_k3);
  gDirect.h_k0 = NULL;
  gDirect.h_k1 = NULL;
  gDirect.h_k2 = NULL;
  gDirect.h_k3 = NULL;

  cudaFree(gDirect.d_k0);
  cudaFree(gDirect.d_k1);
  cudaFree(gDirect.d_k2);
  cudaFree(gDirect.d_k3);
  cudaFree(gDirect.d_sgm);
  cudaFree(gDirect.d_pot);
  gDirect.d_k0 = NULL;
  gDirect.d_k1 = NULL;
  gDirect.d_k2 = NULL;
  gDirect.d_k3 = NULL;
  gDirect.d_sgm = NULL;
  gDirect.d_pot = NULL;

  gDirect.sys = NULL;
  gDirect.nPnls = 0;
  gDirect.nInteractions = 0;
}

void freeRhsCache() {
  free(gRhs.h_panels);
  free(gRhs.h_chrPos);
  free(gRhs.h_chrVal);
  gRhs.h_panels = NULL;
  gRhs.h_chrPos = NULL;
  gRhs.h_chrVal = NULL;

  cudaFree(gRhs.d_panels);
  cudaFree(gRhs.d_chrPos);
  cudaFree(gRhs.d_chrVal);
  cudaFree(gRhs.d_sgm);
  gRhs.d_panels = NULL;
  gRhs.d_chrPos = NULL;
  gRhs.d_chrVal = NULL;
  gRhs.d_sgm = NULL;

  gRhs.sys = NULL;
  gRhs.nPnls = 0;
  gRhs.nChar = 0;
  gRhs.qOrder = 0;
}

int allocateHostArrays(long long n) {
  size_t ni = (size_t)n;
  gNear.h_src = (int *)malloc(ni * sizeof(int));
  gNear.h_dst = (int *)malloc(ni * sizeof(int));
  gNear.h_pairSrcCount = (int *)malloc((size_t)gNear.sys->nNearPairsFlat * sizeof(int));
  gNear.h_pairInteractionOffset = (long long *)malloc((size_t)gNear.sys->nNearPairsFlat * sizeof(long long));
  gNear.h_leafPanelStart = (int *)malloc((size_t)gNear.sys->nLeafCubesFlat * sizeof(int));
  gNear.h_leafPanelCount = (int *)malloc((size_t)gNear.sys->nLeafCubesFlat * sizeof(int));
  gNear.h_leafPairOffset = (int *)calloc((size_t)gNear.sys->nLeafCubesFlat + 1, sizeof(int));
  gNear.h_k0 = (double *)malloc(ni * sizeof(double));
  gNear.h_k1 = (double *)malloc(ni * sizeof(double));
  gNear.h_k2 = (double *)malloc(ni * sizeof(double));
  gNear.h_k3 = (double *)malloc(ni * sizeof(double));
  if (!gNear.h_src || !gNear.h_dst || !gNear.h_pairSrcCount ||
      !gNear.h_pairInteractionOffset || !gNear.h_leafPanelStart ||
      !gNear.h_leafPanelCount || !gNear.h_leafPairOffset ||
      !gNear.h_k0 || !gNear.h_k1 || !gNear.h_k2 || !gNear.h_k3) {
    return 0;
  }
  return 1;
}

int allocateDeviceArrays(int nPnls, long long nInteractions) {
  size_t ni = (size_t)nInteractions;
  size_t vecBytes = (size_t)(2 * nPnls) * sizeof(double);
  cudaError_t err = cudaSuccess;

  err = cudaMalloc((void **)&gNear.d_src, ni * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_dst, ni * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_pairSrcCount, (size_t)gNear.sys->nNearPairsFlat * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_pairInteractionOffset, (size_t)gNear.sys->nNearPairsFlat * sizeof(long long));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_leafPanelStart, (size_t)gNear.sys->nLeafCubesFlat * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_leafPanelCount, (size_t)gNear.sys->nLeafCubesFlat * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_leafPairOffset, ((size_t)gNear.sys->nLeafCubesFlat + 1) * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_k0, ni * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_k1, ni * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_k2, ni * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_k3, ni * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_sgm, vecBytes);
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gNear.d_pot, vecBytes);
  if (err != cudaSuccess) return 0;

  return 1;
}

int allocateDirectArrays(int nPnls, long long nInteractions) {
  size_t ni = (size_t)nInteractions;
  size_t vecBytes = (size_t)(2 * nPnls) * sizeof(double);
  cudaError_t err = cudaSuccess;

  gDirect.h_k0 = (double *)malloc(ni * sizeof(double));
  gDirect.h_k1 = (double *)malloc(ni * sizeof(double));
  gDirect.h_k2 = (double *)malloc(ni * sizeof(double));
  gDirect.h_k3 = (double *)malloc(ni * sizeof(double));
  if (!gDirect.h_k0 || !gDirect.h_k1 || !gDirect.h_k2 || !gDirect.h_k3) {
    return 0;
  }

  err = cudaMalloc((void **)&gDirect.d_k0, ni * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gDirect.d_k1, ni * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gDirect.d_k2, ni * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gDirect.d_k3, ni * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gDirect.d_sgm, vecBytes);
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gDirect.d_pot, vecBytes);
  if (err != cudaSuccess) return 0;

  return 1;
}

int allocateM2LHostArrays() {
  size_t nPairs = (size_t)gM2L.nPairs;
  size_t nGroups = (size_t)gM2L.nGroups;
  size_t nCubes = (size_t)gM2L.nCubes;
  size_t nMom = (size_t)gM2L.totalCubeCoeff;
  size_t nPairCoeff = (size_t)gM2L.totalPairCoeff;
  size_t idxFlat = (size_t)gM2L.maxIdxDim * (size_t)gM2L.maxIdxDim * (size_t)gM2L.maxIdxDim;
  size_t maxMom = (size_t)(((gM2L.maxOrder + 1) * (gM2L.maxOrder + 2) * (gM2L.maxOrder + 3)) / 6);

  gM2L.h_pairCoeffOffset = (int *)malloc((nPairs + 1U) * sizeof(int));
  gM2L.h_groupDst = (int *)malloc(nGroups * sizeof(int));
  gM2L.h_groupOrder = (int *)malloc(nGroups * sizeof(int));
  gM2L.h_cubeCoeffOffset = (int *)malloc(nCubes * sizeof(int));
  gM2L.h_cubeNMom = (int *)malloc(nCubes * sizeof(int));
  gM2L.h_idxI1 = (int *)malloc(maxMom * sizeof(int));
  gM2L.h_idxI2 = (int *)malloc(maxMom * sizeof(int));
  gM2L.h_idxI3 = (int *)malloc(maxMom * sizeof(int));
  gM2L.h_idx3Flat = (int *)malloc(idxFlat * sizeof(int));
  gM2L.h_g0 = (double *)malloc(nPairCoeff * sizeof(double));
  gM2L.h_gk = (double *)malloc(nPairCoeff * sizeof(double));
  gM2L.h_momPot = (double *)malloc(nMom * sizeof(double));
  gM2L.h_momDpdn = (double *)malloc(nMom * sizeof(double));
  gM2L.h_lec1 = (double *)malloc(nMom * sizeof(double));
  gM2L.h_lec2 = (double *)malloc(nMom * sizeof(double));
  gM2L.h_lec3 = (double *)malloc(nMom * sizeof(double));
  gM2L.h_lec4 = (double *)malloc(nMom * sizeof(double));

  return gM2L.h_pairCoeffOffset && gM2L.h_groupDst && gM2L.h_groupOrder &&
         gM2L.h_cubeCoeffOffset && gM2L.h_cubeNMom &&
         gM2L.h_idxI1 && gM2L.h_idxI2 && gM2L.h_idxI3 && gM2L.h_idx3Flat &&
         gM2L.h_g0 && gM2L.h_gk &&
         gM2L.h_momPot && gM2L.h_momDpdn &&
         gM2L.h_lec1 && gM2L.h_lec2 && gM2L.h_lec3 && gM2L.h_lec4;
}

int allocateM2LDeviceArrays() {
  cudaError_t err = cudaSuccess;
  size_t nPairs = (size_t)gM2L.nPairs;
  size_t nGroups = (size_t)gM2L.nGroups;
  size_t nCubes = (size_t)gM2L.nCubes;
  size_t nMom = (size_t)gM2L.totalCubeCoeff;
  size_t nPairCoeff = (size_t)gM2L.totalPairCoeff;
  size_t idxFlat = (size_t)gM2L.maxIdxDim * (size_t)gM2L.maxIdxDim * (size_t)gM2L.maxIdxDim;
  size_t maxMom = (size_t)(((gM2L.maxOrder + 1) * (gM2L.maxOrder + 2) * (gM2L.maxOrder + 3)) / 6);

  err = cudaMalloc((void **)&gM2L.d_pairSrc, nPairs * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_pairCoeffOffset, (nPairs + 1U) * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_groupStart, nGroups * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_groupCount, nGroups * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_groupDst, nGroups * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_groupOrder, nGroups * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_cubeCoeffOffset, nCubes * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_cubeNMom, nCubes * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_idxI1, maxMom * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_idxI2, maxMom * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_idxI3, maxMom * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_idx3Flat, idxFlat * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_sgn3, maxMom * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_g0, nPairCoeff * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_gk, nPairCoeff * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_momPot, nMom * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_momDpdn, nMom * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_lec1, nMom * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_lec2, nMom * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_lec3, nMom * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gM2L.d_lec4, nMom * sizeof(double));
  if (err != cudaSuccess) return 0;

  return 1;
}

int allocateLeafHostArrays(const ssystem *sys, int nMom, long long totalEntries) {
  size_t nLeaves = (size_t)sys->nLeafCubesFlat;
  size_t total = (size_t)totalEntries;
  size_t vecPanels = (size_t)(2 * sys->nPnls);
  size_t cubeMom = (size_t)nLeaves * (size_t)nMom;

  gLeaf.h_leafMatrixOffset = (int *)malloc((nLeaves + 1U) * sizeof(int));
  gLeaf.h_q2m0 = (double *)malloc(total * sizeof(double));
  gLeaf.h_q2m1 = (double *)malloc(total * sizeof(double));
  gLeaf.h_l2p0 = (double *)malloc(total * sizeof(double));
  gLeaf.h_l2p1 = (double *)malloc(total * sizeof(double));
  gLeaf.h_sgm = (double *)malloc(vecPanels * sizeof(double));
  gLeaf.h_pot = (double *)malloc(vecPanels * sizeof(double));
  gLeaf.h_momPot = (double *)malloc(cubeMom * sizeof(double));
  gLeaf.h_momDpdn = (double *)malloc(cubeMom * sizeof(double));
  gLeaf.h_lec1 = (double *)malloc(cubeMom * sizeof(double));
  gLeaf.h_lec2 = (double *)malloc(cubeMom * sizeof(double));
  gLeaf.h_lec3 = (double *)malloc(cubeMom * sizeof(double));
  gLeaf.h_lec4 = (double *)malloc(cubeMom * sizeof(double));

  return gLeaf.h_leafMatrixOffset && gLeaf.h_q2m0 && gLeaf.h_q2m1 &&
         gLeaf.h_l2p0 && gLeaf.h_l2p1 &&
         gLeaf.h_sgm && gLeaf.h_pot &&
         gLeaf.h_momPot && gLeaf.h_momDpdn &&
         gLeaf.h_lec1 && gLeaf.h_lec2 && gLeaf.h_lec3 && gLeaf.h_lec4;
}

int allocateLeafDeviceArrays(const ssystem *sys, int nMom, long long totalEntries) {
  cudaError_t err = cudaSuccess;
  size_t nLeaves = (size_t)sys->nLeafCubesFlat;
  size_t total = (size_t)totalEntries;
  size_t vecPanels = (size_t)(2 * sys->nPnls);
  size_t cubeMom = (size_t)nLeaves * (size_t)nMom;

  err = cudaMalloc((void **)&gLeaf.d_leafPanelStart, nLeaves * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_leafPanelCount, nLeaves * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_leafMatrixOffset, (nLeaves + 1U) * sizeof(int));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_q2m0, total * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_q2m1, total * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_l2p0, total * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_l2p1, total * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_sgm, vecPanels * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_pot, vecPanels * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_momPot, cubeMom * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_momDpdn, cubeMom * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_lec1, cubeMom * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_lec2, cubeMom * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_lec3, cubeMom * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMalloc((void **)&gLeaf.d_lec4, cubeMom * sizeof(double));
  if (err != cudaSuccess) return 0;

  return 1;
}

int nearfieldBuildThreadCount(int nTasks) {
  const char *env = getenv("FABIPB_NEARFIELD_BUILD_THREADS");
  unsigned int hc = std::thread::hardware_concurrency();
  int threads;

  if (env != NULL) {
    threads = atoi(env);
  } else if (hc > 0U) {
    threads = (int)hc;
  } else {
    threads = 1;
  }
  if (threads < 1) {
    threads = 1;
  }
  if (threads > nTasks) {
    threads = nTasks;
  }
  if (threads > 32) {
    threads = 32;
  }
  return threads;
}

void buildM2LIndexTables() {
  int i = 0;
  int n, i1, i2;
  int dim = gM2L.maxIdxDim;
  size_t total = (size_t)dim * (size_t)dim * (size_t)dim;
  size_t idx;

  for (idx = 0; idx < total; idx++) {
    gM2L.h_idx3Flat[idx] = -1;
  }

  for (n = 0; n <= gM2L.maxOrder; n++) {
    for (i1 = 0; i1 <= n; i1++) {
      for (i2 = 0; i2 <= n - i1; i2++, i++) {
        int i3 = n - i1 - i2;
        gM2L.h_idxI1[i] = i1;
        gM2L.h_idxI2[i] = i2;
        gM2L.h_idxI3[i] = i3;
        gM2L.h_idx3Flat[((size_t)i1 * (size_t)dim + (size_t)i2) * (size_t)dim + (size_t)i3] = i;
      }
    }
  }
}

int buildM2LTables(const ssystem *sys) {
  int cubeIdx;
  int pairIdx;
  int groupIdx;
  double t0;
  double r[3];

  gM2L.sys = sys;
  gM2L.nCubes = sys->nFmmCubesFlat;
  gM2L.nPairs = sys->nM2LPairsFlat;
  gM2L.nGroups = sys->nM2LDstGroups;
  gM2L.maxOrder = sys->maxOrder;
  gM2L.maxIdxDim = sys->maxOrder + 1;
  gM2L.totalPairCoeff = 0;
  gM2L.totalCubeCoeff = 0;

  for (cubeIdx = 0; cubeIdx < sys->nFmmCubesFlat; cubeIdx++) {
    cube *cb = sys->fmmCubeByIdx[cubeIdx];
    int order = sys->ordM2L[cb->level];
    int nMom = sys->nMom[order];
    gM2L.totalCubeCoeff += nMom;
  }
  for (pairIdx = 0; pairIdx < sys->nM2LPairsFlat; pairIdx++) {
    int order = sys->m2lPairOrder[pairIdx];
    gM2L.totalPairCoeff += (long long)sys->nMom[order];
  }

  if (gM2L.nPairs <= 0 || gM2L.nGroups <= 0 || gM2L.nCubes <= 0) {
    return 0;
  }
  if (!allocateM2LHostArrays()) {
    return 0;
  }

  buildM2LIndexTables();

  gM2L.h_pairCoeffOffset[0] = 0;
  for (cubeIdx = 0; cubeIdx < sys->nFmmCubesFlat; cubeIdx++) {
    cube *cb = sys->fmmCubeByIdx[cubeIdx];
    int order = sys->ordM2L[cb->level];
    int nMom = sys->nMom[order];
    if (cubeIdx == 0) {
      gM2L.h_cubeCoeffOffset[cubeIdx] = 0;
    } else {
      gM2L.h_cubeCoeffOffset[cubeIdx] =
          gM2L.h_cubeCoeffOffset[cubeIdx - 1] + gM2L.h_cubeNMom[cubeIdx - 1];
    }
    gM2L.h_cubeNMom[cubeIdx] = nMom;
  }

  for (groupIdx = 0; groupIdx < sys->nM2LDstGroups; groupIdx++) {
    int start = sys->m2lDstGroupStart[groupIdx];
    gM2L.h_groupDst[groupIdx] = sys->m2lPairDst[start];
    gM2L.h_groupOrder[groupIdx] = sys->m2lPairOrder[start];
  }

  t0 = wall_seconds_cuda_local();
  for (pairIdx = 0; pairIdx < sys->nM2LPairsFlat; pairIdx++) {
    cube *src = sys->fmmCubeByIdx[sys->m2lPairSrc[pairIdx]];
    cube *dst = sys->fmmCubeByIdx[sys->m2lPairDst[pairIdx]];
    int order = sys->m2lPairOrder[pairIdx];
    int nMom = sys->nMom[order];
    int coeffOffset = gM2L.h_pairCoeffOffset[pairIdx];
    int k;

    for (k = 0; k < 3; k++) {
      r[k] = dst->x[k] - src->x[k];
    }
    setupDerivs(order, r);
    memcpy(&gM2L.h_g0[coeffOffset], dG0[0], (size_t)nMom * sizeof(double));
    memcpy(&gM2L.h_gk[coeffOffset], dGk[0], (size_t)nMom * sizeof(double));
    gM2L.h_pairCoeffOffset[pairIdx + 1] = coeffOffset + nMom;
  }
  (void)t0;

  if (!allocateM2LDeviceArrays()) {
    return 0;
  }

  if (cudaMemcpy(gM2L.d_pairSrc, sys->m2lPairSrc,
                 (size_t)sys->nM2LPairsFlat * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_pairCoeffOffset, gM2L.h_pairCoeffOffset,
                 ((size_t)sys->nM2LPairsFlat + 1U) * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_groupStart, sys->m2lDstGroupStart,
                 (size_t)sys->nM2LDstGroups * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_groupCount, sys->m2lDstGroupCount,
                 (size_t)sys->nM2LDstGroups * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_groupDst, gM2L.h_groupDst,
                 (size_t)sys->nM2LDstGroups * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_groupOrder, gM2L.h_groupOrder,
                 (size_t)sys->nM2LDstGroups * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_cubeCoeffOffset, gM2L.h_cubeCoeffOffset,
                 (size_t)sys->nFmmCubesFlat * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_cubeNMom, gM2L.h_cubeNMom,
                 (size_t)sys->nFmmCubesFlat * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_idxI1, gM2L.h_idxI1,
                 (size_t)sys->nMom[sys->maxOrder] * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_idxI2, gM2L.h_idxI2,
                 (size_t)sys->nMom[sys->maxOrder] * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_idxI3, gM2L.h_idxI3,
                 (size_t)sys->nMom[sys->maxOrder] * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_idx3Flat, gM2L.h_idx3Flat,
                 (size_t)gM2L.maxIdxDim * (size_t)gM2L.maxIdxDim * (size_t)gM2L.maxIdxDim * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_sgn3, sgn3,
                 (size_t)sys->nMom[sys->maxOrder] * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_g0, gM2L.h_g0,
                 (size_t)gM2L.totalPairCoeff * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gM2L.d_gk, gM2L.h_gk,
                 (size_t)gM2L.totalPairCoeff * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;

  if (sys->benchmarkMode > 0)
    printf("GPU M2L cache: cubes=%d pairs=%d coeff=%lld\n",
           gM2L.nCubes, gM2L.nPairs, gM2L.totalPairCoeff);
  return 1;
}

int buildLeafTables(const ssystem *sys) {
  int idx;
  int nMom = sys->nMom[sys->ordMom[sys->depth]];
  long long totalEntries = 0;

  gLeaf.sys = sys;
  gLeaf.nLeaves = sys->nLeafCubesFlat;
  gLeaf.nMom = nMom;
  for (idx = 0; idx < sys->nLeafCubesFlat; idx++) {
    totalEntries += (long long)sys->leafPanelCount[idx] * (long long)nMom;
  }
  gLeaf.totalMatrixEntries = totalEntries;
  if (totalEntries <= 0) {
    return 0;
  }

  if (!allocateLeafHostArrays(sys, nMom, totalEntries)) {
    return 0;
  }

  gLeaf.h_leafMatrixOffset[0] = 0;
  for (idx = 0; idx < sys->nLeafCubesFlat; idx++) {
    int count = sys->leafPanelCount[idx];
    size_t bytes = (size_t)nMom * (size_t)count * sizeof(double);
    int offset = gLeaf.h_leafMatrixOffset[idx];
    memcpy(&gLeaf.h_q2m0[offset], Q2M0[idx], bytes);
    memcpy(&gLeaf.h_q2m1[offset], Q2M1[idx], bytes);
    memcpy(&gLeaf.h_l2p0[offset], L2P0[idx], bytes);
    memcpy(&gLeaf.h_l2p1[offset], L2P1[idx], bytes);
    gLeaf.h_leafMatrixOffset[idx + 1] = offset + nMom * count;
  }

  if (!allocateLeafDeviceArrays(sys, nMom, totalEntries)) {
    return 0;
  }

  if (cudaMemcpy(gLeaf.d_leafPanelStart, sys->leafPanelStart,
                 (size_t)sys->nLeafCubesFlat * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gLeaf.d_leafPanelCount, sys->leafPanelCount,
                 (size_t)sys->nLeafCubesFlat * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gLeaf.d_leafMatrixOffset, gLeaf.h_leafMatrixOffset,
                 ((size_t)sys->nLeafCubesFlat + 1U) * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gLeaf.d_q2m0, gLeaf.h_q2m0,
                 (size_t)totalEntries * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gLeaf.d_q2m1, gLeaf.h_q2m1,
                 (size_t)totalEntries * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gLeaf.d_l2p0, gLeaf.h_l2p0,
                 (size_t)totalEntries * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gLeaf.d_l2p1, gLeaf.h_l2p1,
                 (size_t)totalEntries * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess) return 0;

  if (sys->benchmarkMode > 0)
    printf("GPU leaf cache: leaves=%d coeff=%d matrix-entries=%lld\n",
           gLeaf.nLeaves, gLeaf.nLeaves * nMom, totalEntries);
  return 1;
}

int buildNearfieldTables(const ssystem *sys) {
  int pairIdx;
  long long totalInteractions = 0;
  long long k = 0;
  double t0, t1;

  for (pairIdx = 0; pairIdx < sys->nNearPairsFlat; pairIdx++) {
    int srcLeaf = sys->nearPairSrc[pairIdx];
    int dstLeaf = sys->nearPairDst[pairIdx];
    long long srcCount = (long long)sys->leafPanelCount[srcLeaf];
    long long dstCount = (long long)sys->leafPanelCount[dstLeaf];
    totalInteractions += srcCount * dstCount;
  }

  if (totalInteractions <= 0) {
    return 0;
  }
  gNear.sys = sys;
  gNear.nearfieldMode = sys->gpuNearfieldMode;
  t0 = wall_seconds_cuda_local();
  if (!allocateHostArrays(totalInteractions)) {
    return 0;
  }

  memcpy(gNear.h_leafPanelStart, sys->leafPanelStart, (size_t)sys->nLeafCubesFlat * sizeof(int));
  memcpy(gNear.h_leafPanelCount, sys->leafPanelCount, (size_t)sys->nLeafCubesFlat * sizeof(int));
  for (pairIdx = 0; pairIdx < sys->nNearPairsFlat; pairIdx++) {
    int srcLeaf = sys->nearPairSrc[pairIdx];
    int dstLeaf = sys->nearPairDst[pairIdx];
    int srcCount = sys->leafPanelCount[srcLeaf];
    gNear.h_leafPairOffset[dstLeaf + 1] += 1;
    gNear.h_pairSrcCount[pairIdx] = srcCount;
    gNear.h_pairInteractionOffset[pairIdx] = k;
    k += (long long)srcCount * (long long)sys->leafPanelCount[dstLeaf];
  }
  for (pairIdx = 0; pairIdx < sys->nLeafCubesFlat; pairIdx++) {
    gNear.h_leafPairOffset[pairIdx + 1] += gNear.h_leafPairOffset[pairIdx];
  }
  t1 = wall_seconds_cuda_local();
  fmmNearGpuMetaTime += (t1 - t0);

  kernel = kernelKER4;
  t0 = wall_seconds_cuda_local();
  {
    int nThreads = nearfieldBuildThreadCount(sys->nNearPairsFlat);
    if (nThreads <= 1) {
      for (pairIdx = 0; pairIdx < sys->nNearPairsFlat; pairIdx++) {
        int srcLeaf = sys->nearPairSrc[pairIdx];
        int dstLeaf = sys->nearPairDst[pairIdx];
        int srcStart = sys->leafPanelStart[srcLeaf];
        int srcCount = sys->leafPanelCount[srcLeaf];
        int dstStart = sys->leafPanelStart[dstLeaf];
        int dstCount = sys->leafPanelCount[dstLeaf];
        long long base = gNear.h_pairInteractionOffset[pairIdx];
        int i, j;

        for (i = 0; i < dstCount; i++) {
          int dstPanelIdx = dstStart + i;
          panel *pnlX = sys->panelByIdx[dstPanelIdx];
          for (j = 0; j < srcCount; j++) {
            long long idx = base + (long long)i * (long long)srcCount + (long long)j;
            int srcPanelIdx = srcStart + j;
            panel *pnlY = sys->panelByIdx[srcPanelIdx];
            double *KER = panelIA0(pnlX, pnlY);

            gNear.h_dst[idx] = dstPanelIdx;
            gNear.h_src[idx] = srcPanelIdx;
            gNear.h_k0[idx] = KER[0];
            gNear.h_k1[idx] = KER[1];
            gNear.h_k2[idx] = KER[2];
            gNear.h_k3[idx] = KER[3];
          }
        }
      }
    } else {
      std::vector<std::thread> workers;
      int t;

      workers.reserve((size_t)nThreads);
      for (t = 0; t < nThreads; t++) {
        int begin = (sys->nNearPairsFlat * t) / nThreads;
        int end = (sys->nNearPairsFlat * (t + 1)) / nThreads;
        workers.emplace_back([=]() {
          int localPairIdx;
          for (localPairIdx = begin; localPairIdx < end; localPairIdx++) {
            int srcLeaf = sys->nearPairSrc[localPairIdx];
            int dstLeaf = sys->nearPairDst[localPairIdx];
            int srcStart = sys->leafPanelStart[srcLeaf];
            int srcCount = sys->leafPanelCount[srcLeaf];
            int dstStart = sys->leafPanelStart[dstLeaf];
            int dstCount = sys->leafPanelCount[dstLeaf];
            long long base = gNear.h_pairInteractionOffset[localPairIdx];
            int i, j;

            for (i = 0; i < dstCount; i++) {
              int dstPanelIdx = dstStart + i;
              panel *pnlX = sys->panelByIdx[dstPanelIdx];
              for (j = 0; j < srcCount; j++) {
                long long idx = base + (long long)i * (long long)srcCount + (long long)j;
                int srcPanelIdx = srcStart + j;
                panel *pnlY = sys->panelByIdx[srcPanelIdx];
                double *KER = panelIA0(pnlX, pnlY);

                gNear.h_dst[idx] = dstPanelIdx;
                gNear.h_src[idx] = srcPanelIdx;
                gNear.h_k0[idx] = KER[0];
                gNear.h_k1[idx] = KER[1];
                gNear.h_k2[idx] = KER[2];
                gNear.h_k3[idx] = KER[3];
              }
            }
          }
        });
      }
      for (t = 0; t < nThreads; t++) {
        workers[(size_t)t].join();
      }
    }
  }
  t1 = wall_seconds_cuda_local();
  fmmNearGpuCoeffTime += (t1 - t0);

  t0 = wall_seconds_cuda_local();
  if (!allocateDeviceArrays(sys->nPnls, totalInteractions)) {
    return 0;
  }

  if (cudaMemcpy(gNear.d_src, gNear.h_src, (size_t)totalInteractions * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gNear.d_dst, gNear.h_dst, (size_t)totalInteractions * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gNear.d_pairSrcCount, gNear.h_pairSrcCount, (size_t)sys->nNearPairsFlat * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gNear.d_pairInteractionOffset, gNear.h_pairInteractionOffset, (size_t)sys->nNearPairsFlat * sizeof(long long), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gNear.d_leafPanelStart, gNear.h_leafPanelStart, (size_t)sys->nLeafCubesFlat * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gNear.d_leafPanelCount, gNear.h_leafPanelCount, (size_t)sys->nLeafCubesFlat * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gNear.d_leafPairOffset, gNear.h_leafPairOffset, ((size_t)sys->nLeafCubesFlat + 1) * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gNear.d_k0, gNear.h_k0, (size_t)totalInteractions * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gNear.d_k1, gNear.h_k1, (size_t)totalInteractions * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gNear.d_k2, gNear.h_k2, (size_t)totalInteractions * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gNear.d_k3, gNear.h_k3, (size_t)totalInteractions * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  t1 = wall_seconds_cuda_local();
  fmmNearGpuUploadTime += (t1 - t0);

  gNear.nPnls = sys->nPnls;
  gNear.nInteractions = totalInteractions;
  if (sys->benchmarkMode > 0)
    printf("GPU nearfield cache: panel-pairs=%lld mode=%d\n", totalInteractions, sys->gpuNearfieldMode);
  return 1;
}

int buildDirectTables(const ssystem *sys) {
  long long nInteractions;
  size_t coeffBytes;
  size_t vecBytes;
  size_t totalBytes;
  size_t hostCoeffBytes;
  size_t combinedBytes;
  size_t freeBytes = 0, totalGpuBytes = 0;
  int i, j;

  nInteractions = (long long)sys->nPnls * (long long)sys->nPnls;
  if (nInteractions <= 0) {
    return 0;
  }

  coeffBytes = (size_t)nInteractions * sizeof(double);
  vecBytes = (size_t)(2 * sys->nPnls) * sizeof(double);
  hostCoeffBytes = 4 * coeffBytes;
  totalBytes = 4 * coeffBytes + 2 * vecBytes;
  combinedBytes = hostCoeffBytes + totalBytes;
  if (sys->benchmarkMode > 0)
    printf("Direct GPU memory estimate: solver-host=%.3f GB host-coeff=%.3f GB device-total=%.3f GB combined=%.3f GB\n",
           (double)memcount / (1024.0 * 1024.0 * 1024.0),
           (double)hostCoeffBytes / (1024.0 * 1024.0 * 1024.0),
           (double)totalBytes / (1024.0 * 1024.0 * 1024.0),
           (double)combinedBytes / (1024.0 * 1024.0 * 1024.0));
  if (cudaMemGetInfo(&freeBytes, &totalGpuBytes) == cudaSuccess) {
    if (sys->benchmarkMode > 0)
      printf("Direct GPU device memory: free=%.3f GB total=%.3f GB\n",
             (double)freeBytes / (1024.0 * 1024.0 * 1024.0),
             (double)totalGpuBytes / (1024.0 * 1024.0 * 1024.0));
    if (totalBytes > freeBytes * 7 / 10) {
      printf("Direct GPU cache unavailable: need %.3f GB, free %.3f GB\n",
             (double)totalBytes / (1024.0 * 1024.0 * 1024.0),
             (double)freeBytes / (1024.0 * 1024.0 * 1024.0));
      return 0;
    }
  }

  if (!allocateDirectArrays(sys->nPnls, nInteractions)) {
    return 0;
  }

  kernel = kernelKER4;
  for (i = 0; i < sys->nPnls; i++) {
    panel *pnlX = sys->panelByIdx[i];
    long long base = (long long)i * (long long)sys->nPnls;
    for (j = 0; j < sys->nPnls; j++) {
      panel *pnlY = sys->panelByIdx[j];
      double *KER = panelIA0(pnlX, pnlY);
      long long idx = base + (long long)j;

      gDirect.h_k0[idx] = KER[0];
      gDirect.h_k1[idx] = KER[1];
      gDirect.h_k2[idx] = KER[2];
      gDirect.h_k3[idx] = KER[3];
    }
  }

  if (cudaMemcpy(gDirect.d_k0, gDirect.h_k0, coeffBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gDirect.d_k1, gDirect.h_k1, coeffBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gDirect.d_k2, gDirect.h_k2, coeffBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gDirect.d_k3, gDirect.h_k3, coeffBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;

  gDirect.sys = sys;
  gDirect.nPnls = sys->nPnls;
  gDirect.nInteractions = nInteractions;
  if (sys->benchmarkMode > 0)
    printf("GPU direct cache: panel-pairs=%lld\n", nInteractions);
  return 1;
}

int buildRhsTables(const ssystem *sys, int qOrder) {
  int i;
  size_t panelBytes;
  size_t chrPosBytes;
  size_t chrValBytes;
  size_t sgmBytes;

  if (qOrder < 1 || qOrder > 10) {
    return 0;
  }

  gRhs.h_panels = (RhsPanelGeom *)malloc((size_t)sys->nPnls * sizeof(RhsPanelGeom));
  gRhs.h_chrPos = (double *)malloc((size_t)(3 * sys->nChar) * sizeof(double));
  gRhs.h_chrVal = (double *)malloc((size_t)sys->nChar * sizeof(double));
  if (!gRhs.h_panels || !gRhs.h_chrPos || !gRhs.h_chrVal) {
    return 0;
  }

  for (i = 0; i < sys->nPnls; i++) {
    panel *p = sys->panelByIdx[i];
    int k;
    for (k = 0; k < 3; k++) {
      gRhs.h_panels[i].v0[k] = p->vtx[0][k];
      gRhs.h_panels[i].a0[k] = p->a[0][k];
      gRhs.h_panels[i].a2[k] = p->a[2][k];
      gRhs.h_panels[i].normal[k] = p->normal[k];
    }
    gRhs.h_panels[i].area = p->area;
  }
  memcpy(gRhs.h_chrPos, sys->pos, (size_t)(3 * sys->nChar) * sizeof(double));
  memcpy(gRhs.h_chrVal, sys->chr, (size_t)sys->nChar * sizeof(double));

  panelBytes = (size_t)sys->nPnls * sizeof(RhsPanelGeom);
  chrPosBytes = (size_t)(3 * sys->nChar) * sizeof(double);
  chrValBytes = (size_t)sys->nChar * sizeof(double);
  sgmBytes = (size_t)(2 * sys->nPnls) * sizeof(double);

  if (cudaMalloc((void **)&gRhs.d_panels, panelBytes) != cudaSuccess) return 0;
  if (cudaMalloc((void **)&gRhs.d_chrPos, chrPosBytes) != cudaSuccess) return 0;
  if (cudaMalloc((void **)&gRhs.d_chrVal, chrValBytes) != cudaSuccess) return 0;
  if (cudaMalloc((void **)&gRhs.d_sgm, sgmBytes) != cudaSuccess) return 0;

  if (cudaMemcpy(gRhs.d_panels, gRhs.h_panels, panelBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gRhs.d_chrPos, gRhs.h_chrPos, chrPosBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpy(gRhs.d_chrVal, gRhs.h_chrVal, chrValBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpyToSymbol(c_rhsTLeg, tLegA[qOrder], (size_t)qOrder * sizeof(double), 0, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
  if (cudaMemcpyToSymbol(c_rhsWLeg, wLegA[qOrder], (size_t)qOrder * sizeof(double), 0, cudaMemcpyHostToDevice) != cudaSuccess) return 0;

  gRhs.sys = sys;
  gRhs.nPnls = sys->nPnls;
  gRhs.nChar = sys->nChar;
  gRhs.qOrder = qOrder;
  if (sys->benchmarkMode > 0)
    printf("GPU RHS cache: panels=%d charges=%d qOrder=%d\n", gRhs.nPnls, gRhs.nChar, gRhs.qOrder);
  return 1;
}

__global__ void nearfieldApplyKernel(
    int nPnls,
    long long nInteractions,
    const int *src,
    const int *dst,
    const double *k0,
    const double *k1,
    const double *k2,
    const double *k3,
    double alpha,
    const double *sgm,
    double *pot) {
  long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (tid >= nInteractions) {
    return;
  }

  int s = src[tid];
  int d = dst[tid];
  double x_pot = sgm[s];
  double x_dpdn = sgm[s + nPnls];
  double addPot = (k0[tid] * x_dpdn + k1[tid] * x_pot) * alpha;
  double addDpdn = (k2[tid] * x_dpdn + k3[tid] * x_pot) * alpha;

#if __CUDA_ARCH__ >= 600
  atomicAdd(&pot[d], addPot);
  atomicAdd(&pot[d + nPnls], addDpdn);
#else
  unsigned long long int *addr1 = (unsigned long long int *)&pot[d];
  unsigned long long int old1 = *addr1, assumed1;
  do {
    assumed1 = old1;
    old1 = atomicCAS(addr1, assumed1,
                     __double_as_longlong(addPot + __longlong_as_double(assumed1)));
  } while (assumed1 != old1);

  unsigned long long int *addr2 = (unsigned long long int *)&pot[d + nPnls];
  unsigned long long int old2 = *addr2, assumed2;
  do {
    assumed2 = old2;
    old2 = atomicCAS(addr2, assumed2,
                     __double_as_longlong(addDpdn + __longlong_as_double(assumed2)));
  } while (assumed2 != old2);
#endif
}

__global__ void rhsApplyKernel(
    int nPnls,
    int nChar,
    int qOrder,
    double fac,
    const RhsPanelGeom *panels,
    const double *chrPos,
    const double *chrVal,
    double *sgm) {
  int panelIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (panelIdx >= nPnls) {
    return;
  }

  RhsPanelGeom pnl = panels[panelIdx];
  double sum0 = 0.0;
  double sum1 = 0.0;

  for (int chrIdx = 0; chrIdx < nChar; chrIdx++) {
    const double *chrY = &chrPos[3 * chrIdx];
    double chr = chrVal[chrIdx];
    double r0[3];
    r0[0] = pnl.v0[0] - chrY[0];
    r0[1] = pnl.v0[1] - chrY[1];
    r0[2] = pnl.v0[2] - chrY[2];

    for (int ix = 0; ix < qOrder; ix++) {
      double tx = c_rhsTLeg[ix];
      double wx = c_rhsWLeg[ix];
      for (int jx = 0; jx < qOrder; jx++) {
        double inner = c_rhsTLeg[jx];
        double wy = c_rhsWLeg[jx];
        double r[3];
        double r2, ri, r3i, ip;
        for (int qk = 0; qk < 3; qk++) {
          r[qk] = r0[qk] + tx * (pnl.a2[qk] + inner * pnl.a0[qk]);
        }
        r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        ri = rsqrt(r2);
        r3i = ri / r2;
        ip = pnl.normal[0] * r[0] + pnl.normal[1] * r[1] + pnl.normal[2] * r[2];
        sum0 += chr * ri * tx * wx * wy;
        sum1 += chr * (-ip * r3i) * tx * wx * wy;
      }
    }
  }

  sgm[panelIdx] = fac * (2.0 * pnl.area * sum0);
  sgm[panelIdx + nPnls] = fac * (2.0 * pnl.area * sum1);
}

__global__ void nearfieldLeafApplyKernel(
    int nPnls,
    const int *leafPanelStart,
    const int *leafPanelCount,
    const int *leafPairOffset,
    const int *pairSrcCount,
    const long long *pairInteractionOffset,
    const int *src,
    const double *k0,
    const double *k1,
    const double *k2,
    const double *k3,
    double alpha,
    const double *sgm,
    double *pot) {
  int leaf = blockIdx.x;
  int tid = threadIdx.x;
  int dstStart = leafPanelStart[leaf];
  int dstCount = leafPanelCount[leaf];
  int pairBegin = leafPairOffset[leaf];
  int pairEnd = leafPairOffset[leaf + 1];
  int localDst;

  for (localDst = tid; localDst < dstCount; localDst += blockDim.x) {
    int d = dstStart + localDst;
    double sumPot = 0.0;
    double sumDpdn = 0.0;
    int pairIdx;

    for (pairIdx = pairBegin; pairIdx < pairEnd; pairIdx++) {
      int srcCount = pairSrcCount[pairIdx];
      long long base = pairInteractionOffset[pairIdx] + (long long)localDst * (long long)srcCount;
      int j;

      for (j = 0; j < srcCount; j++) {
        long long idx = base + (long long)j;
        int s = src[idx];
        double x_pot = sgm[s];
        double x_dpdn = sgm[s + nPnls];
        sumPot += (k0[idx] * x_dpdn + k1[idx] * x_pot) * alpha;
        sumDpdn += (k2[idx] * x_dpdn + k3[idx] * x_pot) * alpha;
      }
    }

    pot[d] += sumPot;
    pot[d + nPnls] += sumDpdn;
  }
}

__global__ void directApplyKernel(
    int nPnls,
    const double *k0,
    const double *k1,
    const double *k2,
    const double *k3,
    double alpha,
    double beta,
    const double *sgm,
    double *pot) {
  __shared__ double shPot[256];
  __shared__ double shDpdn[256];
  int d = blockIdx.x;
  int tid = threadIdx.x;
  int s;
  long long base;
  double sumPot = 0.0;
  double sumDpdn = 0.0;
  int stride;

  if (d >= nPnls) {
    return;
  }

  base = (long long)d * (long long)nPnls;
  for (s = tid; s < nPnls; s += blockDim.x) {
    long long idx = base + (long long)s;
    double x_pot = sgm[s];
    double x_dpdn = sgm[s + nPnls];
    sumPot += k0[idx] * x_dpdn + k1[idx] * x_pot;
    sumDpdn += k2[idx] * x_dpdn + k3[idx] * x_pot;
  }

  shPot[tid] = sumPot;
  shDpdn[tid] = sumDpdn;
  __syncthreads();

  for (stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shPot[tid] += shPot[tid + stride];
      shDpdn[tid] += shDpdn[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    pot[d] = beta * pot[d] + alpha * shPot[0];
    pot[d + nPnls] = beta * pot[d + nPnls] + alpha * shDpdn[0];
  }
}

__global__ void m2lGroupedKernel(
    int nGroups,
    int maxIdxDim,
    const int *groupStart,
    const int *groupCount,
    const int *groupDst,
    const int *groupOrder,
    const int *pairSrc,
    const int *pairCoeffOffset,
    const int *cubeCoeffOffset,
    const int *idxI1,
    const int *idxI2,
    const int *idxI3,
    const int *idx3Flat,
    const int *sgn3Vals,
    const double *g0,
    const double *gk,
    double epsilonLocal,
    const double *momPot,
    const double *momDpdn,
    double *lec1,
    double *lec2,
    double *lec3,
    double *lec4) {
  int group = blockIdx.x;
  int tid = threadIdx.x;

  if (group >= nGroups) {
    return;
  }

  {
    int order = groupOrder[group];
    int dstCube = groupDst[group];
    int dstOffset = cubeCoeffOffset[dstCube];
    int groupBegin = groupStart[group];
    int groupEnd = groupBegin + groupCount[group];
    int nMom = ((order + 1) * (order + 2) * (order + 3)) / 6;
    int i;

    for (i = tid; i < nMom; i += blockDim.x) {
      int i1 = idxI1[i];
      int i2 = idxI2[i];
      int i3 = idxI3[i];
      int n = i1 + i2 + i3;
      double tmp1 = 0.0;
      double tmp2 = 0.0;
      double tmp3 = 0.0;
      double tmp4 = 0.0;
      int pairIdx;

      for (pairIdx = groupBegin; pairIdx < groupEnd; pairIdx++) {
        int srcCube = pairSrc[pairIdx];
        int srcOffset = cubeCoeffOffset[srcCube];
        int coeffOffset = pairCoeffOffset[pairIdx];
        int j = 0;
        int m;

        for (m = 0; m <= order - n; m++) {
          int j1;
          for (j1 = 0; j1 <= m; j1++) {
            int k1 = i1 + j1;
            int j2;
            for (j2 = 0; j2 <= m - j1; j2++, j++) {
              int j3 = m - j1 - j2;
              int k2 = i2 + j2;
              int k3 = i3 + j3;
              int idxk = idx3Flat[((k1 * maxIdxDim) + k2) * maxIdxDim + k3];
              double c1 = (double)sgn3Vals[j] * momPot[srcOffset + j];
              double c2 = (double)sgn3Vals[j] * momDpdn[srcOffset + j];
              double veck1 = g0[coeffOffset + idxk] - gk[coeffOffset + idxk];
              double veck2 = epsilonLocal * gk[coeffOffset + idxk] - g0[coeffOffset + idxk];
              double veck3 = g0[coeffOffset + idxk] - gk[coeffOffset + idxk] / epsilonLocal;

              tmp1 += c2 * veck1;
              tmp2 += c1 * veck2;
              tmp3 += c2 * veck3;
              tmp4 += c1 * (-veck1);
            }
          }
        }
      }

      lec1[dstOffset + i] += tmp1;
      lec2[dstOffset + i] += tmp2;
      lec3[dstOffset + i] += tmp3;
      lec4[dstOffset + i] += tmp4;
    }
  }
}

__global__ void q2mLeafKernel(
    int nPnls,
    int nMom,
    const int *leafPanelStart,
    const int *leafPanelCount,
    const int *leafMatrixOffset,
    const double *q2m0,
    const double *q2m1,
    const double *sgm,
    double *momPot,
    double *momDpdn) {
  int leaf = blockIdx.x;
  int row = threadIdx.x;
  int count = leafPanelCount[leaf];
  int start = leafPanelStart[leaf];
  int offset = leafMatrixOffset[leaf];

  if (row >= nMom) {
    return;
  }

  {
    double sumPot = 0.0;
    double sumDpdn = 0.0;
    int j;
    for (j = 0; j < count; j++) {
      int panelIdx = start + j;
      int matIdx = offset + j * nMom + row;
      sumPot += q2m1[matIdx] * sgm[panelIdx];
      sumDpdn += q2m0[matIdx] * sgm[panelIdx + nPnls];
    }
    momPot[leaf * nMom + row] = sumPot;
    momDpdn[leaf * nMom + row] = sumDpdn;
  }
}

__global__ void l2pLeafKernel(
    int nPnls,
    int nMom,
    const int *leafPanelStart,
    const int *leafPanelCount,
    const int *leafMatrixOffset,
    const double *l2p0,
    const double *l2p1,
    double alpha,
    double beta,
    const double *lec1,
    const double *lec2,
    const double *lec3,
    const double *lec4,
    double *pot) {
  int leaf = blockIdx.x;
  int panelLocal = threadIdx.x;
  int count = leafPanelCount[leaf];
  int start = leafPanelStart[leaf];
  int offset = leafMatrixOffset[leaf];

  for (; panelLocal < count; panelLocal += blockDim.x) {
    int panelIdx = start + panelLocal;
    double sumPot = 0.0;
    double sumDpdn = 0.0;
    int row;
    for (row = 0; row < nMom; row++) {
      int matIdx = offset + panelLocal * nMom + row;
      int lecIdx = leaf * nMom + row;
      sumPot += l2p0[matIdx] * lec1[lecIdx] + l2p0[matIdx] * lec2[lecIdx];
      sumDpdn += l2p1[matIdx] * lec3[lecIdx] + l2p1[matIdx] * lec4[lecIdx];
    }
    pot[panelIdx] = beta * pot[panelIdx] + alpha * sumPot;
    pot[panelIdx + nPnls] = beta * pot[panelIdx + nPnls] + alpha * sumDpdn;
  }
}
}  // namespace

int gpuBackendAvailable(void) {
  static int warned = 0;
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    if (!warned) {
      printf("CUDA backend unavailable: %s\n", cudaGetErrorString(err));
      warned = 1;
    }
    return 0;
  }
  return (deviceCount > 0) ? 1 : 0;
}

int gpuNearfieldApply(ssystem *sys, double alpha, const double *sgm, double *pot) {
  cudaError_t err;
  size_t vecBytes;
  int blockSize;
  int gridSize;
  double t0, t1;
  static int printedMode = -1;

  if (sys == NULL || sgm == NULL || pot == NULL) {
    return 0;
  }

  if (gNear.sys != sys) {
    t0 = wall_seconds_cuda_local();
    freeNearfieldCache();
    if (!buildNearfieldTables(sys)) {
      freeNearfieldCache();
      return 0;
    }
    t1 = wall_seconds_cuda_local();
    fmmNearGpuBuildTime += (t1 - t0);
  }

  vecBytes = (size_t)(2 * gNear.nPnls) * sizeof(double);
  t0 = wall_seconds_cuda_local();
  err = cudaMemcpy(gNear.d_sgm, sgm, vecBytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gNear.d_pot, pot, vecBytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  t1 = wall_seconds_cuda_local();
  fmmNearGpuH2DTime += (t1 - t0);

  blockSize = 256;
  if (sys->benchmarkMode > 0 && printedMode != sys->gpuNearfieldMode) {
    printf("GPU nearfield apply mode=%d (%s)\n",
           sys->gpuNearfieldMode,
           (sys->gpuNearfieldMode == 0) ? "interaction" : "destination-leaf");
    printedMode = sys->gpuNearfieldMode;
  }
  t0 = wall_seconds_cuda_local();
  if (sys->gpuNearfieldMode == 1) {
    gridSize = sys->nLeafCubesFlat;
    nearfieldLeafApplyKernel<<<gridSize, blockSize>>>(
        gNear.nPnls,
        gNear.d_leafPanelStart, gNear.d_leafPanelCount, gNear.d_leafPairOffset,
        gNear.d_pairSrcCount, gNear.d_pairInteractionOffset,
        gNear.d_src, gNear.d_k0, gNear.d_k1, gNear.d_k2, gNear.d_k3,
        alpha, gNear.d_sgm, gNear.d_pot);
  } else {
    gridSize = (int)((gNear.nInteractions + blockSize - 1) / blockSize);
    nearfieldApplyKernel<<<gridSize, blockSize>>>(
        gNear.nPnls, gNear.nInteractions,
        gNear.d_src, gNear.d_dst, gNear.d_k0, gNear.d_k1, gNear.d_k2, gNear.d_k3,
        alpha, gNear.d_sgm, gNear.d_pot);
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) return 0;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 0;
  t1 = wall_seconds_cuda_local();
  fmmNearGpuKernelTime += (t1 - t0);

  t0 = wall_seconds_cuda_local();
  err = cudaMemcpy(pot, gNear.d_pot, vecBytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  t1 = wall_seconds_cuda_local();
  fmmNearGpuD2HTime += (t1 - t0);
  return 1;
}

int gpuDirectApply(ssystem *sys, double alpha, double beta, const double *sgm, double *pot) {
  cudaError_t err;
  size_t vecBytes;
  int blockSize;
  int gridSize;

  if (sys == NULL || sgm == NULL || pot == NULL) {
    return 0;
  }

  if (gDirect.sys != sys) {
    freeDirectCache();
    if (!buildDirectTables(sys)) {
      freeDirectCache();
      return 0;
    }
  }

  vecBytes = (size_t)(2 * gDirect.nPnls) * sizeof(double);
  err = cudaMemcpy(gDirect.d_sgm, sgm, vecBytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gDirect.d_pot, pot, vecBytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;

  blockSize = 256;
  gridSize = gDirect.nPnls;
  directApplyKernel<<<gridSize, blockSize>>>(
      gDirect.nPnls, gDirect.d_k0, gDirect.d_k1, gDirect.d_k2, gDirect.d_k3,
      alpha, beta, gDirect.d_sgm, gDirect.d_pot);
  err = cudaGetLastError();
  if (err != cudaSuccess) return 0;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 0;

  err = cudaMemcpy(pot, gDirect.d_pot, vecBytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  return 1;
}

int gpuM2LApply(ssystem *sys) {
  int cubeIdx;
  cudaError_t err;
  int blockSize;

  if (sys == NULL || sys->nM2LPairsFlat <= 0) {
    return 0;
  }

  if (gM2L.sys != sys) {
    freeM2LCache();
    if (!buildM2LTables(sys)) {
      freeM2LCache();
      return 0;
    }
  }

  for (cubeIdx = 0; cubeIdx < sys->nFmmCubesFlat; cubeIdx++) {
    cube *cb = sys->fmmCubeByIdx[cubeIdx];
    int offset = gM2L.h_cubeCoeffOffset[cubeIdx];
    int nMom = gM2L.h_cubeNMom[cubeIdx];
    memcpy(&gM2L.h_momPot[offset], cb->mom_pot, (size_t)nMom * sizeof(double));
    memcpy(&gM2L.h_momDpdn[offset], cb->mom_dpdn, (size_t)nMom * sizeof(double));
  }

  err = cudaMemcpy(gM2L.d_momPot, gM2L.h_momPot,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gM2L.d_momDpdn, gM2L.h_momDpdn,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemset(gM2L.d_lec1, 0, (size_t)gM2L.totalCubeCoeff * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMemset(gM2L.d_lec2, 0, (size_t)gM2L.totalCubeCoeff * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMemset(gM2L.d_lec3, 0, (size_t)gM2L.totalCubeCoeff * sizeof(double));
  if (err != cudaSuccess) return 0;
  err = cudaMemset(gM2L.d_lec4, 0, (size_t)gM2L.totalCubeCoeff * sizeof(double));
  if (err != cudaSuccess) return 0;

  blockSize = 128;
  m2lGroupedKernel<<<gM2L.nGroups, blockSize>>>(
      gM2L.nGroups, gM2L.maxIdxDim,
      gM2L.d_groupStart, gM2L.d_groupCount, gM2L.d_groupDst, gM2L.d_groupOrder,
      gM2L.d_pairSrc, gM2L.d_pairCoeffOffset,
      gM2L.d_cubeCoeffOffset,
      gM2L.d_idxI1, gM2L.d_idxI2, gM2L.d_idxI3, gM2L.d_idx3Flat, gM2L.d_sgn3,
      gM2L.d_g0, gM2L.d_gk, epsilon,
      gM2L.d_momPot, gM2L.d_momDpdn,
      gM2L.d_lec1, gM2L.d_lec2, gM2L.d_lec3, gM2L.d_lec4);
  err = cudaGetLastError();
  if (err != cudaSuccess) return 0;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 0;

  err = cudaMemcpy(gM2L.h_lec1, gM2L.d_lec1,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gM2L.h_lec2, gM2L.d_lec2,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gM2L.h_lec3, gM2L.d_lec3,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gM2L.h_lec4, gM2L.d_lec4,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;

  for (cubeIdx = 0; cubeIdx < sys->nFmmCubesFlat; cubeIdx++) {
    cube *cb = sys->fmmCubeByIdx[cubeIdx];
    int offset = gM2L.h_cubeCoeffOffset[cubeIdx];
    int nMom = gM2L.h_cubeNMom[cubeIdx];
    memcpy(cb->lec_k1, &gM2L.h_lec1[offset], (size_t)nMom * sizeof(double));
    memcpy(cb->lec_k2, &gM2L.h_lec2[offset], (size_t)nMom * sizeof(double));
    memcpy(cb->lec_k3, &gM2L.h_lec3[offset], (size_t)nMom * sizeof(double));
    memcpy(cb->lec_k4, &gM2L.h_lec4[offset], (size_t)nMom * sizeof(double));
  }

  return 1;
}

int gpuQ2MApply(ssystem *sys, const double *sgm) {
  int cubeIdx;
  cube *cb;
  cudaError_t err;

  if (sys == NULL || sgm == NULL) {
    return 0;
  }
  if (gLeaf.sys != sys) {
    freeLeafCache();
    if (!buildLeafTables(sys)) {
      freeLeafCache();
      return 0;
    }
  }

  err = cudaMemcpy(gLeaf.d_sgm, sgm,
                   (size_t)(2 * sys->nPnls) * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;

  q2mLeafKernel<<<gLeaf.nLeaves, gLeaf.nMom>>>(
      sys->nPnls, gLeaf.nMom,
      gLeaf.d_leafPanelStart, gLeaf.d_leafPanelCount, gLeaf.d_leafMatrixOffset,
      gLeaf.d_q2m0, gLeaf.d_q2m1,
      gLeaf.d_sgm, gLeaf.d_momPot, gLeaf.d_momDpdn);
  err = cudaGetLastError();
  if (err != cudaSuccess) return 0;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 0;

  err = cudaMemcpy(gLeaf.h_momPot, gLeaf.d_momPot,
                   (size_t)gLeaf.nLeaves * (size_t)gLeaf.nMom * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gLeaf.h_momDpdn, gLeaf.d_momDpdn,
                   (size_t)gLeaf.nLeaves * (size_t)gLeaf.nMom * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;

  for (cubeIdx = 0, cb = sys->cubeList[sys->depth];
       cubeIdx < sys->nLeafCubesFlat && cb != NULL;
       cubeIdx++, cb = cb->next) {
    memcpy(cb->mom_pot, &gLeaf.h_momPot[cubeIdx * gLeaf.nMom], (size_t)gLeaf.nMom * sizeof(double));
    memcpy(cb->mom_dpdn, &gLeaf.h_momDpdn[cubeIdx * gLeaf.nMom], (size_t)gLeaf.nMom * sizeof(double));
  }
  return 1;
}

int gpuL2PApply(ssystem *sys, double alpha, double beta, double *pot) {
  int cubeIdx;
  cudaError_t err;

  if (sys == NULL || pot == NULL) {
    return 0;
  }
  if (gLeaf.sys != sys) {
    freeLeafCache();
    if (!buildLeafTables(sys)) {
      freeLeafCache();
      return 0;
    }
  }

  {
    cube *cb = sys->cubeList[sys->depth];
    for (cubeIdx = 0; cubeIdx < sys->nLeafCubesFlat; cubeIdx++, cb = cb->next) {
      memcpy(&gLeaf.h_lec1[cubeIdx * gLeaf.nMom], cb->lec_k1, (size_t)gLeaf.nMom * sizeof(double));
      memcpy(&gLeaf.h_lec2[cubeIdx * gLeaf.nMom], cb->lec_k2, (size_t)gLeaf.nMom * sizeof(double));
      memcpy(&gLeaf.h_lec3[cubeIdx * gLeaf.nMom], cb->lec_k3, (size_t)gLeaf.nMom * sizeof(double));
      memcpy(&gLeaf.h_lec4[cubeIdx * gLeaf.nMom], cb->lec_k4, (size_t)gLeaf.nMom * sizeof(double));
    }
  }

  err = cudaMemcpy(gLeaf.d_pot, pot,
                   (size_t)(2 * sys->nPnls) * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gLeaf.d_lec1, gLeaf.h_lec1,
                   (size_t)gLeaf.nLeaves * (size_t)gLeaf.nMom * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gLeaf.d_lec2, gLeaf.h_lec2,
                   (size_t)gLeaf.nLeaves * (size_t)gLeaf.nMom * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gLeaf.d_lec3, gLeaf.h_lec3,
                   (size_t)gLeaf.nLeaves * (size_t)gLeaf.nMom * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gLeaf.d_lec4, gLeaf.h_lec4,
                   (size_t)gLeaf.nLeaves * (size_t)gLeaf.nMom * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;

  l2pLeafKernel<<<gLeaf.nLeaves, 256>>>(
      sys->nPnls, gLeaf.nMom,
      gLeaf.d_leafPanelStart, gLeaf.d_leafPanelCount, gLeaf.d_leafMatrixOffset,
      gLeaf.d_l2p0, gLeaf.d_l2p1,
      alpha, beta,
      gLeaf.d_lec1, gLeaf.d_lec2, gLeaf.d_lec3, gLeaf.d_lec4,
      gLeaf.d_pot);
  err = cudaGetLastError();
  if (err != cudaSuccess) return 0;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 0;

  err = cudaMemcpy(pot, gLeaf.d_pot,
                   (size_t)(2 * sys->nPnls) * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  return 1;
}

int gpuSetupRHS(ssystem *sys, int qOrder, double fac, double *sgm) {
  cudaError_t err;
  int blockSize;
  int gridSize;
  size_t sgmBytes;

  if (sys == NULL || sgm == NULL) {
    return 0;
  }

  if (gRhs.sys != sys || gRhs.qOrder != qOrder) {
    freeRhsCache();
    if (!buildRhsTables(sys, qOrder)) {
      freeRhsCache();
      return 0;
    }
  }

  sgmBytes = (size_t)(2 * gRhs.nPnls) * sizeof(double);
  err = cudaMemset(gRhs.d_sgm, 0, sgmBytes);
  if (err != cudaSuccess) return 0;

  blockSize = 256;
  gridSize = (gRhs.nPnls + blockSize - 1) / blockSize;
  rhsApplyKernel<<<gridSize, blockSize>>>(
      gRhs.nPnls, gRhs.nChar, gRhs.qOrder, fac,
      gRhs.d_panels, gRhs.d_chrPos, gRhs.d_chrVal, gRhs.d_sgm);
  err = cudaGetLastError();
  if (err != cudaSuccess) return 0;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 0;

  err = cudaMemcpy(sgm, gRhs.d_sgm, sgmBytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  return 1;
}
