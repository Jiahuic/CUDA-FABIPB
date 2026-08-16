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

namespace {
double wall_seconds_cuda_local() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

int nearCaseIndex(int nVtx) {
  switch (nVtx) {
    case 0: return 0;
    case 1: return 1;
    case 2: return 2;
    case -2: return 3;
    case 3: return 4;
    default: return -1;
  }
}

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
  /*
   * Cache of the streaming path's touching/self ("special") classification.
   *
   * Which pairs are special, and their panelIA0 coefficients, depend only on
   * the mesh -- not on sgm -- so the streaming path was recomputing an
   * identical answer on every matvec. Measured on a 420k-panel mesh that was
   * 78% of nearfield time single-threaded, and still 52% after threading.
   *
   * specialOffset holds one entry per chunk plus a terminator, so chunk c
   * occupies [specialOffset[c], specialOffset[c+1]). Chunk boundaries depend
   * only on geometry and chunkCapacity, so they repeat exactly every matvec.
   */
  /*
   * Flattened leaf-pair table, uploaded once, from which the streaming
   * disjoint kernel derives panel indices instead of having them shipped
   * every matvec. Per pair: the running interaction offset, and the panel
   * ranges of its source and destination leaves.
   */
  int streamPairCount;
  long long *d_streamPairOffset;
  int *d_streamPairSrcStart;
  int *d_streamPairSrcCount;
  int *d_streamPairDstStart;
  /* Whether each chunk staging buffer came from cudaHostAlloc (page-locked)
   * or plain malloc, so the right free is used. */
  int h_srcPinned;
  int h_dstPinned;
  int h_k0Pinned;
  int h_k1Pinned;
  int h_k2Pinned;
  int h_k3Pinned;
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

NearfieldGpuCache gNear = {};
char gNearfieldLastError[512] = "";

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

DirectGpuCache gDirect = {};

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

PrecondGpuCache gPrecond = {};
std::mutex gPrecondMutex;

void setNearfieldLastError(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(gNearfieldLastError, sizeof(gNearfieldLastError), fmt, ap);
  va_end(ap);
}

int envFlagEnabled(const char *name) {
  const char *value = getenv(name);
  return value != NULL && atoi(value) != 0;
}

double bytesToGiB(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

size_t nearfieldHostBytes(const ssystem *sys, long long nInteractions) {
  size_t ni = (size_t)nInteractions;
  size_t nNearPairs = (size_t)sys->nNearPairsFlat;
  size_t nLeaves = (size_t)sys->nLeafCubesFlat;
  size_t nPnls = (size_t)sys->nPnls;
  return nPnls * sizeof(NearPanelGeom) +
         2U * ni * sizeof(int) +
         nNearPairs * sizeof(int) +
         nNearPairs * sizeof(long long) +
         2U * nLeaves * sizeof(int) +
         (nLeaves + 1U) * sizeof(int) +
         4U * ni * sizeof(double);
}

size_t nearfieldDeviceBytes(const ssystem *sys, long long nInteractions) {
  size_t ni = (size_t)nInteractions;
  size_t nNearPairs = (size_t)sys->nNearPairsFlat;
  size_t nLeaves = (size_t)sys->nLeafCubesFlat;
  size_t nPnls = (size_t)sys->nPnls;
  size_t vecBytes = 2U * nPnls * sizeof(double);
  return nPnls * sizeof(NearPanelGeom) +
         2U * ni * sizeof(int) +
         nNearPairs * sizeof(int) +
         nNearPairs * sizeof(long long) +
         2U * nLeaves * sizeof(int) +
         (nLeaves + 1U) * sizeof(int) +
         4U * ni * sizeof(double) +
         2U * vecBytes;
}

int cudaMallocNearfield(void **ptr, size_t bytes, const char *name) {
  cudaError_t err = cudaMalloc(ptr, bytes);
  if (err != cudaSuccess) {
    setNearfieldLastError("cudaMalloc %s failed for %.3f GiB: %s",
                          name, bytesToGiB(bytes), cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

int cudaMemcpyNearfield(void *dst, const void *src, size_t bytes,
                        enum cudaMemcpyKind kind, const char *name) {
  cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
  if (err != cudaSuccess) {
    setNearfieldLastError("cudaMemcpy %s failed for %.3f GiB: %s",
                          name, bytesToGiB(bytes), cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

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
__constant__ double c_nearKappa;
__constant__ double c_nearEpsilon;

__global__ void nearfieldDisjointQ1BuildKernel(
    long long nInteractions,
    const NearPanelGeom *panels,
    const int *src,
    const int *dst,
    double *k0,
    double *k1,
    double *k2,
    double *k3);

__global__ void nearfieldDisjointQ1ApplyKernel(
    int nPnls,
    long long nInteractions,
    const NearPanelGeom *panels,
    const int *src,
    const int *dst,
    double alpha,
    const double *sgm,
    double *pot);

/* Bytes held per cached special pair: two indices plus four coefficients. */
#define NEAR_SPECIAL_CACHE_BYTES_PER_PAIR (2U * sizeof(int) + 4U * sizeof(double))

/*
 * Page-locked staging allocation, falling back to malloc. Pinned pages cannot
 * be swapped, so cudaHostAlloc can fail where malloc succeeds; the caller only
 * loses copy bandwidth, never correctness. *pinned records which was used so
 * the buffer is released with the matching call.
 */
static void *allocNearfieldStagingHost(size_t bytes, int *pinned) {
  void *p = NULL;

  *pinned = 0;
  if (bytes == 0U) {
    return NULL;
  }
  if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess && p != NULL) {
    *pinned = 1;
    return p;
  }
  cudaGetLastError();  /* clear the sticky error from the failed attempt */
  return malloc(bytes);
}

static void freeNearfieldStagingHost(void *p, int *pinned) {
  if (p == NULL) {
    *pinned = 0;
    return;
  }
  if (*pinned) {
    cudaFreeHost(p);
  } else {
    free(p);
  }
  *pinned = 0;
}

static void freeNearfieldSpecialCache(void) {
  delete gNear.specialOffset;
  delete gNear.cacheSrc;
  delete gNear.cacheDst;
  delete gNear.cacheK0;
  delete gNear.cacheK1;
  delete gNear.cacheK2;
  delete gNear.cacheK3;
  gNear.specialOffset = NULL;
  gNear.cacheSrc = NULL;
  gNear.cacheDst = NULL;
  gNear.cacheK0 = NULL;
  gNear.cacheK1 = NULL;
  gNear.cacheK2 = NULL;
  gNear.cacheK3 = NULL;
  gNear.specialCacheValid = 0;
  gNear.specialCacheChunks = 0;
}

/*
 * Ceiling on the special cache, in bytes. Defaults to 32 GiB, which covers the
 * 22.5 GiB the full virus mesh needs while leaving room for the GMRES
 * workspace and the panel array on a 256 GiB host. Set
 * FABIPB_GPU_NEARFIELD_SPECIAL_CACHE_MIB=0 to disable caching entirely and go
 * back to reclassifying every matvec.
 */
static size_t nearfieldSpecialCacheBudget(void) {
  const char *env = getenv("FABIPB_GPU_NEARFIELD_SPECIAL_CACHE_MIB");
  double mib = 32.0 * 1024.0;

  if (env != NULL && env[0] != '\0') {
    char *endptr = NULL;
    double v = strtod(env, &endptr);
    if (endptr != env && *endptr == '\0' && v >= 0.0) {
      mib = v;
    } else {
      fprintf(stderr,
              "Warning: ignoring invalid FABIPB_GPU_NEARFIELD_SPECIAL_CACHE_MIB='%s'\n",
              env);
    }
  }
  return (size_t)(mib * 1024.0 * 1024.0);
}

void freeNearfieldCache() {
  freeNearfieldSpecialCache();
  free(gNear.h_panels);
  /* The streaming path may have page-locked these; the cached path mallocs
   * them, in which case the pinned flags are 0 and this is a plain free. */
  freeNearfieldStagingHost(gNear.h_src, &gNear.h_srcPinned);
  freeNearfieldStagingHost(gNear.h_dst, &gNear.h_dstPinned);
  free(gNear.h_pairSrcCount);
  free(gNear.h_pairInteractionOffset);
  free(gNear.h_leafPanelStart);
  free(gNear.h_leafPanelCount);
  free(gNear.h_leafPairOffset);
  freeNearfieldStagingHost(gNear.h_k0, &gNear.h_k0Pinned);
  freeNearfieldStagingHost(gNear.h_k1, &gNear.h_k1Pinned);
  freeNearfieldStagingHost(gNear.h_k2, &gNear.h_k2Pinned);
  freeNearfieldStagingHost(gNear.h_k3, &gNear.h_k3Pinned);
  gNear.h_panels = NULL;
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

  cudaFree(gNear.d_panels);
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
  cudaFree(gNear.d_streamPairOffset);
  cudaFree(gNear.d_streamPairSrcStart);
  cudaFree(gNear.d_streamPairSrcCount);
  cudaFree(gNear.d_streamPairDstStart);
  gNear.d_streamPairOffset = NULL;
  gNear.d_streamPairSrcStart = NULL;
  gNear.d_streamPairSrcCount = NULL;
  gNear.d_streamPairDstStart = NULL;
  gNear.streamPairCount = 0;
  gNear.d_panels = NULL;
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
  gNear.streaming = 0;
  gNear.nInteractions = 0;
  gNear.chunkCapacity = 0;
}

void freeM2LCache() {
  free(gM2L.h_pairSrc);
  free(gM2L.h_pairCoeffOffset);
  free(gM2L.h_groupStart);
  free(gM2L.h_groupCount);
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
  gM2L.h_pairSrc = NULL;
  gM2L.h_pairCoeffOffset = NULL;
  gM2L.h_groupStart = NULL;
  gM2L.h_groupCount = NULL;
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
  gM2L.streaming = 0;
  gM2L.nCubes = 0;
  gM2L.nPairs = 0;
  gM2L.nGroups = 0;
  gM2L.maxOrder = 0;
  gM2L.maxIdxDim = 0;
  gM2L.totalPairCoeff = 0;
  gM2L.totalCubeCoeff = 0;
  gM2L.streamPairCapacity = 0;
  gM2L.streamGroupCapacity = 0;
  gM2L.streamCoeffCapacity = 0;
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
  gDirect.mode = 0;
  gDirect.blockDstCount = 0;
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

void freePrecondCache() {
  free(gPrecond.h_panels);
  free(gPrecond.h_src);
  free(gPrecond.h_dst);
  free(gPrecond.h_k0);
  free(gPrecond.h_k1);
  free(gPrecond.h_k2);
  free(gPrecond.h_k3);
  gPrecond.h_panels = NULL;
  gPrecond.h_src = NULL;
  gPrecond.h_dst = NULL;
  gPrecond.h_k0 = NULL;
  gPrecond.h_k1 = NULL;
  gPrecond.h_k2 = NULL;
  gPrecond.h_k3 = NULL;

  cudaFree(gPrecond.d_panels);
  cudaFree(gPrecond.d_src);
  cudaFree(gPrecond.d_dst);
  cudaFree(gPrecond.d_k0);
  cudaFree(gPrecond.d_k1);
  cudaFree(gPrecond.d_k2);
  cudaFree(gPrecond.d_k3);
  gPrecond.d_panels = NULL;
  gPrecond.d_src = NULL;
  gPrecond.d_dst = NULL;
  gPrecond.d_k0 = NULL;
  gPrecond.d_k1 = NULL;
  gPrecond.d_k2 = NULL;
  gPrecond.d_k3 = NULL;
  gPrecond.panelCapacity = 0;
  gPrecond.pairCapacity = 0;
}

int ensurePrecondCapacity(int nPanels, int nPairs) {
  if (nPanels > gPrecond.panelCapacity) {
    free(gPrecond.h_panels);
    cudaFree(gPrecond.d_panels);
    gPrecond.h_panels = (NearPanelGeom *)malloc((size_t)nPanels * sizeof(NearPanelGeom));
    if (gPrecond.h_panels == NULL) return 0;
    if (cudaMalloc((void **)&gPrecond.d_panels, (size_t)nPanels * sizeof(NearPanelGeom)) != cudaSuccess) {
      freePrecondCache();
      return 0;
    }
    gPrecond.panelCapacity = nPanels;
  }
  if (nPairs > gPrecond.pairCapacity) {
    free(gPrecond.h_src);
    free(gPrecond.h_dst);
    free(gPrecond.h_k0);
    free(gPrecond.h_k1);
    free(gPrecond.h_k2);
    free(gPrecond.h_k3);
    cudaFree(gPrecond.d_src);
    cudaFree(gPrecond.d_dst);
    cudaFree(gPrecond.d_k0);
    cudaFree(gPrecond.d_k1);
    cudaFree(gPrecond.d_k2);
    cudaFree(gPrecond.d_k3);
    gPrecond.h_src = (int *)malloc((size_t)nPairs * sizeof(int));
    gPrecond.h_dst = (int *)malloc((size_t)nPairs * sizeof(int));
    gPrecond.h_k0 = (double *)malloc((size_t)nPairs * sizeof(double));
    gPrecond.h_k1 = (double *)malloc((size_t)nPairs * sizeof(double));
    gPrecond.h_k2 = (double *)malloc((size_t)nPairs * sizeof(double));
    gPrecond.h_k3 = (double *)malloc((size_t)nPairs * sizeof(double));
    if (gPrecond.h_src == NULL || gPrecond.h_dst == NULL ||
        gPrecond.h_k0 == NULL || gPrecond.h_k1 == NULL ||
        gPrecond.h_k2 == NULL || gPrecond.h_k3 == NULL) {
      freePrecondCache();
      return 0;
    }
    if (cudaMalloc((void **)&gPrecond.d_src, (size_t)nPairs * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void **)&gPrecond.d_dst, (size_t)nPairs * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void **)&gPrecond.d_k0, (size_t)nPairs * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&gPrecond.d_k1, (size_t)nPairs * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&gPrecond.d_k2, (size_t)nPairs * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void **)&gPrecond.d_k3, (size_t)nPairs * sizeof(double)) != cudaSuccess) {
      freePrecondCache();
      return 0;
    }
    gPrecond.pairCapacity = nPairs;
  }
  return 1;
}

int allocateHostArrays(long long n) {
  size_t ni = (size_t)n;
  gNear.h_panels = (NearPanelGeom *)malloc((size_t)gNear.sys->nPnls * sizeof(NearPanelGeom));
  if (!gNear.h_panels) {
    setNearfieldLastError("host malloc h_panels failed for %.3f GiB",
                          bytesToGiB((size_t)gNear.sys->nPnls * sizeof(NearPanelGeom)));
    return 0;
  }
  gNear.h_src = (int *)malloc(ni * sizeof(int));
  if (!gNear.h_src) {
    setNearfieldLastError("host malloc h_src failed for %.3f GiB",
                          bytesToGiB(ni * sizeof(int)));
    return 0;
  }
  gNear.h_dst = (int *)malloc(ni * sizeof(int));
  if (!gNear.h_dst) {
    setNearfieldLastError("host malloc h_dst failed for %.3f GiB",
                          bytesToGiB(ni * sizeof(int)));
    return 0;
  }
  gNear.h_pairSrcCount = (int *)malloc((size_t)gNear.sys->nNearPairsFlat * sizeof(int));
  if (!gNear.h_pairSrcCount) {
    setNearfieldLastError("host malloc h_pairSrcCount failed for %.3f GiB",
                          bytesToGiB((size_t)gNear.sys->nNearPairsFlat * sizeof(int)));
    return 0;
  }
  gNear.h_pairInteractionOffset = (long long *)malloc((size_t)gNear.sys->nNearPairsFlat * sizeof(long long));
  if (!gNear.h_pairInteractionOffset) {
    setNearfieldLastError("host malloc h_pairInteractionOffset failed for %.3f GiB",
                          bytesToGiB((size_t)gNear.sys->nNearPairsFlat * sizeof(long long)));
    return 0;
  }
  gNear.h_leafPanelStart = (int *)malloc((size_t)gNear.sys->nLeafCubesFlat * sizeof(int));
  if (!gNear.h_leafPanelStart) {
    setNearfieldLastError("host malloc h_leafPanelStart failed for %.3f GiB",
                          bytesToGiB((size_t)gNear.sys->nLeafCubesFlat * sizeof(int)));
    return 0;
  }
  gNear.h_leafPanelCount = (int *)malloc((size_t)gNear.sys->nLeafCubesFlat * sizeof(int));
  if (!gNear.h_leafPanelCount) {
    setNearfieldLastError("host malloc h_leafPanelCount failed for %.3f GiB",
                          bytesToGiB((size_t)gNear.sys->nLeafCubesFlat * sizeof(int)));
    return 0;
  }
  gNear.h_leafPairOffset = (int *)calloc((size_t)gNear.sys->nLeafCubesFlat + 1, sizeof(int));
  if (!gNear.h_leafPairOffset) {
    setNearfieldLastError("host calloc h_leafPairOffset failed for %.3f GiB",
                          bytesToGiB(((size_t)gNear.sys->nLeafCubesFlat + 1U) * sizeof(int)));
    return 0;
  }
  gNear.h_k0 = (double *)malloc(ni * sizeof(double));
  if (!gNear.h_k0) {
    setNearfieldLastError("host malloc h_k0 failed for %.3f GiB",
                          bytesToGiB(ni * sizeof(double)));
    return 0;
  }
  gNear.h_k1 = (double *)malloc(ni * sizeof(double));
  if (!gNear.h_k1) {
    setNearfieldLastError("host malloc h_k1 failed for %.3f GiB",
                          bytesToGiB(ni * sizeof(double)));
    return 0;
  }
  gNear.h_k2 = (double *)malloc(ni * sizeof(double));
  if (!gNear.h_k2) {
    setNearfieldLastError("host malloc h_k2 failed for %.3f GiB",
                          bytesToGiB(ni * sizeof(double)));
    return 0;
  }
  gNear.h_k3 = (double *)malloc(ni * sizeof(double));
  if (!gNear.h_k3) {
    setNearfieldLastError("host malloc h_k3 failed for %.3f GiB",
                          bytesToGiB(ni * sizeof(double)));
    return 0;
  }
  if (!gNear.h_panels || !gNear.h_src || !gNear.h_dst || !gNear.h_pairSrcCount ||
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

  if (!cudaMallocNearfield((void **)&gNear.d_panels,
                           (size_t)gNear.sys->nPnls * sizeof(NearPanelGeom),
                           "d_panels")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_src, ni * sizeof(int), "d_src")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_dst, ni * sizeof(int), "d_dst")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_pairSrcCount,
                           (size_t)gNear.sys->nNearPairsFlat * sizeof(int),
                           "d_pairSrcCount")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_pairInteractionOffset,
                           (size_t)gNear.sys->nNearPairsFlat * sizeof(long long),
                           "d_pairInteractionOffset")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_leafPanelStart,
                           (size_t)gNear.sys->nLeafCubesFlat * sizeof(int),
                           "d_leafPanelStart")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_leafPanelCount,
                           (size_t)gNear.sys->nLeafCubesFlat * sizeof(int),
                           "d_leafPanelCount")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_leafPairOffset,
                           ((size_t)gNear.sys->nLeafCubesFlat + 1U) * sizeof(int),
                           "d_leafPairOffset")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_k0, ni * sizeof(double), "d_k0")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_k1, ni * sizeof(double), "d_k1")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_k2, ni * sizeof(double), "d_k2")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_k3, ni * sizeof(double), "d_k3")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_sgm, vecBytes, "d_sgm")) return 0;
  if (!cudaMallocNearfield((void **)&gNear.d_pot, vecBytes, "d_pot")) return 0;

  return 1;
}

int buildNearfieldPanelGeometry(const ssystem *sys) {
  int i, j, k;

  for (i = 0; i < sys->nPnls; i++) {
    panel *pnl = sys->panelByIdx[i];
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
        gNear.h_panels[i].vtx[j][k] = pnl->vtx[j][k];
      }
    }
    for (k = 0; k < 3; k++) {
      gNear.h_panels[i].a0[k] = pnl->a[0][k];
      gNear.h_panels[i].a1[k] = pnl->a[1][k];
      gNear.h_panels[i].a2[k] = pnl->a[2][k];
      gNear.h_panels[i].normal[k] = pnl->normal[k];
    }
    gNear.h_panels[i].area = pnl->area;
  }
  return 1;
}

long long nearfieldStreamingCapacity(size_t freeBytes) {
  const char *env = getenv("FABIPB_GPU_NEARFIELD_CHUNK_MIB");
  size_t requestedBytes = 512U * 1024U * 1024U;
  const size_t bytesPerInteraction = 2U * sizeof(int) + 4U * sizeof(double);
  size_t reserveBytes = 256U * 1024U * 1024U;

  if (env != NULL) {
    double mib = atof(env);
    if (mib > 0.0) {
      requestedBytes = (size_t)(mib * 1024.0 * 1024.0);
    }
  }
  if (freeBytes > reserveBytes && requestedBytes > freeBytes - reserveBytes) {
    requestedBytes = freeBytes - reserveBytes;
  }
  if (requestedBytes < bytesPerInteraction * 1024U) {
    return 0;
  }
  return (long long)(requestedBytes / bytesPerInteraction);
}

int buildNearfieldStreamingCache(const ssystem *sys, long long totalInteractions,
                                 size_t freeBytes) {
  long long capacity = nearfieldStreamingCapacity(freeBytes);
  size_t ni;
  size_t vecBytes;

  if (sys->maxQuadOrder != 1) {
    setNearfieldLastError("streaming nearfield currently requires qOrd=1 (got %d)",
                          sys->maxQuadOrder);
    return 0;
  }
  if (capacity <= 0) {
    setNearfieldLastError("not enough free GPU memory for a nearfield streaming chunk");
    return 0;
  }

  gNear.sys = sys;
  gNear.nPnls = sys->nPnls;
  gNear.nearfieldMode = 0;
  gNear.streaming = 1;
  gNear.nInteractions = totalInteractions;
  gNear.chunkCapacity = capacity;
  freeNearfieldSpecialCache();
  gNear.specialCacheEnabled = (nearfieldSpecialCacheBudget() > 0U) ? 1 : 0;
  if (gNear.specialCacheEnabled) {
    gNear.specialOffset = new std::vector<long long>();
    gNear.cacheSrc = new std::vector<int>();
    gNear.cacheDst = new std::vector<int>();
    gNear.cacheK0 = new std::vector<double>();
    gNear.cacheK1 = new std::vector<double>();
    gNear.cacheK2 = new std::vector<double>();
    gNear.cacheK3 = new std::vector<double>();
    gNear.specialOffset->push_back(0);
  }
  ni = (size_t)capacity;
  vecBytes = (size_t)(2 * sys->nPnls) * sizeof(double);

  gNear.h_panels = (NearPanelGeom *)malloc((size_t)sys->nPnls * sizeof(NearPanelGeom));
  /*
   * The chunk staging buffers are page-locked. Every chunk copies its indices
   * to the device on every matvec -- 3.8 GB per matvec on a 420k-panel mesh --
   * and out of pageable memory the driver has to bounce that through an
   * internal staging buffer, which measured 4.0 s of an 11.0 s nearfield
   * stage. Pinning lets the DMA engine read the buffer directly.
   *
   * cudaHostAlloc can fail where malloc would succeed, since pinned pages
   * cannot be swapped, so each falls back to ordinary memory rather than
   * failing the run; correctness does not depend on it.
   */
  gNear.h_src = (int *)allocNearfieldStagingHost(ni * sizeof(int), &gNear.h_srcPinned);
  gNear.h_dst = (int *)allocNearfieldStagingHost(ni * sizeof(int), &gNear.h_dstPinned);
  gNear.h_k0 = (double *)allocNearfieldStagingHost(ni * sizeof(double), &gNear.h_k0Pinned);
  gNear.h_k1 = (double *)allocNearfieldStagingHost(ni * sizeof(double), &gNear.h_k1Pinned);
  gNear.h_k2 = (double *)allocNearfieldStagingHost(ni * sizeof(double), &gNear.h_k2Pinned);
  gNear.h_k3 = (double *)allocNearfieldStagingHost(ni * sizeof(double), &gNear.h_k3Pinned);
  if (gNear.h_panels == NULL || gNear.h_src == NULL || gNear.h_dst == NULL ||
      gNear.h_k0 == NULL || gNear.h_k1 == NULL || gNear.h_k2 == NULL ||
      gNear.h_k3 == NULL) {
    setNearfieldLastError("host allocation for %.3f GiB nearfield streaming chunk failed",
                          bytesToGiB(ni * (2U * sizeof(int) + 4U * sizeof(double))));
    return 0;
  }
  if (sys->benchmarkMode > 0) {
    printf("GPU nearfield streaming staging: pinned=%d/6 chunk-host=%.3f GiB\n",
           gNear.h_srcPinned + gNear.h_dstPinned + gNear.h_k0Pinned +
           gNear.h_k1Pinned + gNear.h_k2Pinned + gNear.h_k3Pinned,
           bytesToGiB(ni * (2U * sizeof(int) + 4U * sizeof(double))));
  }
  if (!buildNearfieldPanelGeometry(sys)) return 0;

  if (!cudaMallocNearfield((void **)&gNear.d_panels,
                           (size_t)sys->nPnls * sizeof(NearPanelGeom), "stream d_panels") ||
      !cudaMallocNearfield((void **)&gNear.d_src, ni * sizeof(int), "stream d_src") ||
      !cudaMallocNearfield((void **)&gNear.d_dst, ni * sizeof(int), "stream d_dst") ||
      !cudaMallocNearfield((void **)&gNear.d_k0, ni * sizeof(double), "stream d_k0") ||
      !cudaMallocNearfield((void **)&gNear.d_k1, ni * sizeof(double), "stream d_k1") ||
      !cudaMallocNearfield((void **)&gNear.d_k2, ni * sizeof(double), "stream d_k2") ||
      !cudaMallocNearfield((void **)&gNear.d_k3, ni * sizeof(double), "stream d_k3") ||
      !cudaMallocNearfield((void **)&gNear.d_sgm, vecBytes, "stream d_sgm") ||
      !cudaMallocNearfield((void **)&gNear.d_pot, vecBytes, "stream d_pot")) {
    return 0;
  }
  if (!cudaMemcpyNearfield(gNear.d_panels, gNear.h_panels,
                           (size_t)sys->nPnls * sizeof(NearPanelGeom),
                           cudaMemcpyHostToDevice, "stream panels")) return 0;

  /*
   * Build and upload the leaf-pair table once. It lets the disjoint kernel
   * derive its own panel indices, so from the second matvec onward neither the
   * host index loop nor the per-chunk index upload has to run at all. The
   * table is ~20 bytes per leaf pair, against 8 bytes per interaction for the
   * arrays it replaces.
   */
  {
    int nPairs = sys->nNearPairsFlat;
    std::vector<long long> hOffset((size_t)nPairs + 1U);
    std::vector<int> hSrcStart((size_t)nPairs), hSrcCount((size_t)nPairs),
                     hDstStart((size_t)nPairs);
    long long running = 0;
    int p;

    for (p = 0; p < nPairs; p++) {
      int srcLeaf = sys->nearPairSrc[p];
      int dstLeaf = sys->nearPairDst[p];
      hOffset[(size_t)p] = running;
      hSrcStart[(size_t)p] = sys->leafPanelStart[srcLeaf];
      hSrcCount[(size_t)p] = sys->leafPanelCount[srcLeaf];
      hDstStart[(size_t)p] = sys->leafPanelStart[dstLeaf];
      running += (long long)sys->leafPanelCount[srcLeaf] *
                 (long long)sys->leafPanelCount[dstLeaf];
    }
    hOffset[(size_t)nPairs] = running;

    if (running != totalInteractions) {
      setNearfieldLastError("leaf-pair table totals %lld interactions, expected %lld",
                            running, totalInteractions);
      return 0;
    }

    if (!cudaMallocNearfield((void **)&gNear.d_streamPairOffset,
                             ((size_t)nPairs + 1U) * sizeof(long long), "stream pairOffset") ||
        !cudaMallocNearfield((void **)&gNear.d_streamPairSrcStart,
                             (size_t)nPairs * sizeof(int), "stream pairSrcStart") ||
        !cudaMallocNearfield((void **)&gNear.d_streamPairSrcCount,
                             (size_t)nPairs * sizeof(int), "stream pairSrcCount") ||
        !cudaMallocNearfield((void **)&gNear.d_streamPairDstStart,
                             (size_t)nPairs * sizeof(int), "stream pairDstStart")) {
      return 0;
    }
    if (!cudaMemcpyNearfield(gNear.d_streamPairOffset, &hOffset[0],
                             ((size_t)nPairs + 1U) * sizeof(long long),
                             cudaMemcpyHostToDevice, "stream pairOffset") ||
        !cudaMemcpyNearfield(gNear.d_streamPairSrcStart, &hSrcStart[0],
                             (size_t)nPairs * sizeof(int),
                             cudaMemcpyHostToDevice, "stream pairSrcStart") ||
        !cudaMemcpyNearfield(gNear.d_streamPairSrcCount, &hSrcCount[0],
                             (size_t)nPairs * sizeof(int),
                             cudaMemcpyHostToDevice, "stream pairSrcCount") ||
        !cudaMemcpyNearfield(gNear.d_streamPairDstStart, &hDstStart[0],
                             (size_t)nPairs * sizeof(int),
                             cudaMemcpyHostToDevice, "stream pairDstStart")) {
      return 0;
    }
    gNear.streamPairCount = nPairs;
  }

  if (sys->benchmarkMode > 0) {
    printf("GPU nearfield streaming: interactions=%lld chunk-capacity=%lld chunk-device=%.3f GiB qOrd=1\n",
           totalInteractions, capacity,
           bytesToGiB(ni * (2U * sizeof(int) + 4U * sizeof(double))));
    printf("GPU nearfield streaming pair table: pairs=%d device=%.3f GiB\n",
           gNear.streamPairCount,
           bytesToGiB(((size_t)gNear.streamPairCount + 1U) * sizeof(long long) +
                      3U * (size_t)gNear.streamPairCount * sizeof(int)));
  }
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

int allocateBlockedDirectArrays(int nPnls, int dstBlockCount) {
  size_t ni = (size_t)nPnls * (size_t)dstBlockCount;
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

int allocateM2LStreamingArrays(size_t chunkBytes) {
  size_t nCubes = (size_t)gM2L.nCubes;
  size_t nMom = (size_t)gM2L.totalCubeCoeff;
  size_t idxFlat = (size_t)gM2L.maxIdxDim * (size_t)gM2L.maxIdxDim *
                   (size_t)gM2L.maxIdxDim;
  size_t maxMom = (size_t)(((gM2L.maxOrder + 1) * (gM2L.maxOrder + 2) *
                            (gM2L.maxOrder + 3)) / 6);
  size_t coeffCapacity = (chunkBytes * 3U / 4U) / (2U * sizeof(double));
  size_t pairCapacity = (chunkBytes / 4U) / (2U * sizeof(int));
  size_t groupCapacity = std::min((size_t)gM2L.nGroups, (size_t)1048576U);

  if (coeffCapacity > (size_t)INT_MAX) coeffCapacity = (size_t)INT_MAX;
  if (pairCapacity > (size_t)INT_MAX) pairCapacity = (size_t)INT_MAX;
  if (coeffCapacity == 0 || pairCapacity == 0 || groupCapacity == 0) return 0;
  gM2L.streamCoeffCapacity = (int)coeffCapacity;
  gM2L.streamPairCapacity = (int)pairCapacity;
  gM2L.streamGroupCapacity = (int)groupCapacity;

  gM2L.h_pairSrc = (int *)malloc(pairCapacity * sizeof(int));
  gM2L.h_pairCoeffOffset = (int *)malloc((pairCapacity + 1U) * sizeof(int));
  gM2L.h_groupStart = (int *)malloc(groupCapacity * sizeof(int));
  gM2L.h_groupCount = (int *)malloc(groupCapacity * sizeof(int));
  gM2L.h_groupDst = (int *)malloc(groupCapacity * sizeof(int));
  gM2L.h_groupOrder = (int *)malloc(groupCapacity * sizeof(int));
  gM2L.h_cubeCoeffOffset = (int *)malloc(nCubes * sizeof(int));
  gM2L.h_cubeNMom = (int *)malloc(nCubes * sizeof(int));
  gM2L.h_idxI1 = (int *)malloc(maxMom * sizeof(int));
  gM2L.h_idxI2 = (int *)malloc(maxMom * sizeof(int));
  gM2L.h_idxI3 = (int *)malloc(maxMom * sizeof(int));
  gM2L.h_idx3Flat = (int *)malloc(idxFlat * sizeof(int));
  gM2L.h_g0 = (double *)malloc(coeffCapacity * sizeof(double));
  gM2L.h_gk = (double *)malloc(coeffCapacity * sizeof(double));
  gM2L.h_momPot = (double *)malloc(nMom * sizeof(double));
  gM2L.h_momDpdn = (double *)malloc(nMom * sizeof(double));
  gM2L.h_lec1 = (double *)malloc(nMom * sizeof(double));
  gM2L.h_lec2 = (double *)malloc(nMom * sizeof(double));
  gM2L.h_lec3 = (double *)malloc(nMom * sizeof(double));
  gM2L.h_lec4 = (double *)malloc(nMom * sizeof(double));
  if (!gM2L.h_pairSrc || !gM2L.h_pairCoeffOffset ||
      !gM2L.h_groupStart || !gM2L.h_groupCount ||
      !gM2L.h_groupDst || !gM2L.h_groupOrder || !gM2L.h_cubeCoeffOffset ||
      !gM2L.h_cubeNMom || !gM2L.h_idxI1 || !gM2L.h_idxI2 || !gM2L.h_idxI3 ||
      !gM2L.h_idx3Flat || !gM2L.h_g0 || !gM2L.h_gk || !gM2L.h_momPot ||
      !gM2L.h_momDpdn || !gM2L.h_lec1 || !gM2L.h_lec2 ||
      !gM2L.h_lec3 || !gM2L.h_lec4) return 0;

#define M2L_STREAM_CUDA_ALLOC(ptr, count, type) \
  do { if (cudaMalloc((void **)&(ptr), (count) * sizeof(type)) != cudaSuccess) return 0; } while (0)
  M2L_STREAM_CUDA_ALLOC(gM2L.d_pairSrc, pairCapacity, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_pairCoeffOffset, pairCapacity + 1U, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_groupStart, groupCapacity, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_groupCount, groupCapacity, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_groupDst, groupCapacity, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_groupOrder, groupCapacity, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_cubeCoeffOffset, nCubes, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_cubeNMom, nCubes, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_idxI1, maxMom, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_idxI2, maxMom, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_idxI3, maxMom, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_idx3Flat, idxFlat, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_sgn3, maxMom, int);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_g0, coeffCapacity, double);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_gk, coeffCapacity, double);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_momPot, nMom, double);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_momDpdn, nMom, double);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_lec1, nMom, double);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_lec2, nMom, double);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_lec3, nMom, double);
  M2L_STREAM_CUDA_ALLOC(gM2L.d_lec4, nMom, double);
#undef M2L_STREAM_CUDA_ALLOC
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

int directBuildThreadCount(int nTasks) {
  const char *env = getenv("FABIPB_DIRECT_THREADS");
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

int gpuNearfieldDisjointBuildEnabled(const ssystem *sys) {
  const char *env = getenv("FABIPB_GPU_NEARFIELD_BUILD_DISJOINT");
  if (sys->maxQuadOrder != 1) {
    return 0;
  }
  if (env != NULL && atoi(env) == 0) {
    return 0;
  }
  return 1;
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

size_t m2lStreamingChunkBytes() {
  const char *env = getenv("FABIPB_GPU_M2L_CHUNK_MIB");
  double mib = 512.0;
  if (env != NULL && atof(env) > 0.0) mib = atof(env);
  return (size_t)(mib * 1024.0 * 1024.0);
}

size_t m2lFullDeviceBytes() {
  size_t nPairs = (size_t)gM2L.nPairs;
  size_t nGroups = (size_t)gM2L.nGroups;
  size_t nCubes = (size_t)gM2L.nCubes;
  size_t nMom = (size_t)gM2L.totalCubeCoeff;
  size_t nPairCoeff = (size_t)gM2L.totalPairCoeff;
  return 2U * nPairCoeff * sizeof(double) +
         2U * nPairs * sizeof(int) +
         4U * nGroups * sizeof(int) +
         2U * nCubes * sizeof(int) +
         6U * nMom * sizeof(double);
}

int buildM2LStreamingTables(const ssystem *sys) {
  int cubeIdx;
  size_t chunkBytes = m2lStreamingChunkBytes();
  size_t nCubes = (size_t)gM2L.nCubes;
  size_t maxMom = (size_t)sys->nMom[sys->maxOrder];
  size_t idxFlat = (size_t)gM2L.maxIdxDim * (size_t)gM2L.maxIdxDim *
                   (size_t)gM2L.maxIdxDim;

  if (gM2L.totalCubeCoeff > INT_MAX) {
    printf("GPU M2L streaming unavailable: cube coefficient offsets exceed INT_MAX\n");
    return 0;
  }
  gM2L.streaming = 1;
  if (!allocateM2LStreamingArrays(chunkBytes)) return 0;
  buildM2LIndexTables();

  for (cubeIdx = 0; cubeIdx < gM2L.nCubes; cubeIdx++) {
    cube *cb = sys->fmmCubeByIdx[cubeIdx];
    int nMom = sys->nMom[sys->ordM2L[cb->level]];
    if (cubeIdx == 0) {
      gM2L.h_cubeCoeffOffset[cubeIdx] = 0;
    } else {
      gM2L.h_cubeCoeffOffset[cubeIdx] =
          gM2L.h_cubeCoeffOffset[cubeIdx - 1] + gM2L.h_cubeNMom[cubeIdx - 1];
    }
    gM2L.h_cubeNMom[cubeIdx] = nMom;
  }

#define M2L_STREAM_UPLOAD(dst, src, count, type) \
  do { if (cudaMemcpy((dst), (src), (count) * sizeof(type), cudaMemcpyHostToDevice) != cudaSuccess) return 0; } while (0)
  M2L_STREAM_UPLOAD(gM2L.d_cubeCoeffOffset, gM2L.h_cubeCoeffOffset, nCubes, int);
  M2L_STREAM_UPLOAD(gM2L.d_cubeNMom, gM2L.h_cubeNMom, nCubes, int);
  M2L_STREAM_UPLOAD(gM2L.d_idxI1, gM2L.h_idxI1, maxMom, int);
  M2L_STREAM_UPLOAD(gM2L.d_idxI2, gM2L.h_idxI2, maxMom, int);
  M2L_STREAM_UPLOAD(gM2L.d_idxI3, gM2L.h_idxI3, maxMom, int);
  M2L_STREAM_UPLOAD(gM2L.d_idx3Flat, gM2L.h_idx3Flat, idxFlat, int);
  M2L_STREAM_UPLOAD(gM2L.d_sgn3, sgn3, maxMom, int);
#undef M2L_STREAM_UPLOAD

  if (sys->benchmarkMode > 0) {
    printf("GPU M2L streaming cache: cubes=%d pairs=%d coeff=%lld pair-capacity=%d coeff-capacity=%d group-capacity=%d device-chunk=%.3f GiB\n",
           gM2L.nCubes, gM2L.nPairs, gM2L.totalPairCoeff,
           gM2L.streamPairCapacity, gM2L.streamCoeffCapacity,
           gM2L.streamGroupCapacity, bytesToGiB(chunkBytes));
  }
  return 1;
}

int buildM2LTables(const ssystem *sys) {
  int cubeIdx;
  int pairIdx;
  int groupIdx;
  double r[3];

  gM2L.sys = sys;
  gM2L.nCubes = sys->nFmmCubesFlat;
  gM2L.nPairs = sys->nM2LPairsFlat;
  gM2L.nGroups = sys->nM2LDstGroups;
  gM2L.maxOrder = sys->maxOrder;
  gM2L.maxIdxDim = sys->maxOrder + 1;
  gM2L.totalPairCoeff = 0;
  gM2L.totalCubeCoeff = 0;
  gM2L.streaming = 0;

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
  {
    size_t freeBytes = 0, totalBytes = 0;
    size_t fullBytes = m2lFullDeviceBytes();
    int forceStreaming = envFlagEnabled("FABIPB_GPU_M2L_FORCE_STREAMING");
    int offsetOverflow = gM2L.totalPairCoeff > INT_MAX;
    int memoryOverflow = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
      memoryOverflow = fullBytes > (freeBytes * 4U / 5U);
      if (sys->benchmarkMode > 0) {
        printf("GPU M2L estimate: coeff=%lld full-device=%.3f GiB free=%.3f GiB total=%.3f GiB offset64-required=%d\n",
               gM2L.totalPairCoeff, bytesToGiB(fullBytes), bytesToGiB(freeBytes),
               bytesToGiB(totalBytes), offsetOverflow);
      }
    }
    if (forceStreaming || offsetOverflow || memoryOverflow) {
      return buildM2LStreamingTables(sys);
    }
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
  long long disjointBuildCount = 0;
  size_t hostBytes = 0;
  size_t deviceBytes = 0;
  size_t freeBytes = 0;
  size_t totalGpuBytes = 0;
  double t0, t1;
  int collectCaseCounts = (sys->benchmarkMode > 0);
  int useGpuDisjointBuild = gpuNearfieldDisjointBuildEnabled(sys);

  setNearfieldLastError("not attempted");
  for (pairIdx = 0; pairIdx < sys->nNearPairsFlat; pairIdx++) {
    int srcLeaf = sys->nearPairSrc[pairIdx];
    int dstLeaf = sys->nearPairDst[pairIdx];
    long long srcCount = (long long)sys->leafPanelCount[srcLeaf];
    long long dstCount = (long long)sys->leafPanelCount[dstLeaf];
    totalInteractions += srcCount * dstCount;
  }

  if (totalInteractions <= 0) {
    setNearfieldLastError("nearfield cache has no interactions");
    return 0;
  }
  hostBytes = nearfieldHostBytes(sys, totalInteractions);
  deviceBytes = nearfieldDeviceBytes(sys, totalInteractions);
  if (cudaMemGetInfo(&freeBytes, &totalGpuBytes) == cudaSuccess) {
    if (sys->benchmarkMode > 0) {
      printf("GPU nearfield estimate: interactions=%lld host-cache=%.3f GiB device-cache=%.3f GiB free=%.3f GiB total=%.3f GiB mode=%d\n",
             totalInteractions, bytesToGiB(hostBytes), bytesToGiB(deviceBytes),
             bytesToGiB(freeBytes), bytesToGiB(totalGpuBytes), sys->gpuNearfieldMode);
    }
    if (envFlagEnabled("FABIPB_GPU_NEARFIELD_FORCE_STREAMING")) {
      return buildNearfieldStreamingCache(sys, totalInteractions, freeBytes);
    }
    if (deviceBytes > freeBytes && !envFlagEnabled("FABIPB_GPU_NEARFIELD_ALLOW_OVERSIZE")) {
      if (!envFlagEnabled("FABIPB_GPU_NEARFIELD_DISABLE_STREAMING")) {
        return buildNearfieldStreamingCache(sys, totalInteractions, freeBytes);
      }
      setNearfieldLastError("estimated device cache %.3f GiB exceeds free GPU memory %.3f GiB and streaming is disabled",
                            bytesToGiB(deviceBytes), bytesToGiB(freeBytes));
      return 0;
    }
  } else if (sys->benchmarkMode > 0) {
    printf("GPU nearfield estimate: interactions=%lld host-cache=%.3f GiB device-cache=%.3f GiB free=unknown mode=%d\n",
           totalInteractions, bytesToGiB(hostBytes), bytesToGiB(deviceBytes),
           sys->gpuNearfieldMode);
  }
  gNear.sys = sys;
  gNear.nearfieldMode = sys->gpuNearfieldMode;
  gNear.streaming = 0;
  gNear.caseDisjointCount = 0;
  gNear.caseOneCommonCount = 0;
  gNear.caseTwoCommonCount = 0;
  gNear.caseTwoCommonRevCount = 0;
  gNear.caseSelfCount = 0;
  t0 = wall_seconds_cuda_local();
  if (!allocateHostArrays(totalInteractions)) {
    return 0;
  }
  if (!buildNearfieldPanelGeometry(sys)) {
    setNearfieldLastError("nearfield panel geometry build failed");
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
            int idxX[3], idxY[3];
            int nVtx = 0;
            if (collectCaseCounts || useGpuDisjointBuild) {
              nVtx = nrCommonVtx(pnlX, pnlY, idxX, idxY);
            }
            if (collectCaseCounts) {
              if (nVtx == 0) {
                gNear.caseDisjointCount++;
              } else {
                int caseIdx = nearCaseIndex(nVtx);
                switch (caseIdx) {
                  case 1: gNear.caseOneCommonCount++; break;
                  case 2: gNear.caseTwoCommonCount++; break;
                  case 3: gNear.caseTwoCommonRevCount++; break;
                  case 4: gNear.caseSelfCount++; break;
                  default: break;
                }
              }
            }
            gNear.h_dst[idx] = dstPanelIdx;
            gNear.h_src[idx] = srcPanelIdx;
            if (useGpuDisjointBuild && nVtx == 0) {
              disjointBuildCount++;
              gNear.h_k0[idx] = 0.0;
              gNear.h_k1[idx] = 0.0;
              gNear.h_k2[idx] = 0.0;
              gNear.h_k3[idx] = 0.0;
            } else {
              double *KER = panelIA0(pnlX, pnlY);
              gNear.h_k0[idx] = KER[0];
              gNear.h_k1[idx] = KER[1];
              gNear.h_k2[idx] = KER[2];
              gNear.h_k3[idx] = KER[3];
            }
          }
        }
      }
    } else {
      std::vector<std::thread> workers;
      std::vector<long long> threadCounts((size_t)nThreads * 5U, 0);
      std::vector<long long> threadDisjointBuildCounts((size_t)nThreads, 0);
      int t;

      workers.reserve((size_t)nThreads);
      for (t = 0; t < nThreads; t++) {
        int begin = (sys->nNearPairsFlat * t) / nThreads;
        int end = (sys->nNearPairsFlat * (t + 1)) / nThreads;
        workers.emplace_back([=, &threadCounts, &threadDisjointBuildCounts]() {
          int localPairIdx;
          long long *localCounts = &threadCounts[(size_t)t * 5U];
          long long localDisjointBuildCount = 0;
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
                int idxX[3], idxY[3];
                int nVtx = 0;
                if (collectCaseCounts || useGpuDisjointBuild) {
                  nVtx = nrCommonVtx(pnlX, pnlY, idxX, idxY);
                }
                if (collectCaseCounts) {
                  if (nVtx == 0) {
                    localCounts[0]++;
                  } else {
                    int caseIdx = nearCaseIndex(nVtx);
                    if (caseIdx >= 1 && caseIdx <= 4) {
                      localCounts[caseIdx]++;
                    }
                  }
                }
                gNear.h_dst[idx] = dstPanelIdx;
                gNear.h_src[idx] = srcPanelIdx;
                if (useGpuDisjointBuild && nVtx == 0) {
                  localDisjointBuildCount++;
                  gNear.h_k0[idx] = 0.0;
                  gNear.h_k1[idx] = 0.0;
                  gNear.h_k2[idx] = 0.0;
                  gNear.h_k3[idx] = 0.0;
                } else {
                  double *KER = panelIA0(pnlX, pnlY);
                  gNear.h_k0[idx] = KER[0];
                  gNear.h_k1[idx] = KER[1];
                  gNear.h_k2[idx] = KER[2];
                  gNear.h_k3[idx] = KER[3];
                }
              }
            }
          }
          threadDisjointBuildCounts[(size_t)t] = localDisjointBuildCount;
        });
      }
      for (t = 0; t < nThreads; t++) {
        workers[(size_t)t].join();
        disjointBuildCount += threadDisjointBuildCounts[(size_t)t];
        if (collectCaseCounts) {
          long long *localCounts = &threadCounts[(size_t)t * 5U];
          gNear.caseDisjointCount += localCounts[0];
          gNear.caseOneCommonCount += localCounts[1];
          gNear.caseTwoCommonCount += localCounts[2];
          gNear.caseTwoCommonRevCount += localCounts[3];
          gNear.caseSelfCount += localCounts[4];
        }
      }
    }
  }
  t1 = wall_seconds_cuda_local();
  fmmNearGpuCoeffTime += (t1 - t0);

  gNear.nPnls = sys->nPnls;
  gNear.nInteractions = totalInteractions;
  t0 = wall_seconds_cuda_local();
  if (!allocateDeviceArrays(sys->nPnls, totalInteractions)) {
    return 0;
  }

  if (!cudaMemcpyNearfield(gNear.d_panels, gNear.h_panels,
                           (size_t)sys->nPnls * sizeof(NearPanelGeom),
                           cudaMemcpyHostToDevice, "h_panels -> d_panels")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_src, gNear.h_src,
                           (size_t)totalInteractions * sizeof(int),
                           cudaMemcpyHostToDevice, "h_src -> d_src")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_dst, gNear.h_dst,
                           (size_t)totalInteractions * sizeof(int),
                           cudaMemcpyHostToDevice, "h_dst -> d_dst")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_pairSrcCount, gNear.h_pairSrcCount,
                           (size_t)sys->nNearPairsFlat * sizeof(int),
                           cudaMemcpyHostToDevice, "h_pairSrcCount -> d_pairSrcCount")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_pairInteractionOffset, gNear.h_pairInteractionOffset,
                           (size_t)sys->nNearPairsFlat * sizeof(long long),
                           cudaMemcpyHostToDevice, "h_pairInteractionOffset -> d_pairInteractionOffset")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_leafPanelStart, gNear.h_leafPanelStart,
                           (size_t)sys->nLeafCubesFlat * sizeof(int),
                           cudaMemcpyHostToDevice, "h_leafPanelStart -> d_leafPanelStart")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_leafPanelCount, gNear.h_leafPanelCount,
                           (size_t)sys->nLeafCubesFlat * sizeof(int),
                           cudaMemcpyHostToDevice, "h_leafPanelCount -> d_leafPanelCount")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_leafPairOffset, gNear.h_leafPairOffset,
                           ((size_t)sys->nLeafCubesFlat + 1U) * sizeof(int),
                           cudaMemcpyHostToDevice, "h_leafPairOffset -> d_leafPairOffset")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_k0, gNear.h_k0,
                           (size_t)totalInteractions * sizeof(double),
                           cudaMemcpyHostToDevice, "h_k0 -> d_k0")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_k1, gNear.h_k1,
                           (size_t)totalInteractions * sizeof(double),
                           cudaMemcpyHostToDevice, "h_k1 -> d_k1")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_k2, gNear.h_k2,
                           (size_t)totalInteractions * sizeof(double),
                           cudaMemcpyHostToDevice, "h_k2 -> d_k2")) return 0;
  if (!cudaMemcpyNearfield(gNear.d_k3, gNear.h_k3,
                           (size_t)totalInteractions * sizeof(double),
                           cudaMemcpyHostToDevice, "h_k3 -> d_k3")) return 0;
  t1 = wall_seconds_cuda_local();
  fmmNearGpuUploadTime += (t1 - t0);

  if (useGpuDisjointBuild && disjointBuildCount > 0) {
    int blockSize = 256;
    int gridSize = (int)((gNear.nInteractions + blockSize - 1) / blockSize);
    cudaError_t err = cudaMemcpyToSymbol(c_nearKappa, &kappa, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      setNearfieldLastError("cudaMemcpyToSymbol c_nearKappa failed: %s",
                            cudaGetErrorString(err));
      return 0;
    }
    err = cudaMemcpyToSymbol(c_nearEpsilon, &epsilon, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      setNearfieldLastError("cudaMemcpyToSymbol c_nearEpsilon failed: %s",
                            cudaGetErrorString(err));
      return 0;
    }
    t0 = wall_seconds_cuda_local();
    nearfieldDisjointQ1BuildKernel<<<gridSize, blockSize>>>(
        gNear.nInteractions, gNear.d_panels, gNear.d_src, gNear.d_dst,
        gNear.d_k0, gNear.d_k1, gNear.d_k2, gNear.d_k3);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      setNearfieldLastError("nearfield disjoint build kernel launch failed: %s",
                            cudaGetErrorString(err));
      return 0;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      setNearfieldLastError("nearfield disjoint build kernel failed: %s",
                            cudaGetErrorString(err));
      return 0;
    }
    t1 = wall_seconds_cuda_local();
    fmmNearGpuCoeffTime += (t1 - t0);
  }
  if (sys->benchmarkMode > 0) {
    printf("GPU nearfield cache: panel-pairs=%lld mode=%d\n", totalInteractions, sys->gpuNearfieldMode);
    printf("GPU nearfield panelIA0 cases: disjoint=%lld one-common=%lld two-common=%lld two-common-rev=%lld self=%lld\n",
           gNear.caseDisjointCount, gNear.caseOneCommonCount, gNear.caseTwoCommonCount,
           gNear.caseTwoCommonRevCount, gNear.caseSelfCount);
    printf("GPU nearfield stage2 metadata: panels=%d disjoint-interactions=%lld\n",
           sys->nPnls, disjointBuildCount);
    if (useGpuDisjointBuild) {
      printf("GPU nearfield stage2 mode: disjoint-q1 builder enabled\n");
    }
  }
  return 1;
}

int buildDirectTables(const ssystem *sys) {
  long long nInteractions;
  size_t coeffBytes;
  size_t blockCoeffBytes;
  size_t vecBytes;
  size_t totalBytes;
  size_t hostCoeffBytes;
  size_t combinedBytes;
  size_t blockHostCoeffBytes;
  size_t blockTotalBytes;
  size_t blockCombinedBytes;
  size_t usableBytes = 0;
  size_t freeBytes = 0, totalGpuBytes = 0;
  int dstBlockCount;
  int i, j;
  double t0, t1;

  nInteractions = (long long)sys->nPnls * (long long)sys->nPnls;
  if (nInteractions <= 0) {
    return 0;
  }
  t0 = wall_seconds_cuda_local();

  coeffBytes = (size_t)nInteractions * sizeof(double);
  vecBytes = (size_t)(2 * sys->nPnls) * sizeof(double);
  hostCoeffBytes = 4 * coeffBytes;
  totalBytes = 4 * coeffBytes + 2 * vecBytes;
  combinedBytes = hostCoeffBytes + totalBytes;
  dstBlockCount = 1;
  if (cudaMemGetInfo(&freeBytes, &totalGpuBytes) == cudaSuccess) {
    size_t safetyBytes = freeBytes / 5;
    if (safetyBytes < (size_t)(512U * 1024U * 1024U)) {
      safetyBytes = (size_t)(512U * 1024U * 1024U);
    }
    usableBytes = (freeBytes > safetyBytes) ? (freeBytes - safetyBytes) : 0;
    if (usableBytes > 2 * vecBytes) {
      size_t coeffBudget = usableBytes - 2 * vecBytes;
      size_t bytesPerDstPanel = (size_t)sys->nPnls * 4U * sizeof(double);
      if (bytesPerDstPanel > 0) {
        size_t candidate = coeffBudget / bytesPerDstPanel;
        if (candidate > (size_t)sys->nPnls) {
          candidate = (size_t)sys->nPnls;
        }
        if (candidate > 0) {
          dstBlockCount = (int)candidate;
        }
      }
    }
  }
  blockCoeffBytes = (size_t)dstBlockCount * (size_t)sys->nPnls * sizeof(double);
  blockHostCoeffBytes = 4 * blockCoeffBytes;
  blockTotalBytes = 4 * blockCoeffBytes + 2 * vecBytes;
  blockCombinedBytes = blockHostCoeffBytes + blockTotalBytes;
  if (sys->benchmarkMode > 0) {
    printf("Direct GPU memory estimate: solver-host=%.3f GB host-coeff=%.3f GB device-total=%.3f GB combined=%.3f GB\n",
           (double)memcount / (1024.0 * 1024.0 * 1024.0),
           (double)hostCoeffBytes / (1024.0 * 1024.0 * 1024.0),
           (double)totalBytes / (1024.0 * 1024.0 * 1024.0),
           (double)combinedBytes / (1024.0 * 1024.0 * 1024.0));
    printf("Blocked direct GPU estimate: dst-block=%d host-coeff=%.3f GB device-total=%.3f GB combined=%.3f GB\n",
           dstBlockCount,
           (double)blockHostCoeffBytes / (1024.0 * 1024.0 * 1024.0),
           (double)blockTotalBytes / (1024.0 * 1024.0 * 1024.0),
           (double)blockCombinedBytes / (1024.0 * 1024.0 * 1024.0));
  }
  if (freeBytes > 0) {
    if (sys->benchmarkMode > 0)
      printf("Direct GPU device memory: free=%.3f GB total=%.3f GB\n",
             (double)freeBytes / (1024.0 * 1024.0 * 1024.0),
             (double)totalGpuBytes / (1024.0 * 1024.0 * 1024.0));
  }

  if (freeBytes > 0 && totalBytes <= freeBytes * 7 / 10) {
    if (!allocateDirectArrays(sys->nPnls, nInteractions)) {
      return 0;
    }

    kernel = kernelKER4;
    {
      double tc0 = wall_seconds_cuda_local();
      int nThreads = directBuildThreadCount(sys->nPnls);
      if (nThreads <= 1) {
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
      } else {
        std::vector<std::thread> workers;
        int t;

        workers.reserve((size_t)nThreads);
        for (t = 0; t < nThreads; t++) {
          int begin = (sys->nPnls * t) / nThreads;
          int end = (sys->nPnls * (t + 1)) / nThreads;
          workers.emplace_back([=]() {
            int localI, localJ;
            for (localI = begin; localI < end; localI++) {
              panel *pnlX = sys->panelByIdx[localI];
              long long base = (long long)localI * (long long)sys->nPnls;
              for (localJ = 0; localJ < sys->nPnls; localJ++) {
                panel *pnlY = sys->panelByIdx[localJ];
                double *KER = panelIA0(pnlX, pnlY);
                long long idx = base + (long long)localJ;

                gDirect.h_k0[idx] = KER[0];
                gDirect.h_k1[idx] = KER[1];
                gDirect.h_k2[idx] = KER[2];
                gDirect.h_k3[idx] = KER[3];
              }
            }
          });
        }
        for (t = 0; t < nThreads; t++) {
          workers[t].join();
        }
      }
      directGpuCoeffTime += (wall_seconds_cuda_local() - tc0);
    }

    t0 = wall_seconds_cuda_local();
    if (cudaMemcpy(gDirect.d_k0, gDirect.h_k0, coeffBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(gDirect.d_k1, gDirect.h_k1, coeffBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(gDirect.d_k2, gDirect.h_k2, coeffBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(gDirect.d_k3, gDirect.h_k3, coeffBytes, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    t1 = wall_seconds_cuda_local();
    directGpuStoreTime += (t1 - t0);

    gDirect.sys = sys;
    gDirect.nPnls = sys->nPnls;
    gDirect.nInteractions = nInteractions;
    gDirect.mode = 1;
    gDirect.blockDstCount = sys->nPnls;
    t1 = wall_seconds_cuda_local();
    directGpuBuildTime += (t1 - t0);
    if (sys->benchmarkMode > 0)
      printf("GPU direct cache: panel-pairs=%lld mode=dense\n", nInteractions);
    return 1;
  }

  if (freeBytes > 0 && blockTotalBytes > freeBytes * 7 / 10) {
    printf("Direct GPU cache unavailable: dense need %.3f GB, blocked need %.3f GB, free %.3f GB\n",
           (double)totalBytes / (1024.0 * 1024.0 * 1024.0),
           (double)blockTotalBytes / (1024.0 * 1024.0 * 1024.0),
           (double)freeBytes / (1024.0 * 1024.0 * 1024.0));
    return 0;
  }

  if (!allocateBlockedDirectArrays(sys->nPnls, dstBlockCount)) {
    return 0;
  }

  gDirect.sys = sys;
  gDirect.nPnls = sys->nPnls;
  gDirect.nInteractions = nInteractions;
  gDirect.mode = 2;
  gDirect.blockDstCount = dstBlockCount;
  t1 = wall_seconds_cuda_local();
  directGpuBuildTime += (t1 - t0);
  if (sys->benchmarkMode > 0)
    printf("GPU direct cache: panel-pairs=%lld mode=blocked dst-block=%d\n", nInteractions, dstBlockCount);
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

__device__ int samePoint3(const double a[3], const double b[3]) {
  return (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]);
}

__device__ int nearDisjointCase(const NearPanelGeom *dst, const NearPanelGeom *src) {
  int i, j;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      if (samePoint3(dst->vtx[i], src->vtx[j])) {
        return 0;
      }
    }
  }
  return 1;
}

__device__ void kernelKer4Device(
    const double r[3],
    const double nrmX[3],
    const double nrmY[3],
    double out[4]) {
  const double fourPiInv = 0.07957747154594767;
  double r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
  double rnorm = sqrt(r2);
  double expKa = exp(-c_nearKappa * rnorm);
  double ip0 = nrmX[0] * nrmY[0] + nrmX[1] * nrmY[1] + nrmX[2] * nrmY[2];
  double ipX = nrmX[0] * r[0] + nrmX[1] * r[1] + nrmX[2] * r[2];
  double ipY = nrmY[0] * r[0] + nrmY[1] * r[1] + nrmY[2] * r[2];
  double G0 = (1.0 / rnorm) * fourPiInv;
  double Gk = expKa * G0;
  double coef = (c_nearKappa * rnorm + 1.0) * expKa;
  double dG0dy = ipY * G0 / r2;
  double dGkdy = coef * dG0dy;
  double dG0dx = ipX * G0 / r2;
  double dGkdx = coef * dG0dx;
  double ddG0dxdy = (ip0 * G0 - dG0dy * ipX * 3.0) / r2;
  double ddGkdxdy = coef * ddG0dxdy - c_nearKappa * c_nearKappa * expKa * ipX * dG0dy;

  out[0] = G0 - Gk;
  out[1] = c_nearEpsilon * dGkdy - dG0dy;
  out[2] = dGkdx / c_nearEpsilon - dG0dx;
  out[3] = ddGkdxdy - ddG0dxdy;
}

__global__ void nearfieldDisjointQ1BuildKernel(
    long long nInteractions,
    const NearPanelGeom *panels,
    const int *src,
    const int *dst,
    double *k0,
    double *k1,
    double *k2,
    double *k3) {
  long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (tid >= nInteractions) {
    return;
  }

  int srcIdx = src[tid];
  int dstIdx = dst[tid];
  const NearPanelGeom *pnlX = &panels[dstIdx];
  const NearPanelGeom *pnlY = &panels[srcIdx];
  double r0[3], r1[3], r[3], out[4];
  int k;

  if (!nearDisjointCase(pnlX, pnlY)) {
    return;
  }

  for (k = 0; k < 3; k++) {
    r0[k] = pnlX->vtx[0][k] - pnlY->vtx[0][k];
    r1[k] = r0[k] + 0.5 * (pnlX->a2[k] + 0.5 * pnlX->a0[k]);
    r[k] = r1[k] - 0.5 * (pnlY->a2[k] + 0.5 * pnlY->a0[k]);
  }

  kernelKer4Device(r, pnlX->normal, pnlY->normal, out);
  k0[tid] = pnlX->area * pnlY->area * out[0];
  k1[tid] = pnlX->area * pnlY->area * out[1];
  k2[tid] = pnlX->area * pnlY->area * out[2];
  k3[tid] = pnlX->area * pnlY->area * out[3];
}

/*
 * Shared body for the two disjoint-pair kernels: one reads the panel indices
 * from uploaded arrays, the other derives them from the leaf-pair table. Both
 * call this so the arithmetic cannot drift apart.
 */
__device__ __forceinline__ void nearDisjointQ1Apply(
    int nPnls,
    int s,
    int d,
    const NearPanelGeom *panels,
    double alpha,
    const double *sgm,
    double *pot) {
  const NearPanelGeom *pnlX = &panels[d];
  const NearPanelGeom *pnlY = &panels[s];
  double r0[3], r1[3], r[3], out[4];
  double scale;
  int k;

  if (!nearDisjointCase(pnlX, pnlY)) return;
  for (k = 0; k < 3; k++) {
    r0[k] = pnlX->vtx[0][k] - pnlY->vtx[0][k];
    r1[k] = r0[k] + 0.5 * (pnlX->a2[k] + 0.5 * pnlX->a0[k]);
    r[k] = r1[k] - 0.5 * (pnlY->a2[k] + 0.5 * pnlY->a0[k]);
  }
  kernelKer4Device(r, pnlX->normal, pnlY->normal, out);
  scale = pnlX->area * pnlY->area;
#if __CUDA_ARCH__ >= 600
  atomicAdd(&pot[d], alpha * scale * (out[0] * sgm[s + nPnls] + out[1] * sgm[s]));
  atomicAdd(&pot[d + nPnls], alpha * scale * (out[2] * sgm[s + nPnls] + out[3] * sgm[s]));
#else
  {
    double addPot = alpha * scale * (out[0] * sgm[s + nPnls] + out[1] * sgm[s]);
    double addDpdn = alpha * scale * (out[2] * sgm[s + nPnls] + out[3] * sgm[s]);
    unsigned long long int *addr1 = (unsigned long long int *)&pot[d];
    unsigned long long int *addr2 = (unsigned long long int *)&pot[d + nPnls];
    unsigned long long int old1 = *addr1, assumed1;
    unsigned long long int old2 = *addr2, assumed2;
    do {
      assumed1 = old1;
      old1 = atomicCAS(addr1, assumed1,
                       __double_as_longlong(addPot + __longlong_as_double(assumed1)));
    } while (assumed1 != old1);
    do {
      assumed2 = old2;
      old2 = atomicCAS(addr2, assumed2,
                       __double_as_longlong(addDpdn + __longlong_as_double(assumed2)));
    } while (assumed2 != old2);
  }
#endif
}

__global__ void nearfieldDisjointQ1ApplyKernel(
    int nPnls,
    long long nInteractions,
    const NearPanelGeom *panels,
    const int *src,
    const int *dst,
    double alpha,
    const double *sgm,
    double *pot) {
  long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (tid >= nInteractions) return;
  nearDisjointQ1Apply(nPnls, src[tid], dst[tid], panels, alpha, sgm, pot);
}

/*
 * Same work, but the panel indices are computed instead of transferred.
 *
 * The host used to materialise src/dst for every interaction and upload them
 * on every matvec -- 3.8 GB per matvec on a 420k-panel mesh, and 62 GB on the
 * virus, which is more than the card holds. They are a pure function of the
 * leaf-pair table, so each thread recovers its own pair from the interaction
 * offsets and derives the two panel indices directly.
 *
 * Chunk c covers global interactions [c*chunkCapacity, ...), because the host
 * loop simply counts interactions and flushes at chunkCapacity.
 */
__global__ void nearfieldDisjointQ1GeneratedKernel(
    int nPnls,
    long long chunkStart,
    long long count,
    int nPairs,
    const long long *pairOffset,
    const int *pairSrcStart,
    const int *pairSrcCount,
    const int *pairDstStart,
    const NearPanelGeom *panels,
    double alpha,
    const double *sgm,
    double *pot) {
  long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  long long g;
  long long local;
  int lo = 0;
  int hi = nPairs;
  int p, sc, i, j;

  if (tid >= count) return;
  g = chunkStart + tid;

  /* Largest p with pairOffset[p] <= g; pairOffset holds nPairs+1 entries. */
  while (hi - lo > 1) {
    int mid = (lo + hi) >> 1;
    if (pairOffset[mid] <= g) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  p = lo;

  local = g - pairOffset[p];
  sc = pairSrcCount[p];
  i = (int)(local / (long long)sc);
  j = (int)(local - (long long)i * (long long)sc);

  nearDisjointQ1Apply(nPnls, pairSrcStart[p] + j, pairDstStart[p] + i,
                      panels, alpha, sgm, pot);
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

__global__ void directApplyBlockKernel(
    int nPnls,
    int dstStart,
    int dstCount,
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
  int localD = blockIdx.x;
  int d = dstStart + localD;
  int tid = threadIdx.x;
  int s;
  long long base;
  double sumPot = 0.0;
  double sumDpdn = 0.0;
  int stride;

  if (localD >= dstCount || d >= nPnls) {
    return;
  }

  base = (long long)localD * (long long)nPnls;
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

int applyNearfieldStreamingChunk(const ssystem *sys, long long count, double alpha,
                                 long long chunkIdx, long long *specialTotal) {
  const int blockSize = 256;
  int gridSize = (int)((count + blockSize - 1) / blockSize);
  long long specialCount = 0;
  size_t indexBytes = (size_t)count * sizeof(int);
  cudaError_t err;
  long long idx;
  /* Replay a previously classified chunk instead of redoing the work. */
  int useCache = (gNear.specialCacheEnabled && gNear.specialCacheValid &&
                  chunkIdx < gNear.specialCacheChunks);
  /* With the cache live the host stops filling h_src/h_dst, so the disjoint
   * kernel has to derive the indices itself. */
  int useGenerated = (useCache && gNear.d_streamPairOffset != NULL);
  const int *upSrc;
  const int *upDst;
  const double *upK0;
  const double *upK1;
  const double *upK2;
  const double *upK3;

  if (useGenerated) {
    nearfieldDisjointQ1GeneratedKernel<<<gridSize, blockSize>>>(
        gNear.nPnls, chunkIdx * gNear.chunkCapacity, count,
        gNear.streamPairCount, gNear.d_streamPairOffset,
        gNear.d_streamPairSrcStart, gNear.d_streamPairSrcCount,
        gNear.d_streamPairDstStart,
        gNear.d_panels, alpha, gNear.d_sgm, gNear.d_pot);
  } else {
    if (!cudaMemcpyNearfield(gNear.d_src, gNear.h_src, indexBytes,
                             cudaMemcpyHostToDevice, "stream src") ||
        !cudaMemcpyNearfield(gNear.d_dst, gNear.h_dst, indexBytes,
                             cudaMemcpyHostToDevice, "stream dst")) return 0;

    nearfieldDisjointQ1ApplyKernel<<<gridSize, blockSize>>>(
        gNear.nPnls, count, gNear.d_panels, gNear.d_src, gNear.d_dst,
        alpha, gNear.d_sgm, gNear.d_pot);
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    setNearfieldLastError("streaming disjoint kernel launch failed: %s",
                          cudaGetErrorString(err));
    return 0;
  }

  /*
   * Classify every interaction in the chunk and compact the touching/self
   * ("special") ones to the front of the chunk buffers.
   *
   * This is the dominant cost of the streaming path: nrCommonVtx() runs once
   * per interaction, on every matvec. On a 420k-panel mesh that is 476M calls
   * per matvec, and it used to run on one thread inside a timer labelled
   * "kernel", which is why the GPU looked slow when the work was actually on
   * the host.
   *
   * Parallelising needs care because the compaction is in-place: the loop
   * reads h_src[idx] and writes h_src[specialCount] with specialCount <= idx.
   * That is safe in a serial pass but not across threads, where one thread's
   * writes can land in a range another thread has not read yet. So the work is
   * split: each thread first classifies its own range into thread-local
   * buffers (reads only), then, after every thread has finished reading, the
   * results are copied back at prefix-summed offsets. Output order matches the
   * serial version exactly, so the coefficients uploaded to the GPU are
   * identical.
   *
   * On matvecs after the first this is skipped entirely: the answer depends
   * only on the mesh, so it is replayed from the cache populated below.
   */
  if (useCache) {
    long long lo = (*gNear.specialOffset)[(size_t)chunkIdx];
    specialCount = (*gNear.specialOffset)[(size_t)chunkIdx + 1U] - lo;
    upSrc = specialCount > 0 ? &(*gNear.cacheSrc)[(size_t)lo] : NULL;
    upDst = specialCount > 0 ? &(*gNear.cacheDst)[(size_t)lo] : NULL;
    upK0 = specialCount > 0 ? &(*gNear.cacheK0)[(size_t)lo] : NULL;
    upK1 = specialCount > 0 ? &(*gNear.cacheK1)[(size_t)lo] : NULL;
    upK2 = specialCount > 0 ? &(*gNear.cacheK2)[(size_t)lo] : NULL;
    upK3 = specialCount > 0 ? &(*gNear.cacheK3)[(size_t)lo] : NULL;
  } else {
    int nThreads = nearfieldBuildThreadCount((int)((count > (long long)INT_MAX)
                                                   ? (long long)INT_MAX : count));
    if (nThreads <= 1) {
      for (idx = 0; idx < count; idx++) {
        int srcIdx = gNear.h_src[idx];
        int dstIdx = gNear.h_dst[idx];
        panel *pnlX = sys->panelByIdx[dstIdx];
        panel *pnlY = sys->panelByIdx[srcIdx];
        int idxX[3], idxY[3];
        int nVtx = nrCommonVtx(pnlX, pnlY, idxX, idxY);
        if (nVtx != 0) {
          double *KER = panelIA0(pnlX, pnlY);
          gNear.h_src[specialCount] = srcIdx;
          gNear.h_dst[specialCount] = dstIdx;
          gNear.h_k0[specialCount] = KER[0];
          gNear.h_k1[specialCount] = KER[1];
          gNear.h_k2[specialCount] = KER[2];
          gNear.h_k3[specialCount] = KER[3];
          specialCount++;
        }
      }
    } else {
      std::vector<std::vector<int> > tSrc((size_t)nThreads), tDst((size_t)nThreads);
      std::vector<std::vector<double> > tK0((size_t)nThreads), tK1((size_t)nThreads),
                                        tK2((size_t)nThreads), tK3((size_t)nThreads);
      std::vector<std::thread> workers;
      std::vector<long long> offset((size_t)nThreads + 1U, 0);
      int t;

      workers.reserve((size_t)nThreads);
      for (t = 0; t < nThreads; t++) {
        workers.push_back(std::thread([&, t]() {
          long long begin = (count * (long long)t) / (long long)nThreads;
          long long end = (count * (long long)(t + 1)) / (long long)nThreads;
          std::vector<int> &vs = tSrc[(size_t)t];
          std::vector<int> &vd = tDst[(size_t)t];
          std::vector<double> &v0 = tK0[(size_t)t];
          std::vector<double> &v1 = tK1[(size_t)t];
          std::vector<double> &v2 = tK2[(size_t)t];
          std::vector<double> &v3 = tK3[(size_t)t];
          long long i;
          /* Touching/self pairs are a small minority of the chunk (~1% on a
           * compact protein, ~7% on the virus mesh). Reserving an eighth up
           * front keeps the common case free of reallocation without holding
           * much memory; the vectors still grow if a chunk is unusually dense. */
          size_t guess = (size_t)((end - begin) / 8 + 1);
          vs.reserve(guess); vd.reserve(guess);
          v0.reserve(guess); v1.reserve(guess);
          v2.reserve(guess); v3.reserve(guess);

          for (i = begin; i < end; i++) {
            int srcIdx = gNear.h_src[i];
            int dstIdx = gNear.h_dst[i];
            panel *pnlX = sys->panelByIdx[dstIdx];
            panel *pnlY = sys->panelByIdx[srcIdx];
            int idxX[3], idxY[3];
            int nVtx = nrCommonVtx(pnlX, pnlY, idxX, idxY);
            if (nVtx != 0) {
              double *KER = panelIA0(pnlX, pnlY);
              vs.push_back(srcIdx);
              vd.push_back(dstIdx);
              v0.push_back(KER[0]);
              v1.push_back(KER[1]);
              v2.push_back(KER[2]);
              v3.push_back(KER[3]);
            }
          }
        }));
      }
      for (t = 0; t < nThreads; t++) {
        workers[(size_t)t].join();
      }

      for (t = 0; t < nThreads; t++) {
        offset[(size_t)t + 1U] = offset[(size_t)t] + (long long)tSrc[(size_t)t].size();
      }
      specialCount = offset[(size_t)nThreads];

      /* Every thread has finished reading, so writing back in place is safe. */
      workers.clear();
      for (t = 0; t < nThreads; t++) {
        workers.push_back(std::thread([&, t]() {
          size_t n = tSrc[(size_t)t].size();
          long long at = offset[(size_t)t];
          if (n == 0U) {
            return;
          }
          memcpy(&gNear.h_src[at], &tSrc[(size_t)t][0], n * sizeof(int));
          memcpy(&gNear.h_dst[at], &tDst[(size_t)t][0], n * sizeof(int));
          memcpy(&gNear.h_k0[at], &tK0[(size_t)t][0], n * sizeof(double));
          memcpy(&gNear.h_k1[at], &tK1[(size_t)t][0], n * sizeof(double));
          memcpy(&gNear.h_k2[at], &tK2[(size_t)t][0], n * sizeof(double));
          memcpy(&gNear.h_k3[at], &tK3[(size_t)t][0], n * sizeof(double));
        }));
      }
      for (t = 0; t < nThreads; t++) {
        workers[(size_t)t].join();
      }
    }

    /*
     * Record this chunk for the remaining matvecs. If the cache would exceed
     * its budget, drop it and keep reclassifying rather than risk pushing the
     * host into swap on a mesh where the solve itself is already large.
     */
    if (gNear.specialCacheEnabled && !gNear.specialCacheValid) {
      size_t projected = (size_t)(gNear.cacheSrc->size() + (size_t)specialCount) *
                         NEAR_SPECIAL_CACHE_BYTES_PER_PAIR;
      if (projected > nearfieldSpecialCacheBudget()) {
        if (sys->benchmarkMode > 0) {
          printf("GPU nearfield special cache disabled: %.3f GiB exceeds budget %.3f GiB\n",
                 bytesToGiB(projected), bytesToGiB(nearfieldSpecialCacheBudget()));
        }
        gNear.specialCacheEnabled = 0;
        freeNearfieldSpecialCache();
      } else {
        gNear.cacheSrc->insert(gNear.cacheSrc->end(), gNear.h_src, gNear.h_src + specialCount);
        gNear.cacheDst->insert(gNear.cacheDst->end(), gNear.h_dst, gNear.h_dst + specialCount);
        gNear.cacheK0->insert(gNear.cacheK0->end(), gNear.h_k0, gNear.h_k0 + specialCount);
        gNear.cacheK1->insert(gNear.cacheK1->end(), gNear.h_k1, gNear.h_k1 + specialCount);
        gNear.cacheK2->insert(gNear.cacheK2->end(), gNear.h_k2, gNear.h_k2 + specialCount);
        gNear.cacheK3->insert(gNear.cacheK3->end(), gNear.h_k3, gNear.h_k3 + specialCount);
        gNear.specialOffset->push_back((long long)gNear.cacheSrc->size());
      }
    }
    upSrc = gNear.h_src;
    upDst = gNear.h_dst;
    upK0 = gNear.h_k0;
    upK1 = gNear.h_k1;
    upK2 = gNear.h_k2;
    upK3 = gNear.h_k3;
  }

  if (specialCount > 0) {
    size_t specialIndexBytes = (size_t)specialCount * sizeof(int);
    size_t specialCoeffBytes = (size_t)specialCount * sizeof(double);
    int specialGrid = (int)((specialCount + blockSize - 1) / blockSize);
    if (!cudaMemcpyNearfield(gNear.d_src, upSrc, specialIndexBytes,
                             cudaMemcpyHostToDevice, "stream special src") ||
        !cudaMemcpyNearfield(gNear.d_dst, upDst, specialIndexBytes,
                             cudaMemcpyHostToDevice, "stream special dst") ||
        !cudaMemcpyNearfield(gNear.d_k0, upK0, specialCoeffBytes,
                             cudaMemcpyHostToDevice, "stream special k0") ||
        !cudaMemcpyNearfield(gNear.d_k1, upK1, specialCoeffBytes,
                             cudaMemcpyHostToDevice, "stream special k1") ||
        !cudaMemcpyNearfield(gNear.d_k2, upK2, specialCoeffBytes,
                             cudaMemcpyHostToDevice, "stream special k2") ||
        !cudaMemcpyNearfield(gNear.d_k3, upK3, specialCoeffBytes,
                             cudaMemcpyHostToDevice, "stream special k3")) return 0;
    nearfieldApplyKernel<<<specialGrid, blockSize>>>(
        gNear.nPnls, specialCount, gNear.d_src, gNear.d_dst,
        gNear.d_k0, gNear.d_k1, gNear.d_k2, gNear.d_k3,
        alpha, gNear.d_sgm, gNear.d_pot);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      setNearfieldLastError("streaming special-panel kernel launch failed: %s",
                            cudaGetErrorString(err));
      return 0;
    }
  }
  *specialTotal += specialCount;
  return 1;
}

int applyNearfieldStreaming(const ssystem *sys, double alpha) {
  long long count = 0;
  long long chunks = 0;
  long long specialTotal = 0;
  int pairIdx;
  cudaError_t err;

  kernel = kernelKER4;
  err = cudaMemcpyToSymbol(c_nearKappa, &kappa, sizeof(double), 0,
                           cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    err = cudaMemcpyToSymbol(c_nearEpsilon, &epsilon, sizeof(double), 0,
                             cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) {
    setNearfieldLastError("streaming kernel constant upload failed: %s",
                          cudaGetErrorString(err));
    return 0;
  }

  /*
   * Once the special cache is populated, nothing downstream reads the host
   * index arrays: the specials come from the cache and the disjoint kernel
   * derives its own indices. Skip building them altogether and just walk the
   * chunk boundaries, which are fixed multiples of chunkCapacity. This removes
   * both the index writes and their upload -- 3.8 GB per matvec on a
   * 420k-panel mesh, 62 GB on the virus.
   */
  if (gNear.specialCacheValid && gNear.d_streamPairOffset != NULL) {
    long long remaining = gNear.nInteractions;
    while (remaining > 0) {
      long long thisChunk = (remaining < gNear.chunkCapacity) ? remaining
                                                              : gNear.chunkCapacity;
      if (!applyNearfieldStreamingChunk(sys, thisChunk, alpha, chunks, &specialTotal)) return 0;
      remaining -= thisChunk;
      chunks++;
    }
  } else {
    for (pairIdx = 0; pairIdx < sys->nNearPairsFlat; pairIdx++) {
      int srcLeaf = sys->nearPairSrc[pairIdx];
      int dstLeaf = sys->nearPairDst[pairIdx];
      int srcStart = sys->leafPanelStart[srcLeaf];
      int srcCount = sys->leafPanelCount[srcLeaf];
      int dstStart = sys->leafPanelStart[dstLeaf];
      int dstCount = sys->leafPanelCount[dstLeaf];
      int i, j;
      for (i = 0; i < dstCount; i++) {
        for (j = 0; j < srcCount; j++) {
          gNear.h_src[count] = srcStart + j;
          gNear.h_dst[count] = dstStart + i;
          count++;
          if (count == gNear.chunkCapacity) {
            if (!applyNearfieldStreamingChunk(sys, count, alpha, chunks, &specialTotal)) return 0;
            count = 0;
            chunks++;
          }
        }
      }
    }
    if (count > 0) {
      if (!applyNearfieldStreamingChunk(sys, count, alpha, chunks, &specialTotal)) return 0;
      chunks++;
    }
  }
  /*
   * The first matvec has now classified every chunk, so later ones can replay
   * it. Chunk boundaries depend only on geometry and chunkCapacity, so the
   * count recorded here stays valid for the rest of the solve.
   */
  if (gNear.specialCacheEnabled && !gNear.specialCacheValid) {
    gNear.specialCacheChunks = chunks;
    gNear.specialCacheValid = 1;
    if (sys->benchmarkMode > 0) {
      printf("GPU nearfield special cache: pairs=%lld host=%.3f GiB chunks=%lld\n",
             (long long)gNear.cacheSrc->size(),
             bytesToGiB((size_t)gNear.cacheSrc->size() * NEAR_SPECIAL_CACHE_BYTES_PER_PAIR),
             chunks);
    }
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    setNearfieldLastError("streaming nearfield execution failed: %s",
                          cudaGetErrorString(err));
    return 0;
  }
  if (sys->benchmarkMode > 0) {
    printf("GPU nearfield streaming apply: chunks=%lld interactions=%lld special=%lld\n",
           chunks, gNear.nInteractions, specialTotal);
  }
  return 1;
}

int flushM2LStreamingChunk(int pairCount, int groupCount, int coeffCount) {
  cudaError_t err;
  const int blockSize = 128;
  if (pairCount <= 0 || groupCount <= 0 || coeffCount <= 0) return 1;

#define M2L_CHUNK_UPLOAD(dst, src, count, type) \
  do { \
    err = cudaMemcpy((dst), (src), (size_t)(count) * sizeof(type), cudaMemcpyHostToDevice); \
    if (err != cudaSuccess) return 0; \
  } while (0)
  M2L_CHUNK_UPLOAD(gM2L.d_pairSrc, gM2L.h_pairSrc, pairCount, int);
  M2L_CHUNK_UPLOAD(gM2L.d_pairCoeffOffset, gM2L.h_pairCoeffOffset, pairCount, int);
  M2L_CHUNK_UPLOAD(gM2L.d_groupStart, gM2L.h_groupStart, groupCount, int);
  M2L_CHUNK_UPLOAD(gM2L.d_groupCount, gM2L.h_groupCount, groupCount, int);
  M2L_CHUNK_UPLOAD(gM2L.d_groupDst, gM2L.h_groupDst, groupCount, int);
  M2L_CHUNK_UPLOAD(gM2L.d_groupOrder, gM2L.h_groupOrder, groupCount, int);
  M2L_CHUNK_UPLOAD(gM2L.d_g0, gM2L.h_g0, coeffCount, double);
  M2L_CHUNK_UPLOAD(gM2L.d_gk, gM2L.h_gk, coeffCount, double);
#undef M2L_CHUNK_UPLOAD

  m2lGroupedKernel<<<groupCount, blockSize>>>(
      groupCount, gM2L.maxIdxDim,
      gM2L.d_groupStart, gM2L.d_groupCount, gM2L.d_groupDst, gM2L.d_groupOrder,
      gM2L.d_pairSrc, gM2L.d_pairCoeffOffset,
      gM2L.d_cubeCoeffOffset,
      gM2L.d_idxI1, gM2L.d_idxI2, gM2L.d_idxI3, gM2L.d_idx3Flat, gM2L.d_sgn3,
      gM2L.d_g0, gM2L.d_gk, epsilon,
      gM2L.d_momPot, gM2L.d_momDpdn,
      gM2L.d_lec1, gM2L.d_lec2, gM2L.d_lec3, gM2L.d_lec4);
  return cudaGetLastError() == cudaSuccess;
}

/*
 * A chunk's worth of destination groups, with the output slots each will fill.
 * Recording these first turns the coefficient build into an independent
 * scatter, which is what makes it threadable.
 */
struct M2LChunkGroup {
  int globalStart;
  int count;
  int order;
  int nMom;
  int pairBase;     /* first slot in h_pairSrc / h_pairCoeffOffset */
  int coeffBase;    /* first slot in h_g0 / h_gk */
};

static int m2lBuildThreadCount(int nTasks) {
  const char *env = getenv("FABIPB_SETUP_THREADS");
  long hc;
  int threads;

  (void)hc;
  if (env != NULL && env[0] != '\0') threads = atoi(env);
  else { unsigned int c = std::thread::hardware_concurrency(); threads = (c > 0U) ? (int)c : 1; }
  if (threads < 1) threads = 1;
  if (threads > nTasks) threads = nTasks;
  if (threads > 64) threads = 64;
  return threads;
}

/*
 * Fills the coefficient staging buffers for one chunk.
 *
 * setupDerivs() runs once per M2L pair on every matvec -- 546M pairs times 38
 * matvecs on the virus, 20.8 billion calls -- rebuilding coefficients that
 * depend only on the (fixed) cube centres. It ran on one thread, and at
 * sdens=2 that made M2L 950 s, 68% of the matvec.
 *
 * Groups average only about a hundred pairs, so threading inside a group would
 * be all fork/join. Threading across the chunk's groups instead gives each
 * worker thousands of pairs. Every group writes to slots reserved for it in the
 * bookkeeping pass, so the threads never overlap and the buffers come out
 * byte-identical to the serial order.
 */
static int fillM2LChunkGroups(const ssystem *sys,
                              const std::vector<M2LChunkGroup> &groups) {
  int nGroups = (int)groups.size();
  int nThreads;

  if (nGroups <= 0) return 1;
  nThreads = m2lBuildThreadCount(nGroups);

  if (nThreads <= 1) {
    RhsTreeWorkspace ws;
    int g;
    initRhsTreeWorkspace((ssystem *)sys, &ws);
    for (g = 0; g < nGroups; g++) {
      const M2LChunkGroup &grp = groups[(size_t)g];
      int j;
      for (j = 0; j < grp.count; j++) {
        int globalPair = grp.globalStart + j;
        cube *src = sys->fmmCubeByIdx[sys->m2lPairSrc[globalPair]];
        cube *dst = sys->fmmCubeByIdx[sys->m2lPairDst[globalPair]];
        double r[3];
        int k, at = grp.coeffBase + j * grp.nMom;
        for (k = 0; k < 3; k++) r[k] = dst->x[k] - src->x[k];
        setupDerivsWorkspace((ssystem *)sys, &ws, grp.order, r);
        gM2L.h_pairSrc[grp.pairBase + j] = sys->m2lPairSrc[globalPair];
        gM2L.h_pairCoeffOffset[grp.pairBase + j] = at;
        memcpy(&gM2L.h_g0[at], ws.dg0[0], (size_t)grp.nMom * sizeof(double));
        memcpy(&gM2L.h_gk[at], ws.dgk[0], (size_t)grp.nMom * sizeof(double));
      }
    }
    freeRhsTreeWorkspace(&ws);
    return 1;
  }

  {
    std::vector<std::thread> workers;
    std::vector<int> ok((size_t)nThreads, 1);
    int t;
    workers.reserve((size_t)nThreads);
    for (t = 0; t < nThreads; t++) {
      workers.push_back(std::thread([&, t]() {
        RhsTreeWorkspace ws;
        int begin = (int)(((long long)nGroups * t) / nThreads);
        int end = (int)(((long long)nGroups * (t + 1)) / nThreads);
        int g;
        initRhsTreeWorkspace((ssystem *)sys, &ws);
        for (g = begin; g < end; g++) {
          const M2LChunkGroup &grp = groups[(size_t)g];
          int j;
          for (j = 0; j < grp.count; j++) {
            int globalPair = grp.globalStart + j;
            cube *src = sys->fmmCubeByIdx[sys->m2lPairSrc[globalPair]];
            cube *dst = sys->fmmCubeByIdx[sys->m2lPairDst[globalPair]];
            double r[3];
            int k, at = grp.coeffBase + j * grp.nMom;
            for (k = 0; k < 3; k++) r[k] = dst->x[k] - src->x[k];
            setupDerivsWorkspace((ssystem *)sys, &ws, grp.order, r);
            gM2L.h_pairSrc[grp.pairBase + j] = sys->m2lPairSrc[globalPair];
            gM2L.h_pairCoeffOffset[grp.pairBase + j] = at;
            memcpy(&gM2L.h_g0[at], ws.dg0[0], (size_t)grp.nMom * sizeof(double));
            memcpy(&gM2L.h_gk[at], ws.dgk[0], (size_t)grp.nMom * sizeof(double));
          }
        }
        freeRhsTreeWorkspace(&ws);
      }));
    }
    for (t = 0; t < nThreads; t++) workers[(size_t)t].join();
  }
  return 1;
}

int applyM2LStreamingChunks(const ssystem *sys) {
  int groupIdx;
  int pairCount = 0;
  int groupCount = 0;
  int coeffCount = 0;
  int chunks = 0;
  std::vector<M2LChunkGroup> chunkGroups;

  for (groupIdx = 0; groupIdx < sys->nM2LDstGroups; groupIdx++) {
    int globalStart = sys->m2lDstGroupStart[groupIdx];
    int count = sys->m2lDstGroupCount[groupIdx];
    int order = sys->m2lPairOrder[globalStart];
    int nMom = sys->nMom[order];
    long long neededCoeff = (long long)count * (long long)nMom;
    M2LChunkGroup grp;

    if (count > gM2L.streamPairCapacity || neededCoeff > gM2L.streamCoeffCapacity) {
      printf("GPU M2L streaming group exceeds chunk capacity: group=%d pairs=%d coeff=%lld\n",
             groupIdx, count, neededCoeff);
      return 0;
    }
    if (groupCount == gM2L.streamGroupCapacity ||
        pairCount + count > gM2L.streamPairCapacity ||
        (long long)coeffCount + neededCoeff > gM2L.streamCoeffCapacity) {
      if (!fillM2LChunkGroups(sys, chunkGroups)) return 0;
      if (!flushM2LStreamingChunk(pairCount, groupCount, coeffCount)) return 0;
      chunkGroups.clear();
      pairCount = 0;
      groupCount = 0;
      coeffCount = 0;
      chunks++;
    }

    gM2L.h_groupStart[groupCount] = pairCount;
    gM2L.h_groupCount[groupCount] = count;
    gM2L.h_groupDst[groupCount] = sys->m2lPairDst[globalStart];
    gM2L.h_groupOrder[groupCount] = order;

    grp.globalStart = globalStart;
    grp.count = count;
    grp.order = order;
    grp.nMom = nMom;
    grp.pairBase = pairCount;
    grp.coeffBase = coeffCount;
    chunkGroups.push_back(grp);

    pairCount += count;
    coeffCount += (int)neededCoeff;
    groupCount++;
  }
  if (groupCount > 0) {
    if (!fillM2LChunkGroups(sys, chunkGroups)) return 0;
    if (!flushM2LStreamingChunk(pairCount, groupCount, coeffCount)) return 0;
    chunks++;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) return 0;
  if (sys->benchmarkMode > 0) {
    printf("GPU M2L streaming apply: chunks=%d pairs=%d coeff=%lld\n",
           chunks, gM2L.nPairs, gM2L.totalPairCoeff);
  }
  return 1;
}

/*
 * SM count of the current device, queried once.
 *
 * The charge-tree kernels size their scratch from this: their throughput
 * optimum is a concurrent-thread count, not a byte count (see
 * chargeTreeScratchThreads), and thread counts only mean anything relative to
 * the machine's width. Falls back to the measurement device (RTX 6000 Ada) if
 * the query fails, which only costs a mistuned default, never correctness.
 */
int deviceSmCount() {
  static int smCount = 0;
  if (smCount == 0) {
    cudaDeviceProp prop;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
        prop.multiProcessorCount > 0) {
      smCount = prop.multiProcessorCount;
    } else {
      smCount = 142;
    }
  }
  return smCount;
}

/*
 * Concurrent-thread count for a charge-tree kernel's derivative scratch.
 *
 * Both kernels give every thread a private slab of slabDoubles and then take
 * panels in a grid-stride loop, so this is a pure occupancy/cache trade: too
 * few threads and the machine idles, too many and the slabs stop fitting the
 * cache. It used to be set by a fixed 256 MiB byte budget, which is wrong,
 * because slabDoubles grows with derivMax and so the same byte count buys a
 * different number of threads on every problem. Measured on 7A6A (420,576
 * panels, RTX 6000 Ada, 142 SMs) by sweeping thread count at three slab sizes:
 *
 *   kernel  derivMax  slab(doubles)  best threads  best bytes
 *   energy  5         251            27,392        55 MB
 *   energy  6         420            20,096        67 MB
 *   energy  9         1430           20,480        234 MB
 *   rhs     9         715            12k-16k       70-94 MB
 *
 * The optimum moves by 1.4x in threads across a 5.7x change in slab size, and
 * by 4.3x in bytes -- so threads is the invariant to hold, scaled by SM count
 * so it transfers to other cards. perSm is the tuned constant: the energy
 * kernel tolerates more concurrency than the RHS kernel, which showed a sharp
 * step up in time past ~16k threads. Both curves are flat near the optimum,
 * so these sit mid-plateau rather than on a measured peak.
 *
 * FABIPB_*_GPU_SCRATCH_MIB still overrides with an explicit byte budget.
 */
long long chargeTreeScratchThreads(const char *mibEnv, int perSm,
                                   long long slabDoubles, long long nPanels,
                                   int blockSize) {
  long long stride;
  const char *env = getenv(mibEnv);
  double budgetMiB = -1.0;

  if (env != NULL && env[0] != '\0') {
    char *end = NULL;
    double v = strtod(env, &end);
    if (end != env && *end == '\0' && v > 0.0) budgetMiB = v;
  }
  if (budgetMiB > 0.0) {
    stride = (long long)((budgetMiB * 1024.0 * 1024.0) /
                         ((double)slabDoubles * sizeof(double)));
  } else {
    stride = (long long)deviceSmCount() * perSm;
  }
  if (stride < blockSize) stride = blockSize;
  if (stride > nPanels) stride = nPanels;
  return ((stride + blockSize - 1) / blockSize) * blockSize;
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

const char *gpuNearfieldLastError(void) {
  return gNearfieldLastError[0] ? gNearfieldLastError : "unknown nearfield GPU failure";
}

int gpuNearfieldApply(ssystem *sys, double alpha, const double *sgm, double *pot) {
  cudaError_t err;
  size_t vecBytes;
  int blockSize;
  int gridSize;
  double t0, t1;
  static int printedMode = -1;

  if (sys == NULL || sgm == NULL || pot == NULL) {
    setNearfieldLastError("invalid input pointer");
    return 0;
  }
  setNearfieldLastError("not attempted");

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
  if (err != cudaSuccess) {
    setNearfieldLastError("cudaMemcpy sgm host-to-device failed for %.3f GiB: %s",
                          bytesToGiB(vecBytes), cudaGetErrorString(err));
    return 0;
  }
  err = cudaMemcpy(gNear.d_pot, pot, vecBytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    setNearfieldLastError("cudaMemcpy pot host-to-device failed for %.3f GiB: %s",
                          bytesToGiB(vecBytes), cudaGetErrorString(err));
    return 0;
  }
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
  if (gNear.streaming) {
    if (!applyNearfieldStreaming(sys, alpha)) return 0;
  } else if (sys->gpuNearfieldMode == 1) {
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
  if (!gNear.streaming) {
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      setNearfieldLastError("nearfield apply kernel launch failed: %s",
                            cudaGetErrorString(err));
      return 0;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      setNearfieldLastError("nearfield apply kernel failed: %s",
                            cudaGetErrorString(err));
      return 0;
    }
  }
  t1 = wall_seconds_cuda_local();
  fmmNearGpuKernelTime += (t1 - t0);

  t0 = wall_seconds_cuda_local();
  err = cudaMemcpy(pot, gNear.d_pot, vecBytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    setNearfieldLastError("cudaMemcpy pot device-to-host failed for %.3f GiB: %s",
                          bytesToGiB(vecBytes), cudaGetErrorString(err));
    return 0;
  }
  t1 = wall_seconds_cuda_local();
  fmmNearGpuD2HTime += (t1 - t0);
  return 1;
}

int gpuDirectApply(ssystem *sys, double alpha, double beta, const double *sgm, double *pot) {
  cudaError_t err;
  size_t vecBytes;
  int blockSize;
  int gridSize;
  int dstStart;
  double t0, t1;

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
  t0 = wall_seconds_cuda_local();
  err = cudaMemcpy(gDirect.d_sgm, sgm, vecBytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gDirect.d_pot, pot, vecBytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  t1 = wall_seconds_cuda_local();
  directGpuH2DTime += (t1 - t0);

  blockSize = 256;
  if (gDirect.mode == 1) {
    gridSize = gDirect.nPnls;
    t0 = wall_seconds_cuda_local();
    directApplyKernel<<<gridSize, blockSize>>>(
        gDirect.nPnls, gDirect.d_k0, gDirect.d_k1, gDirect.d_k2, gDirect.d_k3,
        alpha, beta, gDirect.d_sgm, gDirect.d_pot);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 0;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return 0;
    t1 = wall_seconds_cuda_local();
    directGpuKernelTime += (t1 - t0);
  } else if (gDirect.mode == 2) {
    kernel = kernelKER4;
    for (dstStart = 0; dstStart < gDirect.nPnls; dstStart += gDirect.blockDstCount) {
      int dstCount = MIN(gDirect.blockDstCount, gDirect.nPnls - dstStart);
      size_t coeffBytes = (size_t)dstCount * (size_t)gDirect.nPnls * sizeof(double);
      int localD, s;

      t0 = wall_seconds_cuda_local();
      {
        int nThreads = directBuildThreadCount(dstCount);
        if (nThreads <= 1) {
          for (localD = 0; localD < dstCount; localD++) {
            int d = dstStart + localD;
            panel *pnlX = sys->panelByIdx[d];
            long long base = (long long)localD * (long long)gDirect.nPnls;
            for (s = 0; s < gDirect.nPnls; s++) {
              panel *pnlY = sys->panelByIdx[s];
              double *KER = panelIA0(pnlX, pnlY);
              long long idx = base + (long long)s;
              gDirect.h_k0[idx] = KER[0];
              gDirect.h_k1[idx] = KER[1];
              gDirect.h_k2[idx] = KER[2];
              gDirect.h_k3[idx] = KER[3];
            }
          }
        } else {
          std::vector<std::thread> workers;
          int t;

          workers.reserve((size_t)nThreads);
          for (t = 0; t < nThreads; t++) {
            int begin = (dstCount * t) / nThreads;
            int end = (dstCount * (t + 1)) / nThreads;
            workers.emplace_back([=]() {
              int localDst, localS;
              for (localDst = begin; localDst < end; localDst++) {
                int d = dstStart + localDst;
                panel *pnlX = sys->panelByIdx[d];
                long long base = (long long)localDst * (long long)gDirect.nPnls;
                for (localS = 0; localS < gDirect.nPnls; localS++) {
                  panel *pnlY = sys->panelByIdx[localS];
                  double *KER = panelIA0(pnlX, pnlY);
                  long long idx = base + (long long)localS;
                  gDirect.h_k0[idx] = KER[0];
                  gDirect.h_k1[idx] = KER[1];
                  gDirect.h_k2[idx] = KER[2];
                  gDirect.h_k3[idx] = KER[3];
                }
              }
            });
          }
          for (t = 0; t < nThreads; t++) {
            workers[t].join();
          }
        }
      }
      t1 = wall_seconds_cuda_local();
      directGpuCoeffTime += (t1 - t0);

      t0 = wall_seconds_cuda_local();
      err = cudaMemcpy(gDirect.d_k0, gDirect.h_k0, coeffBytes, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) return 0;
      err = cudaMemcpy(gDirect.d_k1, gDirect.h_k1, coeffBytes, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) return 0;
      err = cudaMemcpy(gDirect.d_k2, gDirect.h_k2, coeffBytes, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) return 0;
      err = cudaMemcpy(gDirect.d_k3, gDirect.h_k3, coeffBytes, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) return 0;
      t1 = wall_seconds_cuda_local();
      directGpuStoreTime += (t1 - t0);

      gridSize = dstCount;
      t0 = wall_seconds_cuda_local();
      directApplyBlockKernel<<<gridSize, blockSize>>>(
          gDirect.nPnls, dstStart, dstCount,
          gDirect.d_k0, gDirect.d_k1, gDirect.d_k2, gDirect.d_k3,
          alpha, beta, gDirect.d_sgm, gDirect.d_pot);
      err = cudaGetLastError();
      if (err != cudaSuccess) return 0;
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) return 0;
      t1 = wall_seconds_cuda_local();
      directGpuKernelTime += (t1 - t0);
      beta = 1.0;
    }
  } else {
    return 0;
  }

  t0 = wall_seconds_cuda_local();
  err = cudaMemcpy(pot, gDirect.d_pot, vecBytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  t1 = wall_seconds_cuda_local();
  directGpuD2HTime += (t1 - t0);
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

  if (gM2L.streaming) {
    if (!applyM2LStreamingChunks(sys)) return 0;
  } else {
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
  }

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

int gpuBuildPrecondDisjointBlock(panel **panels, int nPanels,
                                 const int *dstLocal, const int *srcLocal,
                                 int nPairs, double *block) {
  cudaError_t err;
  int i, blockSize, gridSize;
  std::lock_guard<std::mutex> lock(gPrecondMutex);

  if (panels == NULL || dstLocal == NULL || srcLocal == NULL || block == NULL ||
      nPanels <= 0 || nPairs <= 0) {
    return 0;
  }

  if (!ensurePrecondCapacity(nPanels, nPairs)) {
    return 0;
  }

  for (i = 0; i < nPanels; i++) {
    panel *p = panels[i];
    int j, k;
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
        gPrecond.h_panels[i].vtx[j][k] = p->vtx[j][k];
        gPrecond.h_panels[i].a0[k] = p->a[0][k];
        gPrecond.h_panels[i].a1[k] = p->a[1][k];
        gPrecond.h_panels[i].a2[k] = p->a[2][k];
        gPrecond.h_panels[i].normal[k] = p->normal[k];
      }
    }
    gPrecond.h_panels[i].area = p->area;
  }
  memcpy(gPrecond.h_dst, dstLocal, (size_t)nPairs * sizeof(int));
  memcpy(gPrecond.h_src, srcLocal, (size_t)nPairs * sizeof(int));

  if (cudaMemcpyToSymbol(c_nearKappa, &kappa, sizeof(double)) != cudaSuccess) return 0;
  if (cudaMemcpyToSymbol(c_nearEpsilon, &epsilon, sizeof(double)) != cudaSuccess) return 0;
  err = cudaMemcpy(gPrecond.d_panels, gPrecond.h_panels,
                   (size_t)nPanels * sizeof(NearPanelGeom), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gPrecond.d_dst, gPrecond.h_dst,
                   (size_t)nPairs * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gPrecond.d_src, gPrecond.h_src,
                   (size_t)nPairs * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return 0;

  blockSize = 256;
  gridSize = (nPairs + blockSize - 1) / blockSize;
  nearfieldDisjointQ1BuildKernel<<<gridSize, blockSize>>>(
      (long long)nPairs,
      gPrecond.d_panels,
      gPrecond.d_src,
      gPrecond.d_dst,
      gPrecond.d_k0,
      gPrecond.d_k1,
      gPrecond.d_k2,
      gPrecond.d_k3);
  err = cudaGetLastError();
  if (err != cudaSuccess) return 0;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return 0;

  err = cudaMemcpy(gPrecond.h_k0, gPrecond.d_k0,
                   (size_t)nPairs * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gPrecond.h_k1, gPrecond.d_k1,
                   (size_t)nPairs * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gPrecond.h_k2, gPrecond.d_k2,
                   (size_t)nPairs * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;
  err = cudaMemcpy(gPrecond.h_k3, gPrecond.d_k3,
                   (size_t)nPairs * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return 0;

  for (i = 0; i < nPairs; i++) {
    int dst = dstLocal[i];
    int src = srcLocal[i];
    int hm = nPanels;
    int msize = 2 * hm;
    block[dst * msize + src] = -gPrecond.h_k1[i];
    block[dst * msize + src + hm] = -gPrecond.h_k0[i];
    block[(dst + hm) * msize + src] = -gPrecond.h_k3[i];
    block[(dst + hm) * msize + src + hm] = -gPrecond.h_k2[i];
  }
  return 1;
}

/* ===================================================================
 * GPU charge-tree energy evaluator
 *
 * The post-solve energy walks the charge tree once per panel: accept a
 * cluster if it is far enough (eRad < theta*dist) and evaluate its multipole
 * expansion, otherwise descend, and sum charges directly in the leaves. On the
 * virus that was the single largest stage of the solve (445 s of ~983 s) and
 * the largest piece still on the CPU.
 *
 * The tree is flattened once into contiguous arrays. The traversal itself uses
 * an explicit stack rather than recursion.
 *
 * One WARP handles one panel, not one thread. The screened-derivative
 * recurrence needs a scratch tensor per target -- about 2 KB at the order 4
 * that most accepted nodes use -- which is far too much to give every thread,
 * but fits comfortably in shared memory once per warp. The warp also splits
 * the moment sum and the leaf charge sum across its lanes.
 * =================================================================== */

#define CTE_MAX_STACK 64
#define CTE_WARP 32

struct ChargeTreeGpu {
  const ssystem *sys;
  int nNodes;
  int nCharges;
  int nPanels;
  int derivOrderMax;        /* largest order+1 used by any accepted node */
  int idxDim;               /* idx3 flattened as idxDim^3 */
  int nMomMax;              /* nMom[derivOrderMax] */
  /* node arrays */
  double *d_nodeCtr;        /* 3 per node */
  double *d_nodeRad;
  int *d_nodeLevel;
  int *d_nodeKidStart;
  int *d_nodeKidCount;
  int *d_nodeChgStart;
  int *d_nodeChgCount;
  int *d_nodeMomOff;
  int *d_nodeOrder;         /* rhsChargeExpansionOrder for this node's level */
  int *d_kidIdx;
  int *d_chgIdx;
  double *d_moments;
  /* charges and panels */
  double *d_pos;
  double *d_chr;
  NearPanelGeom *d_panels;
  /* small tables */
  int *d_idx3;
  int *d_sgn3;
  int *d_nMom;
  /* results */
  double *d_out;            /* 2 per panel: y0, y1 */
};

static ChargeTreeGpu gCTE = {};

static void freeChargeTreeGpu(void) {
  cudaFree(gCTE.d_nodeCtr);      cudaFree(gCTE.d_nodeRad);
  cudaFree(gCTE.d_nodeLevel);    cudaFree(gCTE.d_nodeKidStart);
  cudaFree(gCTE.d_nodeKidCount); cudaFree(gCTE.d_nodeChgStart);
  cudaFree(gCTE.d_nodeChgCount); cudaFree(gCTE.d_nodeMomOff);
  cudaFree(gCTE.d_nodeOrder);    cudaFree(gCTE.d_kidIdx);
  cudaFree(gCTE.d_chgIdx);       cudaFree(gCTE.d_moments);
  cudaFree(gCTE.d_pos);          cudaFree(gCTE.d_chr);
  cudaFree(gCTE.d_panels);       cudaFree(gCTE.d_idx3);
  cudaFree(gCTE.d_sgn3);         cudaFree(gCTE.d_nMom);
  cudaFree(gCTE.d_out);
  memset(&gCTE, 0, sizeof(gCTE));
}

/* Depth-first flatten of the charge tree into contiguous arrays. */
static int buildChargeTreeGpu(const ssystem *sys) {
  std::vector<const cube *> order;
  std::vector<int> nodeOf;           /* scratch: index of each visited node */
  std::vector<double> ctr, rad, mom;
  std::vector<int> lev, kidStart, kidCount, chgStart, chgCount, momOff, nodeOrd;
  std::vector<int> kidIdx, chgIdx;
  std::vector<const cube *> stack;
  size_t i;
  int derivMax = 0;

  if (sys->chgCubeList == NULL || sys->chgCubeList[0] == NULL) {
    setNearfieldLastError("charge tree not built");
    return 0;
  }

  /* First pass: assign an index to every node, breadth-first from the root. */
  order.push_back(sys->chgCubeList[0]);
  for (i = 0; i < order.size(); i++) {
    const cube *cb = order[i];
    int k;
    for (k = 0; k < cb->nKids; k++) {
      order.push_back(cb->kids[k]);
    }
  }

  /* Map each node pointer to its index so kid lists can be flattened. */
  {
    std::vector<std::pair<const cube *, int> > lookup;
    lookup.reserve(order.size());
    for (i = 0; i < order.size(); i++) lookup.push_back(std::make_pair(order[i], (int)i));
    std::sort(lookup.begin(), lookup.end());

    for (i = 0; i < order.size(); i++) {
      const cube *cb = order[i];
      int ordHere = rhsChargeExpansionOrder((ssystem *)sys, cb->level);
      int nMomHere = sys->nMom[ordHere];
      int k;

      ctr.push_back(cb->x[0]); ctr.push_back(cb->x[1]); ctr.push_back(cb->x[2]);
      rad.push_back(cb->eRad);
      lev.push_back(cb->level);
      nodeOrd.push_back(ordHere);
      if (ordHere + 1 > derivMax) derivMax = ordHere + 1;

      kidStart.push_back((int)kidIdx.size());
      kidCount.push_back(cb->nKids);
      for (k = 0; k < cb->nKids; k++) {
        std::pair<const cube *, int> probe(cb->kids[k], 0);
        std::vector<std::pair<const cube *, int> >::iterator it =
            std::lower_bound(lookup.begin(), lookup.end(), probe);
        if (it == lookup.end() || it->first != cb->kids[k]) {
          setNearfieldLastError("charge tree flatten: child not found");
          return 0;
        }
        kidIdx.push_back(it->second);
      }

      chgStart.push_back((int)chgIdx.size());
      if (cb->level == sys->chgDepth && cb->chgIdx != NULL) {
        chgCount.push_back(cb->nChgs);
        for (k = 0; k < cb->nChgs; k++) chgIdx.push_back(cb->chgIdx[k]);
      } else {
        chgCount.push_back(0);
      }

      momOff.push_back((int)mom.size());
      if (cb->mom_chr != NULL) {
        for (k = 0; k < nMomHere; k++) mom.push_back(cb->mom_chr[k]);
      } else {
        for (k = 0; k < nMomHere; k++) mom.push_back(0.0);
      }
    }
  }

  gCTE.nNodes = (int)order.size();
  gCTE.nCharges = sys->nChar;
  gCTE.nPanels = sys->nPnls;
  gCTE.derivOrderMax = derivMax;
  gCTE.nMomMax = sys->nMom[derivMax];
  /* idx3 was allocated by mkIndex(maxOrder+1), so its valid extent is
   * maxOrder+2 per dimension. The kernel reads it at i1+1 with
   * i1+i2+i3 <= order <= maxOrder, so maxOrder+1 is the largest index needed
   * and that extent is exactly right. Deriving the dimension from derivMax
   * instead would read past the table. */
  gCTE.idxDim = sys->maxOrder + 2;

  /* Flatten idx3 and sgn3 for the device. */
  {
    int D = gCTE.idxDim;
    std::vector<int> fidx((size_t)D * D * D, 0);
    std::vector<int> fsgn((size_t)gCTE.nMomMax, 0);
    int a, b, c;
    for (a = 0; a < D; a++)
      for (b = 0; b < D; b++)
        for (c = 0; c < D; c++)
          if (a + b + c <= sys->maxOrder + 1) fidx[((size_t)a * D + b) * D + c] = idx3[a][b][c];
    for (a = 0; a < gCTE.nMomMax; a++) fsgn[a] = sgn3[a];

    std::vector<int> fnmom((size_t)derivMax + 2, 0);
    for (a = 0; a <= derivMax + 1; a++) fnmom[a] = sys->nMom[a];

    if (!cudaMallocNearfield((void **)&gCTE.d_idx3, fidx.size() * sizeof(int), "cte idx3") ||
        !cudaMallocNearfield((void **)&gCTE.d_sgn3, fsgn.size() * sizeof(int), "cte sgn3") ||
        !cudaMallocNearfield((void **)&gCTE.d_nMom, fnmom.size() * sizeof(int), "cte nMom") ||
        !cudaMemcpyNearfield(gCTE.d_idx3, &fidx[0], fidx.size() * sizeof(int),
                             cudaMemcpyHostToDevice, "cte idx3") ||
        !cudaMemcpyNearfield(gCTE.d_sgn3, &fsgn[0], fsgn.size() * sizeof(int),
                             cudaMemcpyHostToDevice, "cte sgn3") ||
        !cudaMemcpyNearfield(gCTE.d_nMom, &fnmom[0], fnmom.size() * sizeof(int),
                             cudaMemcpyHostToDevice, "cte nMom")) return 0;
  }

#define CTE_UP(dst, vec, type, tag)                                              \
  do {                                                                           \
    if (!cudaMallocNearfield((void **)&(dst), (vec).size() * sizeof(type), tag) ||\
        !cudaMemcpyNearfield((dst), &(vec)[0], (vec).size() * sizeof(type),       \
                             cudaMemcpyHostToDevice, tag)) return 0;              \
  } while (0)

  CTE_UP(gCTE.d_nodeCtr, ctr, double, "cte ctr");
  CTE_UP(gCTE.d_nodeRad, rad, double, "cte rad");
  CTE_UP(gCTE.d_nodeLevel, lev, int, "cte lev");
  CTE_UP(gCTE.d_nodeKidStart, kidStart, int, "cte kidStart");
  CTE_UP(gCTE.d_nodeKidCount, kidCount, int, "cte kidCount");
  CTE_UP(gCTE.d_nodeChgStart, chgStart, int, "cte chgStart");
  CTE_UP(gCTE.d_nodeChgCount, chgCount, int, "cte chgCount");
  CTE_UP(gCTE.d_nodeMomOff, momOff, int, "cte momOff");
  CTE_UP(gCTE.d_nodeOrder, nodeOrd, int, "cte order");
  CTE_UP(gCTE.d_moments, mom, double, "cte moments");
  if (!kidIdx.empty()) CTE_UP(gCTE.d_kidIdx, kidIdx, int, "cte kidIdx");
  if (!chgIdx.empty()) CTE_UP(gCTE.d_chgIdx, chgIdx, int, "cte chgIdx");
#undef CTE_UP

  /* charges */
  if (!cudaMallocNearfield((void **)&gCTE.d_pos, (size_t)3 * sys->nChar * sizeof(double), "cte pos") ||
      !cudaMallocNearfield((void **)&gCTE.d_chr, (size_t)sys->nChar * sizeof(double), "cte chr") ||
      !cudaMemcpyNearfield(gCTE.d_pos, sys->pos, (size_t)3 * sys->nChar * sizeof(double),
                           cudaMemcpyHostToDevice, "cte pos") ||
      !cudaMemcpyNearfield(gCTE.d_chr, sys->chr, (size_t)sys->nChar * sizeof(double),
                           cudaMemcpyHostToDevice, "cte chr")) return 0;

  /* panel geometry: same layout the nearfield already uses */
  {
    std::vector<NearPanelGeom> geom((size_t)sys->nPnls);
    panel *pnl;
    int idx, k;
    for (idx = 0, pnl = sys->pnlLst; pnl != NULL && idx < sys->nPnls; pnl = pnl->nextC, idx++) {
      for (k = 0; k < 3; k++) {
        geom[(size_t)idx].vtx[0][k] = pnl->vtx[0][k];
        geom[(size_t)idx].a0[k] = pnl->a[0][k];
        geom[(size_t)idx].a2[k] = pnl->a[2][k];
        geom[(size_t)idx].normal[k] = pnl->normal[k];
      }
      geom[(size_t)idx].area = pnl->area;
    }
    if (!cudaMallocNearfield((void **)&gCTE.d_panels,
                             (size_t)sys->nPnls * sizeof(NearPanelGeom), "cte panels") ||
        !cudaMemcpyNearfield(gCTE.d_panels, &geom[0],
                             (size_t)sys->nPnls * sizeof(NearPanelGeom),
                             cudaMemcpyHostToDevice, "cte panels")) return 0;
  }

  if (!cudaMallocNearfield((void **)&gCTE.d_out,
                           (size_t)2 * sys->nPnls * sizeof(double), "cte out")) return 0;

  gCTE.sys = sys;
  if (sys->benchmarkMode > 0) {
    printf("GPU charge-tree energy: nodes=%d charges=%d panels=%d derivMax=%d "
           "moments=%.3f GiB\n",
           gCTE.nNodes, gCTE.nCharges, gCTE.nPanels, gCTE.derivOrderMax,
           bytesToGiB(mom.size() * sizeof(double)));
  }
  return 1;
}

/* Radial derivatives of G0 and Gk; port of kernelDG0DGk() in kernel.c. */
__device__ __forceinline__ void cteRadialDerivs(double r, int p, double kap,
                                                double fourPiIv,
                                                double *G0, double *Gk) {
  double kappa2 = kap * kap;
  double rexp = exp(-kap * r);
  double r1 = 1.0 / r;
  double r2 = r1 * r1;
  double nr2 = -r2;
  int k;

  G0[0] = r1 * fourPiIv;
  Gk[0] = rexp * G0[0];
  G0[1] = nr2 * G0[0];
  Gk[1] = -Gk[0] * (r2 + kap * r1);
  for (k = 1; k < p; k++) {
    G0[k + 1] = (2 * k + 1) * nr2 * G0[k];
    Gk[k + 1] = r2 * (kappa2 * Gk[k - 1] - (2 * k + 1) * Gk[k]);
  }
}

/*
 * Cartesian derivative tensors of G0/Gk about a cluster centre; port of
 * setupScreenedDerivsLocal() in treecode.c. Lane 0 of the warp drives the
 * recurrence: each order p is built from p+1, so the levels are sequential,
 * and the per-level work is far too small to be worth splitting across lanes.
 */
__device__ void cteSetupDerivsStrided(int order, const double *x,
                               double kap, double fourPiIv,
                               const int *fidx3, int D,
                               double *gv0, double *gvk,
                               double **dg0, double **dgk, long long S) {
  int p, p1, iRow, iRow1, i1, i2, i3, idx, idx1, idx2;
  double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);

  cteRadialDerivs(r, order, kap, fourPiIv, gv0, gvk);
  for (p = 0; p <= order; p++) { dg0[p][(long long)(0) * S] = gv0[p]; dgk[p][(long long)(0) * S] = gvk[p]; }
  for (p = 0; p < order; p++) {
    p1 = p + 1;
    dg0[p][(long long)(1) * S] = dg0[p1][(long long)(0) * S] * x[2];  dg0[p][(long long)(2) * S] = dg0[p1][(long long)(0) * S] * x[1];  dg0[p][(long long)(3) * S] = dg0[p1][(long long)(0) * S] * x[0];
    dgk[p][(long long)(1) * S] = dgk[p1][(long long)(0) * S] * x[2];  dgk[p][(long long)(2) * S] = dgk[p1][(long long)(0) * S] * x[1];  dgk[p][(long long)(3) * S] = dgk[p1][(long long)(0) * S] * x[0];
  }
#define CTE_IDX(a,b,c) fidx3[(((a) * D) + (b)) * D + (c)]
  for (iRow = 2; iRow <= order; iRow++) {
    for (p = 0; p <= order - iRow; p++) {
      p1 = p + 1;
      idx = CTE_IDX(0, 0, iRow);
      iRow1 = iRow - 1;
      idx1 = CTE_IDX(0, 0, iRow1);
      idx2 = CTE_IDX(0, 0, iRow - 2);
      dg0[p][(long long)(idx) * S] = dg0[p1][(long long)(idx1) * S] * x[2] + dg0[p1][(long long)(idx2) * S] * iRow1;
      dgk[p][(long long)(idx) * S] = dgk[p1][(long long)(idx1) * S] * x[2] + dgk[p1][(long long)(idx2) * S] * iRow1;
      idx++;
      idx1 = CTE_IDX(0, 0, iRow1);
      dg0[p][(long long)(idx) * S] = dg0[p1][(long long)(idx1) * S] * x[1];
      dgk[p][(long long)(idx) * S] = dgk[p1][(long long)(idx1) * S] * x[1];
      idx++;
      for (i2 = 2; i2 <= iRow; i2++, idx++) {
        i3 = iRow - i2;
        idx1 = CTE_IDX(0, i2 - 1, i3);
        idx2 = CTE_IDX(0, i2 - 2, i3);
        dg0[p][(long long)(idx) * S] = dg0[p1][(long long)(idx1) * S] * x[1] + dg0[p1][(long long)(idx2) * S] * (i2 - 1);
        dgk[p][(long long)(idx) * S] = dgk[p1][(long long)(idx1) * S] * x[1] + dgk[p1][(long long)(idx2) * S] * (i2 - 1);
      }
      for (i2 = 0; i2 <= iRow1; i2++, idx++) {
        i3 = iRow1 - i2;
        idx1 = CTE_IDX(0, i2, i3);
        dg0[p][(long long)(idx) * S] = dg0[p1][(long long)(idx1) * S] * x[0];
        dgk[p][(long long)(idx) * S] = dgk[p1][(long long)(idx1) * S] * x[0];
      }
      for (i1 = 2; i1 <= iRow; i1++) {
        for (i2 = 0; i2 <= iRow - i1; i2++, idx++) {
          i3 = iRow - i1 - i2;
          idx1 = CTE_IDX(i1 - 1, i2, i3);
          idx2 = CTE_IDX(i1 - 2, i2, i3);
          dg0[p][(long long)(idx) * S] = dg0[p1][(long long)(idx1) * S] * x[0] + dg0[p1][(long long)(idx2) * S] * (i1 - 1);
          dgk[p][(long long)(idx) * S] = dgk[p1][(long long)(idx1) * S] * x[0] + dgk[p1][(long long)(idx2) * S] * (i1 - 1);
        }
      }
    }
  }
#undef CTE_IDX
}

/*
 * One thread per panel, grid-stride.
 *
 * The first version gave a warp to each panel with the derivative scratch in
 * shared memory. That was correct but 5.6x slower than 72 CPU threads: all 32
 * lanes walked the same tree redundantly, lane 0 ran the derivative recurrence
 * alone while the rest idled, and 43 KB of shared memory per block held
 * occupancy to a single block per SM.
 *
 * Here each thread owns a panel and its own scratch slab in global memory. The
 * slabs are interleaved -- entry i of thread t lives at [i * stride + t] -- so
 * lanes touching the same tensor entry hit consecutive addresses. Concurrency
 * is capped and panels are taken in a grid-stride loop, which bounds the
 * scratch: the whole point is that it is proportional to the number of resident
 * threads, not to the panel count.
 *
 * Per-level sizes follow the CPU: level p holds nMom[derivMax - p] entries,
 * which the recurrence never exceeds because p + iRow <= order <= derivMax.
 */
__global__ void chargeTreeEnergyKernel(
    int nPanels, int derivMax, int idxDim, int chgDepth, int height,
    double theta, double kap, double eps, double fourPiIv,
    const NearPanelGeom *panels,
    const double *nodeCtr, const double *nodeRad, const int *nodeLevel,
    const int *nodeKidStart, const int *nodeKidCount,
    const int *nodeChgStart, const int *nodeChgCount,
    const int *nodeMomOff, const int *nodeOrder,
    const int *kidIdx, const int *chgIdx, const double *moments,
    const double *pos, const double *chr,
    const int *fidx3, const int *sgn3d, const int *levOff,
    double *scratch, long long stride,
    double *out) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long nThreadsTotal = (long long)gridDim.x * blockDim.x;
  double *dg0[16];
  double *dgk[16];
  double gv0[18], gvk[18];
  int p;
  int panelIdx;

  if (tid >= stride) return;
  for (p = 0; p <= derivMax && p < 16; p++) {
    dg0[p] = scratch + (long long)levOff[p] * stride + tid;
    dgk[p] = scratch + ((long long)levOff[derivMax + 1] + levOff[p]) * stride + tid;
  }

  for (panelIdx = (int)tid; panelIdx < nPanels; panelIdx += (int)nThreadsTotal) {
    const NearPanelGeom *pn = &panels[panelIdx];
    double qpt[3], nrm[3];
    double y0 = 0.0, y1 = 0.0;
    int stack[CTE_MAX_STACK];
    int sp = 0;
    int k;

    for (k = 0; k < 3; k++) {
      qpt[k] = pn->vtx[0][k] + 0.5 * (pn->a2[k] + 0.5 * pn->a0[k]);
      nrm[k] = pn->normal[k];
    }
    stack[sp++] = 0;

    while (sp > 0) {
      int node = stack[--sp];
      double r[3], dist;
      for (k = 0; k < 3; k++) r[k] = qpt[k] - nodeCtr[3 * node + k];
      dist = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

      if (nodeRad[node] < theta * dist && nodeLevel[node] >= height) {
        int ordHere = nodeOrder[node];
        const double *mom = &moments[nodeMomOff[node]];
        int i, i1, i2, n;

        cteSetupDerivsStrided(ordHere + 1, r, kap, fourPiIv, fidx3, idxDim,
                              gv0, gvk, dg0, dgk, stride);
        i = 0;
        for (n = 0; n <= ordHere; n++) {
          for (i1 = 0; i1 <= n; i1++) {
            for (i2 = 0; i2 <= n - i1; i2++, i++) {
              int i3 = n - i1 - i2;
              double dnG0, dnGk;
#define CTE_IDX(a,b,c) fidx3[(((a) * idxDim) + (b)) * idxDim + (c)]
              dnG0 = nrm[0] * dg0[0][(long long)CTE_IDX(i1 + 1, i2, i3) * stride]
                   + nrm[1] * dg0[0][(long long)CTE_IDX(i1, i2 + 1, i3) * stride]
                   + nrm[2] * dg0[0][(long long)CTE_IDX(i1, i2, i3 + 1) * stride];
              dnGk = nrm[0] * dgk[0][(long long)CTE_IDX(i1 + 1, i2, i3) * stride]
                   + nrm[1] * dgk[0][(long long)CTE_IDX(i1, i2 + 1, i3) * stride]
                   + nrm[2] * dgk[0][(long long)CTE_IDX(i1, i2, i3 + 1) * stride];
#undef CTE_IDX
              y0 += sgn3d[i] * mom[i] * (dg0[0][(long long)i * stride] -
                                         dgk[0][(long long)i * stride]);
              y1 += sgn3d[i] * mom[i] * (eps * dnGk - dnG0);
            }
          }
        }
        continue;
      }

      if (nodeLevel[node] == chgDepth) {
        int start = nodeChgStart[node];
        int cnt = nodeChgCount[node];
        int i;
        for (i = 0; i < cnt; i++) {
          int j = chgIdx[start + i];
          double x[3], r2, rn, G0, Gk, coef, ip, dG0dn, dGkdn;
          for (k = 0; k < 3; k++) x[k] = qpt[k] - pos[3 * j + k];
          r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
          rn = sqrt(r2);
          G0 = fourPiIv / rn;
          Gk = exp(-kap * rn) * G0;
          coef = (kap * rn + 1.0) * exp(-kap * rn);
          ip = nrm[0] * x[0] + nrm[1] * x[1] + nrm[2] * x[2];
          dG0dn = -ip * G0 / r2;
          dGkdn = coef * dG0dn;
          y0 += chr[j] * (G0 - Gk);
          y1 += chr[j] * (eps * dGkdn - dG0dn);
        }
        continue;
      }

      {
        /* Push children in reverse so they pop in ascending order, matching the
         * order the CPU recursion visits them. Without this the subtrees are
         * summed back-to-front and the result differs at roundoff (4.2e-12
         * measured on the virus RHS) for no reason. */
        int ks = nodeKidStart[node];
        int kc = nodeKidCount[node];
        int i;
        for (i = kc - 1; i >= 0 && sp < CTE_MAX_STACK; i--) stack[sp++] = kidIdx[ks + i];
      }
    }

    /* qOrd==1: weight 0.5 then the 2*area factor, so net = area. */
    out[2 * panelIdx] = y0 * pn->area;
    out[2 * panelIdx + 1] = y1 * pn->area;
  }
}

extern "C" double energyTreeTheta(void);

int gpuPanelChargeTreeEnergy(ssystem *sys, const double *sgm, double *pot) {
  int blockSize = 128, gridSize;
  long long stride, slabDoubles;
  size_t scratchBytes;
  cudaError_t err;
  std::vector<double> host;
  std::vector<int> levOff;
  double total = 0.0;
  int p, i;
  static double *d_scratch = NULL;
  static int *d_levOff = NULL;
  static long long scratchStride = 0;

  if (sys == NULL || sgm == NULL || pot == NULL) return 0;
  if (!gpuBackendAvailable()) return 0;
  if (sys->maxQuadOrder != 1) return 0;   /* kernel assumes the 1-point rule */
  if (sys->chgCubeList == NULL || sys->chgCubeList[0] == NULL) return 0;

  if (gCTE.sys != sys) {
    freeChargeTreeGpu();
    cudaFree(d_scratch); d_scratch = NULL;
    cudaFree(d_levOff);  d_levOff = NULL;
    scratchStride = 0;
    if (!buildChargeTreeGpu(sys)) { freeChargeTreeGpu(); return 0; }
  }
  if (gCTE.derivOrderMax + 1 >= 16) return 0;   /* fixed pointer arrays */

  /* Level p holds nMom[derivMax - p] entries, as on the CPU. levOff[derivMax+1]
   * is the total, which also separates the dg0 half from the dgk half. */
  levOff.resize((size_t)gCTE.derivOrderMax + 2);
  levOff[0] = 0;
  for (p = 0; p <= gCTE.derivOrderMax; p++) {
    levOff[(size_t)p + 1] = levOff[(size_t)p] + sys->nMom[gCTE.derivOrderMax - p];
  }
  slabDoubles = 2LL * levOff[(size_t)gCTE.derivOrderMax + 1];

  /* Cap concurrency so the scratch stays bounded; panels are then taken in a
   * grid-stride loop. See chargeTreeScratchThreads for how perSm was measured. */
  stride = chargeTreeScratchThreads("FABIPB_ENERGY_GPU_SCRATCH_MIB", 160,
                                    slabDoubles, gCTE.nPanels, blockSize);
  scratchBytes = (size_t)stride * (size_t)slabDoubles * sizeof(double);

  if (scratchStride != stride) {
    cudaFree(d_scratch); d_scratch = NULL;
    if (!cudaMallocNearfield((void **)&d_scratch, scratchBytes, "cte scratch")) {
      scratchStride = 0;
      return 0;
    }
    scratchStride = stride;
  }
  if (d_levOff == NULL) {
    if (!cudaMallocNearfield((void **)&d_levOff, levOff.size() * sizeof(int), "cte levOff") ||
        !cudaMemcpyNearfield(d_levOff, &levOff[0], levOff.size() * sizeof(int),
                             cudaMemcpyHostToDevice, "cte levOff")) return 0;
  }

  gridSize = (int)(stride / blockSize);
  chargeTreeEnergyKernel<<<gridSize, blockSize>>>(
      gCTE.nPanels, gCTE.derivOrderMax, gCTE.idxDim, sys->chgDepth, sys->height,
      energyTreeTheta(), kappa, epsilon, fourPiI,
      gCTE.d_panels,
      gCTE.d_nodeCtr, gCTE.d_nodeRad, gCTE.d_nodeLevel,
      gCTE.d_nodeKidStart, gCTE.d_nodeKidCount,
      gCTE.d_nodeChgStart, gCTE.d_nodeChgCount,
      gCTE.d_nodeMomOff, gCTE.d_nodeOrder,
      gCTE.d_kidIdx, gCTE.d_chgIdx, gCTE.d_moments,
      gCTE.d_pos, gCTE.d_chr,
      gCTE.d_idx3, gCTE.d_sgn3, d_levOff,
      d_scratch, stride,
      gCTE.d_out);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    setNearfieldLastError("charge-tree energy launch failed: %s", cudaGetErrorString(err));
    return 0;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    setNearfieldLastError("charge-tree energy kernel failed: %s", cudaGetErrorString(err));
    return 0;
  }

  host.resize((size_t)2 * gCTE.nPanels);
  if (!cudaMemcpyNearfield(&host[0], gCTE.d_out,
                           (size_t)2 * gCTE.nPanels * sizeof(double),
                           cudaMemcpyDeviceToHost, "cte out")) return 0;

  /* Same combination as panelEnergyWorker(): entry i is panelByIdx[i], whose
   * own idx selects the solution components. */
  for (i = 0; i < gCTE.nPanels; i++) {
    panel *pnl = sys->panelByIdx[i];
    total += host[2 * i] * sgm[pnl->idx + sys->nPnls] + host[2 * i + 1] * sgm[pnl->idx];
  }
  *pot = total;
  if (sys->benchmarkMode > 0) {
    printf("GPU charge-tree energy: threads=%lld grid=%d scratch=%.3f GiB slab=%lld doubles\n",
           stride, gridSize, bytesToGiB(scratchBytes), slabDoubles);
  }
  return 1;
}


/* Coulomb-only derivative tensor; port of setupCoulombDerivsLocal(). Half the
 * work and half the scratch of the screened version -- no Gk half. */
__device__ void cteSetupCoulombDerivsStrided(int order, const double *x,
                                             double fourPiIv,
                                             const int *fidx3, int D,
                                             double *gv0, double **dg0,
                                             long long S) {
  int p, p1, iRow, iRow1, i1, i2, i3, idx, idx1, idx2, k;
  double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  double r2;

  gv0[0] = fourPiIv / r;
  r2 = -1.0 / (r * r);
  for (k = 0; k < order; k++) gv0[k + 1] = (2 * k + 1) * r2 * gv0[k];
  for (p = 0; p <= order; p++) dg0[p][0] = gv0[p];
  for (p = 0; p < order; p++) {
    p1 = p + 1;
    dg0[p][(long long)1 * S] = dg0[p1][0] * x[2];
    dg0[p][(long long)2 * S] = dg0[p1][0] * x[1];
    dg0[p][(long long)3 * S] = dg0[p1][0] * x[0];
  }
#define CTE_IDX(a,b,c) fidx3[(((a) * D) + (b)) * D + (c)]
  for (iRow = 2; iRow <= order; iRow++) {
    for (p = 0; p <= order - iRow; p++) {
      p1 = p + 1;
      idx = CTE_IDX(0, 0, iRow);
      iRow1 = iRow - 1;
      idx1 = CTE_IDX(0, 0, iRow1);
      idx2 = CTE_IDX(0, 0, iRow - 2);
      dg0[p][(long long)idx * S] = dg0[p1][(long long)idx1 * S] * x[2]
                                 + dg0[p1][(long long)idx2 * S] * iRow1;
      idx++;
      idx1 = CTE_IDX(0, 0, iRow1);
      dg0[p][(long long)idx * S] = dg0[p1][(long long)idx1 * S] * x[1];
      idx++;
      for (i2 = 2; i2 <= iRow; i2++, idx++) {
        i3 = iRow - i2;
        idx1 = CTE_IDX(0, i2 - 1, i3);
        idx2 = CTE_IDX(0, i2 - 2, i3);
        dg0[p][(long long)idx * S] = dg0[p1][(long long)idx1 * S] * x[1]
                                   + dg0[p1][(long long)idx2 * S] * (i2 - 1);
      }
      for (i2 = 0; i2 <= iRow1; i2++, idx++) {
        i3 = iRow1 - i2;
        idx1 = CTE_IDX(0, i2, i3);
        dg0[p][(long long)idx * S] = dg0[p1][(long long)idx1 * S] * x[0];
      }
      for (i1 = 2; i1 <= iRow; i1++) {
        for (i2 = 0; i2 <= iRow - i1; i2++, idx++) {
          i3 = iRow - i1 - i2;
          idx1 = CTE_IDX(i1 - 1, i2, i3);
          idx2 = CTE_IDX(i1 - 2, i2, i3);
          dg0[p][(long long)idx * S] = dg0[p1][(long long)idx1 * S] * x[0]
                                     + dg0[p1][(long long)idx2 * S] * (i1 - 1);
        }
      }
    }
  }
#undef CTE_IDX
}

/*
 * setupRHS's charge-tree walk. Same traversal as the energy kernel, but the
 * Coulomb-only kernel: no screening, no epsilon, and half the scratch, so more
 * of the working set stays in cache.
 */
__global__ void chargeTreeRhsKernel(
    int nPanels, int derivMax, int idxDim, int chgDepth, int height,
    double theta, double fourPiIv, double fac,
    const NearPanelGeom *panels,
    const double *nodeCtr, const double *nodeRad, const int *nodeLevel,
    const int *nodeKidStart, const int *nodeKidCount,
    const int *nodeChgStart, const int *nodeChgCount,
    const int *nodeMomOff, const int *nodeOrder,
    const int *kidIdx, const int *chgIdx, const double *moments,
    const double *pos, const double *chr,
    const int *fidx3, const int *sgn3d, const int *levOff,
    double *scratch, long long stride,
    double *out) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long nThreadsTotal = (long long)gridDim.x * blockDim.x;
  double *dg0[16];
  double gv0[18];
  int p, panelIdx;

  if (tid >= stride) return;
  for (p = 0; p <= derivMax && p < 16; p++) {
    dg0[p] = scratch + (long long)levOff[p] * stride + tid;
  }

  for (panelIdx = (int)tid; panelIdx < nPanels; panelIdx += (int)nThreadsTotal) {
    const NearPanelGeom *pn = &panels[panelIdx];
    double qpt[3], nrm[3];
    double y0 = 0.0, y1 = 0.0;
    int stack[CTE_MAX_STACK];
    int sp = 0;
    int k;

    for (k = 0; k < 3; k++) {
      qpt[k] = pn->vtx[0][k] + 0.5 * (pn->a2[k] + 0.5 * pn->a0[k]);
      nrm[k] = pn->normal[k];
    }
    stack[sp++] = 0;

    while (sp > 0) {
      int node = stack[--sp];
      double r[3], dist;
      for (k = 0; k < 3; k++) r[k] = qpt[k] - nodeCtr[3 * node + k];
      dist = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

      if (nodeRad[node] < theta * dist && nodeLevel[node] >= height) {
        int ordHere = nodeOrder[node];
        const double *mom = &moments[nodeMomOff[node]];
        double c0 = 0.0, c1 = 0.0;
        int i, i1, i2, n;

        cteSetupCoulombDerivsStrided(ordHere + 1, r, fourPiIv, fidx3, idxDim,
                                     gv0, dg0, stride);
        i = 0;
        for (n = 0; n <= ordHere; n++) {
          for (i1 = 0; i1 <= n; i1++) {
            for (i2 = 0; i2 <= n - i1; i2++, i++) {
              int i3 = n - i1 - i2;
              double dn;
#define CTE_IDX(a,b,c) fidx3[(((a) * idxDim) + (b)) * idxDim + (c)]
              dn = nrm[0] * dg0[0][(long long)CTE_IDX(i1 + 1, i2, i3) * stride]
                 + nrm[1] * dg0[0][(long long)CTE_IDX(i1, i2 + 1, i3) * stride]
                 + nrm[2] * dg0[0][(long long)CTE_IDX(i1, i2, i3 + 1) * stride];
#undef CTE_IDX
              c0 += sgn3d[i] * mom[i] * dg0[0][(long long)i * stride];
              c1 += sgn3d[i] * mom[i] * dn;
            }
          }
        }
        /* chgClusterEvalLocal divides both components by fourPiI */
        y0 += c0 / fourPiIv;
        y1 += c1 / fourPiIv;
        continue;
      }

      if (nodeLevel[node] == chgDepth) {
        int start = nodeChgStart[node];
        int cnt = nodeChgCount[node];
        int i;
        for (i = 0; i < cnt; i++) {
          int j = chgIdx[start + i];
          double x[3], r2, ri, r3i, ip;
          for (k = 0; k < 3; k++) x[k] = qpt[k] - pos[3 * j + k];
          r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
          ri = 1.0 / sqrt(r2);
          r3i = ri / r2;
          ip = nrm[0] * x[0] + nrm[1] * x[1] + nrm[2] * x[2];
          y0 += chr[j] * ri;
          y1 += chr[j] * (-ip * r3i);
        }
        continue;
      }

      {
        /* Push children in reverse so they pop in ascending order, matching the
         * order the CPU recursion visits them. Without this the subtrees are
         * summed back-to-front and the result differs at roundoff (4.2e-12
         * measured on the virus RHS) for no reason. */
        int ks = nodeKidStart[node];
        int kc = nodeKidCount[node];
        int i;
        for (i = kc - 1; i >= 0 && sp < CTE_MAX_STACK; i--) stack[sp++] = kidIdx[ks + i];
      }
    }

    /* qOrd==1: weight 0.5 then 2*area gives area; setupRHS then applies fac. */
    out[panelIdx] = y0 * pn->area * fac;
    out[nPanels + panelIdx] = y1 * pn->area * fac;
  }
}

extern "C" double rhsTreeTheta(void);

int gpuChargeTreeRHS(ssystem *sys, int qOrder, double fac, double *sgm) {
  int blockSize = 128, gridSize;
  long long stride, slabDoubles;
  size_t scratchBytes;
  cudaError_t err;
  std::vector<double> host;
  std::vector<int> levOff;
  int p, i;
  static double *d_scratch = NULL;
  static int *d_levOff = NULL;
  static long long scratchStride = 0;
  static const ssystem *scratchSys = NULL;

  if (sys == NULL || sgm == NULL) return 0;
  if (!gpuBackendAvailable()) return 0;
  if (qOrder != 1) return 0;              /* kernel assumes the 1-point rule */
  if (sys->chgCubeList == NULL || sys->chgCubeList[0] == NULL) return 0;
  if (envFlagEnabled("FABIPB_RHS_GPU_DISABLE")) return 0;

  if (gCTE.sys != sys) {
    freeChargeTreeGpu();
    if (!buildChargeTreeGpu(sys)) { freeChargeTreeGpu(); return 0; }
  }
  if (gCTE.derivOrderMax + 1 >= 16) return 0;

  if (scratchSys != sys) {
    cudaFree(d_scratch); d_scratch = NULL;
    cudaFree(d_levOff);  d_levOff = NULL;
    scratchStride = 0;
    scratchSys = sys;
  }

  /* Only the Coulomb half is needed here, so the slab is half the energy
   * kernel's and more of it stays resident. */
  levOff.resize((size_t)gCTE.derivOrderMax + 2);
  levOff[0] = 0;
  for (p = 0; p <= gCTE.derivOrderMax; p++) {
    levOff[(size_t)p + 1] = levOff[(size_t)p] + sys->nMom[gCTE.derivOrderMax - p];
  }
  slabDoubles = levOff[(size_t)gCTE.derivOrderMax + 1];

  /* Lower perSm than the energy kernel: this one steps up sharply in time past
   * roughly 16k threads on the measurement card. */
  stride = chargeTreeScratchThreads("FABIPB_RHS_GPU_SCRATCH_MIB", 112,
                                    slabDoubles, gCTE.nPanels, blockSize);
  scratchBytes = (size_t)stride * (size_t)slabDoubles * sizeof(double);

  if (scratchStride != stride) {
    cudaFree(d_scratch); d_scratch = NULL;
    if (!cudaMallocNearfield((void **)&d_scratch, scratchBytes, "rhs scratch")) {
      scratchStride = 0; return 0;
    }
    scratchStride = stride;
  }
  if (d_levOff == NULL) {
    if (!cudaMallocNearfield((void **)&d_levOff, levOff.size() * sizeof(int), "rhs levOff") ||
        !cudaMemcpyNearfield(d_levOff, &levOff[0], levOff.size() * sizeof(int),
                             cudaMemcpyHostToDevice, "rhs levOff")) return 0;
    }

  gridSize = (int)(stride / blockSize);
  chargeTreeRhsKernel<<<gridSize, blockSize>>>(
      gCTE.nPanels, gCTE.derivOrderMax, gCTE.idxDim, sys->chgDepth, sys->height,
      rhsTreeTheta(), fourPiI, fac,
      gCTE.d_panels,
      gCTE.d_nodeCtr, gCTE.d_nodeRad, gCTE.d_nodeLevel,
      gCTE.d_nodeKidStart, gCTE.d_nodeKidCount,
      gCTE.d_nodeChgStart, gCTE.d_nodeChgCount,
      gCTE.d_nodeMomOff, gCTE.d_nodeOrder,
      gCTE.d_kidIdx, gCTE.d_chgIdx, gCTE.d_moments,
      gCTE.d_pos, gCTE.d_chr,
      gCTE.d_idx3, gCTE.d_sgn3, d_levOff,
      d_scratch, stride,
      gCTE.d_out);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    setNearfieldLastError("charge-tree RHS launch failed: %s", cudaGetErrorString(err));
    return 0;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    setNearfieldLastError("charge-tree RHS kernel failed");
    return 0;
  }

  /* d_out is sized 2*nPanels, which is exactly the RHS layout. */
  host.resize((size_t)2 * gCTE.nPanels);
  if (!cudaMemcpyNearfield(&host[0], gCTE.d_out,
                           (size_t)2 * gCTE.nPanels * sizeof(double),
                           cudaMemcpyDeviceToHost, "rhs out")) return 0;

  /* The kernel indexes panels in pnlLst order; sgm is indexed by pnl->idx. */
  for (i = 0; i < gCTE.nPanels; i++) {
    panel *pnl = sys->panelByIdx[i];
    sgm[pnl->idx] = host[i];
    sgm[pnl->idx + sys->nPnls] = host[gCTE.nPanels + i];
  }
  if (sys->benchmarkMode > 0) {
    printf("GPU charge-tree RHS: threads=%lld scratch=%.3f GiB slab=%lld doubles theta=%g\n",
           stride, bytesToGiB(scratchBytes), slabDoubles, rhsTreeTheta());
  }
  return 1;
}
