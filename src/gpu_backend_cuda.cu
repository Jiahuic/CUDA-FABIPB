#include "gpu/gpu_backend_cuda_internal.h"

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

NearfieldGpuCache gNear = {};
char gNearfieldLastError[512] = "";
char gM2LLastError[512] = "";

M2LGpuCache gM2L = {};

LeafTransformGpuCache gLeaf = {};

DirectGpuCache gDirect = {};

PrecondGpuCache gPrecond = {};
std::mutex gPrecondMutex;

void setNearfieldLastError(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(gNearfieldLastError, sizeof(gNearfieldLastError), fmt, ap);
  va_end(ap);
}

void setM2LLastError(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(gM2LLastError, sizeof(gM2LLastError), fmt, ap);
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

int cudaMallocM2L(void **ptr, size_t bytes, const char *name) {
  cudaError_t err = cudaMalloc(ptr, bytes);
  if (err != cudaSuccess) {
    setM2LLastError("cudaMalloc %s failed for %.3f GiB: %s",
                    name, bytesToGiB(bytes), cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

int cudaMemcpyM2L(void *dst, const void *src, size_t bytes,
                  enum cudaMemcpyKind kind, const char *name) {
  cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
  if (err != cudaSuccess) {
    setM2LLastError("cudaMemcpy %s failed for %.3f GiB: %s",
                    name, bytesToGiB(bytes), cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

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

int nearfieldBuildThreadCount(int nTasks);
int gpuNearfieldDisjointBuildEnabled(const ssystem *sys);

#include "gpu/gpu_nearfield_cuda.inc"
#include "gpu/gpu_leaf_transform_cuda.inc"
#include "gpu/gpu_m2l_cuda.inc"
#include "gpu/gpu_direct_cuda.inc"
#include "gpu/gpu_rhs_cuda.inc"

extern "C" void gpuReleaseMatvecCaches(void) {
  freeNearfieldCache();
  freeM2LCache();
  freeLeafCache();
  freeDirectCache();
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
                                   double sqrtCoeff,
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
  } else if (sqrtCoeff > 0.0) {
    /*
     * threads * sqrt(slab) = const, per SM. Once the rolling-window kernel cut
     * the slab, a flat per-SM thread count stopped fitting: measured optima on
     * 7A6A were (slab 224 -> 32,768), (480 -> 22,784), (880 -> 16,384), whose
     * products with sqrt(slab) are 490k, 499k and 486k -- constant to 3%, where
     * the thread counts themselves span 2x and the byte budgets span 2x in the
     * other direction. Fitted on one card, so it is still overridable.
     */
    stride = (long long)((double)deviceSmCount() * sqrtCoeff /
                         sqrt((double)slabDoubles));
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

const char *gpuM2LLastError(void) {
  return gM2LLastError[0] ? gM2LLastError : "unknown M2L GPU failure";
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

  setM2LLastError("not attempted");
  if (sys == NULL || sys->nM2LPairsFlat <= 0) {
    setM2LLastError("invalid input or empty M2L pair list");
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
  if (err != cudaSuccess) {
    setM2LLastError("cudaMemcpy momPot host-to-device failed for %.3f GiB: %s",
                    bytesToGiB((size_t)gM2L.totalCubeCoeff * sizeof(double)),
                    cudaGetErrorString(err));
    return 0;
  }
  err = cudaMemcpy(gM2L.d_momDpdn, gM2L.h_momDpdn,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    setM2LLastError("cudaMemcpy momDpdn host-to-device failed for %.3f GiB: %s",
                    bytesToGiB((size_t)gM2L.totalCubeCoeff * sizeof(double)),
                    cudaGetErrorString(err));
    return 0;
  }
  err = cudaMemset(gM2L.d_lec1, 0, (size_t)gM2L.totalCubeCoeff * sizeof(double));
  if (err != cudaSuccess) { setM2LLastError("cudaMemset lec1 failed: %s", cudaGetErrorString(err)); return 0; }
  err = cudaMemset(gM2L.d_lec2, 0, (size_t)gM2L.totalCubeCoeff * sizeof(double));
  if (err != cudaSuccess) { setM2LLastError("cudaMemset lec2 failed: %s", cudaGetErrorString(err)); return 0; }
  err = cudaMemset(gM2L.d_lec3, 0, (size_t)gM2L.totalCubeCoeff * sizeof(double));
  if (err != cudaSuccess) { setM2LLastError("cudaMemset lec3 failed: %s", cudaGetErrorString(err)); return 0; }
  err = cudaMemset(gM2L.d_lec4, 0, (size_t)gM2L.totalCubeCoeff * sizeof(double));
  if (err != cudaSuccess) { setM2LLastError("cudaMemset lec4 failed: %s", cudaGetErrorString(err)); return 0; }

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
    if (err != cudaSuccess) {
      setM2LLastError("m2lGroupedKernel launch failed: %s", cudaGetErrorString(err));
      return 0;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      setM2LLastError("m2lGroupedKernel failed: %s", cudaGetErrorString(err));
      return 0;
    }
  }

  err = cudaMemcpy(gM2L.h_lec1, gM2L.d_lec1,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { setM2LLastError("cudaMemcpy lec1 device-to-host failed: %s", cudaGetErrorString(err)); return 0; }
  err = cudaMemcpy(gM2L.h_lec2, gM2L.d_lec2,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { setM2LLastError("cudaMemcpy lec2 device-to-host failed: %s", cudaGetErrorString(err)); return 0; }
  err = cudaMemcpy(gM2L.h_lec3, gM2L.d_lec3,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { setM2LLastError("cudaMemcpy lec3 device-to-host failed: %s", cudaGetErrorString(err)); return 0; }
  err = cudaMemcpy(gM2L.h_lec4, gM2L.d_lec4,
                   (size_t)gM2L.totalCubeCoeff * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { setM2LLastError("cudaMemcpy lec4 device-to-host failed: %s", cudaGetErrorString(err)); return 0; }

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

#include "gpu/gpu_preconditioner_cuda.inc"

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

/*
 * Depth-first stack depth for the charge-tree walk.
 *
 * Worst case is 7*chgDepth + 1: the walk pops one node and pushes up to eight,
 * a net +7, and at most chgDepth expansions happen before the path reaches a
 * leaf, which does not expand. That is exactly 64 at depth 9 -- the depth ZIKV
 * sdens=2 and H1N1 sdens=1 both run at -- so the previous value of 64 had zero
 * margin, and depth 10 (71) overflowed. On overflow the push loop simply
 * stopped, dropping whole subtrees with no error and returning a plausible but
 * wrong energy.
 *
 * 128 covers depth 18. cteStackDepthOk() refuses the GPU path above that so the
 * CPU fallback engages instead of truncating; the loop guard remains as a
 * backstop against the bound above ever being wrong.
 */
#define CTE_MAX_STACK 128

/* Largest chgDepth the device stack can hold. */
static int cteStackDepthOk(int chgDepth) {
  return (7 * chgDepth + 1) <= CTE_MAX_STACK;
}

/*
 * Row geometry of the Cartesian derivative tensor, from the nMom table already
 * on the device. nMom[m] counts rows 0..m, so row m starts at nMom[m-1] and
 * holds nMom[m]-nMom[m-1] = (m+1)(m+2)/2 entries. Used to convert the global
 * indices in idx3 into offsets within a single row.
 */
#define CTE_ROWOFF(nm, m) (((m) <= 0) ? 0 : (nm)[(m) - 1])
#define CTE_ROWLEN(nm, m) ((nm)[m] - CTE_ROWOFF(nm, m))
#define CTE_WARP 32

#include "gpu/gpu_charge_tree_cuda.inc"
