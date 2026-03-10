#include "gpu_backend.h"
#include "gk.h"
#include "gkGlobal.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

extern "C" double *panelIA0(panel *pnlX, panel *pnlY);
extern "C" void kernelKER4(double *x, double *y);
extern "C" void (*kernel)(double *x, double *y);

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

int buildNearfieldTables(const ssystem *sys) {
  int pairIdx;
  long long totalInteractions = 0;
  long long k = 0;

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
  if (!allocateHostArrays(totalInteractions)) {
    return 0;
  }

  memcpy(gNear.h_leafPanelStart, sys->leafPanelStart, (size_t)sys->nLeafCubesFlat * sizeof(int));
  memcpy(gNear.h_leafPanelCount, sys->leafPanelCount, (size_t)sys->nLeafCubesFlat * sizeof(int));
  for (pairIdx = 0; pairIdx < sys->nNearPairsFlat; pairIdx++) {
    int dstLeaf = sys->nearPairDst[pairIdx];
    gNear.h_leafPairOffset[dstLeaf + 1] += 1;
  }
  for (pairIdx = 0; pairIdx < sys->nLeafCubesFlat; pairIdx++) {
    gNear.h_leafPairOffset[pairIdx + 1] += gNear.h_leafPairOffset[pairIdx];
  }

  kernel = kernelKER4;
  for (pairIdx = 0; pairIdx < sys->nNearPairsFlat; pairIdx++) {
    int srcLeaf = sys->nearPairSrc[pairIdx];
    int dstLeaf = sys->nearPairDst[pairIdx];
    int srcStart = sys->leafPanelStart[srcLeaf];
    int srcCount = sys->leafPanelCount[srcLeaf];
    int dstStart = sys->leafPanelStart[dstLeaf];
    int dstCount = sys->leafPanelCount[dstLeaf];
    int i, j;

    gNear.h_pairSrcCount[pairIdx] = srcCount;
    gNear.h_pairInteractionOffset[pairIdx] = k;

    for (i = 0; i < dstCount; i++) {
      int dstPanelIdx = dstStart + i;
      panel *pnlX = sys->panelByIdx[dstPanelIdx];
      for (j = 0; j < srcCount; j++) {
        int srcPanelIdx = srcStart + j;
        panel *pnlY = sys->panelByIdx[srcPanelIdx];
        double *KER = panelIA0(pnlX, pnlY);

        gNear.h_dst[k] = dstPanelIdx;
        gNear.h_src[k] = srcPanelIdx;
        gNear.h_k0[k] = KER[0];
        gNear.h_k1[k] = KER[1];
        gNear.h_k2[k] = KER[2];
        gNear.h_k3[k] = KER[3];
        k++;
      }
    }
  }

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

  gNear.nPnls = sys->nPnls;
  gNear.nInteractions = totalInteractions;
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
  printf("Direct GPU memory estimate: solver-host=%.3f GB host-coeff=%.3f GB device-total=%.3f GB combined=%.3f GB\n",
         (double)memcount / (1024.0 * 1024.0 * 1024.0),
         (double)hostCoeffBytes / (1024.0 * 1024.0 * 1024.0),
         (double)totalBytes / (1024.0 * 1024.0 * 1024.0),
         (double)combinedBytes / (1024.0 * 1024.0 * 1024.0));
  if (cudaMemGetInfo(&freeBytes, &totalGpuBytes) == cudaSuccess) {
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
  printf("GPU direct cache: panel-pairs=%lld\n", nInteractions);
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
  if (printedMode != sys->gpuNearfieldMode) {
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
