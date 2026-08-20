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
