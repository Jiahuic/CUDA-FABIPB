__global__ void rhsReuseCoulombDotKernel(int nPnls,
                                         double eps1,
                                         const double *rhs,
                                         const double *sgm,
                                         double *partial) {
  extern __shared__ double shmem[];
  int tid = threadIdx.x;
  long long stride = (long long)blockDim.x * gridDim.x;
  long long i = (long long)blockIdx.x * blockDim.x + tid;
  double sum = 0.0;

  for (; i < nPnls; i += stride) {
    sum += eps1 * (rhs[i] * sgm[i + nPnls] -
                   rhs[i + nPnls] * sgm[i]);
  }

  shmem[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shmem[tid] += shmem[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x] = shmem[0];
  }
}

int gpuPanelChargeTreeScreenedEnergy(ssystem *sys, const double *sgm, double *pot);

extern "C" int gpuRhsReuseCoulombDot(ssystem *sys, const double *rhs,
                                      const double *sgm, double *dot) {
  const int blockSize = 256;
  int nPnls;
  int gridSize;
  size_t vecBytes;
  double *d_rhs = NULL;
  double *d_sgm = NULL;
  double *d_partial = NULL;
  std::vector<double> partial;
  cudaError_t err;
  double t0, t1, copyH2DTime, kernelTime, copyD2HTime;
  double local = 0.0;
  int i;

  if (sys == NULL || rhs == NULL || sgm == NULL || dot == NULL) {
    return 0;
  }
  nPnls = sys->nPnls;
  if (nPnls <= 0) {
    *dot = 0.0;
    return 1;
  }

  gridSize = (nPnls + blockSize - 1) / blockSize;
  if (gridSize > 65535) {
    gridSize = 65535;
  }
  vecBytes = (size_t)(2 * nPnls) * sizeof(double);
  partial.resize((size_t)gridSize);

  t0 = wall_seconds_cuda_local();
  err = cudaMalloc((void **)&d_rhs, vecBytes);
  if (err != cudaSuccess) {
    setNearfieldLastError("rhs-reuse dot cudaMalloc d_rhs failed for %.3f GiB: %s",
                          bytesToGiB(vecBytes), cudaGetErrorString(err));
    goto fail;
  }
  err = cudaMalloc((void **)&d_sgm, vecBytes);
  if (err != cudaSuccess) {
    setNearfieldLastError("rhs-reuse dot cudaMalloc d_sgm failed for %.3f GiB: %s",
                          bytesToGiB(vecBytes), cudaGetErrorString(err));
    goto fail;
  }
  err = cudaMalloc((void **)&d_partial, (size_t)gridSize * sizeof(double));
  if (err != cudaSuccess) {
    setNearfieldLastError("rhs-reuse dot cudaMalloc partial failed: %s",
                          cudaGetErrorString(err));
    goto fail;
  }
  err = cudaMemcpy(d_rhs, rhs, vecBytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    setNearfieldLastError("rhs-reuse dot H2D rhs failed for %.3f GiB: %s",
                          bytesToGiB(vecBytes), cudaGetErrorString(err));
    goto fail;
  }
  err = cudaMemcpy(d_sgm, sgm, vecBytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    setNearfieldLastError("rhs-reuse dot H2D sgm failed for %.3f GiB: %s",
                          bytesToGiB(vecBytes), cudaGetErrorString(err));
    goto fail;
  }
  t1 = wall_seconds_cuda_local();
  copyH2DTime = t1 - t0;

  t0 = wall_seconds_cuda_local();
  rhsReuseCoulombDotKernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(
      nPnls, epsilon1, d_rhs, d_sgm, d_partial);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    setNearfieldLastError("rhs-reuse dot kernel launch failed: %s",
                          cudaGetErrorString(err));
    goto fail;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    setNearfieldLastError("rhs-reuse dot kernel failed: %s",
                          cudaGetErrorString(err));
    goto fail;
  }
  t1 = wall_seconds_cuda_local();
  kernelTime = t1 - t0;

  t0 = wall_seconds_cuda_local();
  err = cudaMemcpy(partial.data(), d_partial, (size_t)gridSize * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    setNearfieldLastError("rhs-reuse dot D2H partial failed: %s",
                          cudaGetErrorString(err));
    goto fail;
  }
  for (i = 0; i < gridSize; i++) {
    local += partial[(size_t)i];
  }
  t1 = wall_seconds_cuda_local();
  copyD2HTime = t1 - t0;

  cudaFree(d_partial);
  cudaFree(d_sgm);
  cudaFree(d_rhs);
  *dot = local;

  if (sys->benchmarkMode > 0) {
    printf("GPU rhs-reuse Coulomb dot: panels=%d h2d=%.6f kernel=%.6f d2h+reduce=%.6f\n",
           nPnls, copyH2DTime, kernelTime, copyD2HTime);
  }
  return 1;

fail:
  cudaFree(d_partial);
  cudaFree(d_sgm);
  cudaFree(d_rhs);
  return 0;
}

extern "C" int gpuRhsReusePanelChargeTreeEnergy(ssystem *sys, const double *rhs,
                                                const double *sgm, double *pot) {
  double t0, t1, dotTime, screenedTime;
  double coulombDot = 0.0;
  double screenedTree = 0.0;

  if (sys == NULL || rhs == NULL || sgm == NULL || pot == NULL) return 0;
  if (!gpuBackendAvailable()) return 0;
  if (sys->maxQuadOrder != 1) return 0;
  if (sys->chgCubeList == NULL || sys->chgCubeList[0] == NULL) return 0;

  t0 = wall_seconds_cuda_local();
  if (!gpuRhsReuseCoulombDot(sys, rhs, sgm, &coulombDot)) {
    return 0;
  }
  t1 = wall_seconds_cuda_local();
  dotTime = t1 - t0;

  t0 = wall_seconds_cuda_local();
  if (!gpuPanelChargeTreeScreenedEnergy(sys, sgm, &screenedTree)) {
    return 0;
  }
  t1 = wall_seconds_cuda_local();
  screenedTime = t1 - t0;

  *pot = coulombDot + screenedTree;
  if (sys->benchmarkMode > 0) {
    printf("GPU rhs-reuse energy breakdown: coulomb-dot=%.17g screened-tree=%.17g total=%.17g dot-time=%.6f screened-time=%.6f\n",
           coulombDot, screenedTree, *pot, dotTime, screenedTime);
  }
  return 1;
}
