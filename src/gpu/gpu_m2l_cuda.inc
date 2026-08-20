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
      !gM2L.h_lec3 || !gM2L.h_lec4) {
    setM2LLastError("host allocation for M2L streaming cache failed");
    return 0;
  }

#define M2L_STREAM_CUDA_ALLOC(ptr, count, type) \
  do { if (!cudaMallocM2L((void **)&(ptr), (count) * sizeof(type), #ptr)) return 0; } while (0)
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
  do { if (!cudaMemcpyM2L((dst), (src), (count) * sizeof(type), cudaMemcpyHostToDevice, #dst)) return 0; } while (0)
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

int flushM2LStreamingChunk(int pairCount, int groupCount, int coeffCount) {
  cudaError_t err;
  const int blockSize = 128;
  if (pairCount <= 0 || groupCount <= 0 || coeffCount <= 0) return 1;

#define M2L_CHUNK_UPLOAD(dst, src, count, type) \
  do { \
    if (!cudaMemcpyM2L((dst), (src), (size_t)(count) * sizeof(type), cudaMemcpyHostToDevice, #dst)) return 0; \
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
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    setM2LLastError("streaming m2lGroupedKernel launch failed: %s",
                    cudaGetErrorString(err));
    return 0;
  }
  return 1;
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
      setM2LLastError("streaming group exceeds chunk capacity: group=%d pairs=%d coeff=%lld pair-capacity=%d coeff-capacity=%d",
                      groupIdx, count, neededCoeff,
                      gM2L.streamPairCapacity, gM2L.streamCoeffCapacity);
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
  {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      setM2LLastError("streaming M2L kernels failed: %s", cudaGetErrorString(err));
      return 0;
    }
  }
  if (sys->benchmarkMode > 0) {
    printf("GPU M2L streaming apply: chunks=%d pairs=%d coeff=%lld\n",
           chunks, gM2L.nPairs, gM2L.totalPairCoeff);
  }
  return 1;
}
