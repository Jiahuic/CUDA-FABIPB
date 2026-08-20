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
/*
 * Batched preconditioner block build.
 *
 * gpuBuildPrecondDisjointBlock below does one cube at a time: it takes a global
 * mutex, uploads that cube's panel geometry and index lists, launches, calls
 * cudaDeviceSynchronize, and copies four result arrays back -- about ten
 * synchronous CUDA calls per block. With 11,257 blocks on 7A6A that is ~112k
 * synchronous calls, and the mutex serialises them across all 72 setup threads,
 * which is why the GPU path beat the pure-CPU one by only 1.43x (1.296 s
 * against 1.851 s) on work a GPU should finish in milliseconds.
 *
 * The pairs themselves are not new work either: every pair inside a level
 * depth-1 cube is also a nearfield pair, since children of one parent are
 * always neighbours, and the nearfield computes the same panelIA0 with the same
 * four KER components. This entry point shares that machinery -- the same
 * nearfieldDisjointQ1BuildKernel, one resident copy of the panel geometry
 * indexed by pnl->idx, and chunking sized like the nearfield's -- so the whole
 * preconditioner is built in a handful of launches instead of one per block.
 *
 * Geometry is cached across calls and rebuilt only when the system changes.
 */
struct PrecondBatchCache {
  const ssystem *sys;
  int nPnls;
  NearPanelGeom *d_geom;
  int *d_src;
  int *d_dst;
  double *d_k[4];
  long long capacity;
};
PrecondBatchCache gPcBatch = {NULL, 0, NULL, NULL, NULL, {NULL, NULL, NULL, NULL}, 0};

void freePrecondBatch() {
  int i;
  cudaFree(gPcBatch.d_geom);
  cudaFree(gPcBatch.d_src);
  cudaFree(gPcBatch.d_dst);
  for (i = 0; i < 4; i++) cudaFree(gPcBatch.d_k[i]);
  gPcBatch.d_geom = NULL;
  gPcBatch.d_src = NULL;
  gPcBatch.d_dst = NULL;
  for (i = 0; i < 4; i++) gPcBatch.d_k[i] = NULL;
  gPcBatch.sys = NULL;
  gPcBatch.nPnls = 0;
  gPcBatch.capacity = 0;
}

long long precondBatchCapacity() {
  const char *env = getenv("FABIPB_GPU_PRECOND_CHUNK_MIB");
  double mib = 512.0;
  long long perPair = (long long)(2 * sizeof(int) + 4 * sizeof(double));
  long long cap;

  if (env != NULL && env[0] != '\0') {
    char *end = NULL;
    double v = strtod(env, &end);
    if (end != env && *end == '\0' && v > 0.0) mib = v;
  }
  cap = (long long)((mib * 1024.0 * 1024.0) / (double)perPair);
  if (cap < 65536) cap = 65536;
  return cap;
}
extern "C"
int gpuBuildPrecondPairsBatched(struct ssystem *sys,
                                const int *pairSrc, const int *pairDst,
                                long long nPairs,
                                double *k0, double *k1, double *k2, double *k3) {
  std::lock_guard<std::mutex> lock(gPrecondMutex);
  long long done;
  long long cap;

  if (sys == NULL || pairSrc == NULL || pairDst == NULL || nPairs <= 0) return 0;
  if (k0 == NULL || k1 == NULL || k2 == NULL || k3 == NULL) return 0;
  if (!gpuBackendAvailable()) return 0;

  cap = precondBatchCapacity();
  if (cap > nPairs) cap = nPairs;

  if (gPcBatch.sys != sys || gPcBatch.nPnls != sys->nPnls || gPcBatch.capacity < cap) {
    panel *pnl;
    std::vector<NearPanelGeom> geom;
    int i;

    freePrecondBatch();
    geom.resize((size_t)sys->nPnls);
    /* Indexed by pnl->idx so no reverse lookup is needed anywhere. */
    for (pnl = sys->pnlLst; pnl != NULL; pnl = pnl->nextC) {
      NearPanelGeom *g = &geom[(size_t)pnl->idx];
      int a, b;
      for (a = 0; a < 3; a++) {
        for (b = 0; b < 3; b++) g->vtx[a][b] = pnl->vtx[a][b];
        g->a0[a] = pnl->a[0][a];
        g->a1[a] = pnl->a[1][a];
        g->a2[a] = pnl->a[2][a];
        g->normal[a] = pnl->normal[a];
      }
      g->area = pnl->area;
    }
    if (cudaMalloc((void **)&gPcBatch.d_geom,
                   (size_t)sys->nPnls * sizeof(NearPanelGeom)) != cudaSuccess ||
        cudaMalloc((void **)&gPcBatch.d_src, (size_t)cap * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void **)&gPcBatch.d_dst, (size_t)cap * sizeof(int)) != cudaSuccess) {
      freePrecondBatch();
      return 0;
    }
    for (i = 0; i < 4; i++) {
      if (cudaMalloc((void **)&gPcBatch.d_k[i], (size_t)cap * sizeof(double)) != cudaSuccess) {
        freePrecondBatch();
        return 0;
      }
    }
    if (cudaMemcpy(gPcBatch.d_geom, &geom[0],
                   (size_t)sys->nPnls * sizeof(NearPanelGeom),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      freePrecondBatch();
      return 0;
    }
    gPcBatch.sys = sys;
    gPcBatch.nPnls = sys->nPnls;
    gPcBatch.capacity = cap;
  }

  if (cudaMemcpyToSymbol(c_nearKappa, &kappa, sizeof(double)) != cudaSuccess) return 0;
  if (cudaMemcpyToSymbol(c_nearEpsilon, &epsilon, sizeof(double)) != cudaSuccess) return 0;

  for (done = 0; done < nPairs; ) {
    long long n = nPairs - done;
    int blockSize = 256, gridSize;

    if (n > gPcBatch.capacity) n = gPcBatch.capacity;
    if (cudaMemcpy(gPcBatch.d_src, pairSrc + done, (size_t)n * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    if (cudaMemcpy(gPcBatch.d_dst, pairDst + done, (size_t)n * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 0;

    gridSize = (int)((n + blockSize - 1) / blockSize);
    nearfieldDisjointQ1BuildKernel<<<gridSize, blockSize>>>(
        n, gPcBatch.d_geom, gPcBatch.d_src, gPcBatch.d_dst,
        gPcBatch.d_k[0], gPcBatch.d_k[1], gPcBatch.d_k[2], gPcBatch.d_k[3]);
    if (cudaGetLastError() != cudaSuccess) return 0;
    if (cudaDeviceSynchronize() != cudaSuccess) return 0;

    if (cudaMemcpy(k0 + done, gPcBatch.d_k[0], (size_t)n * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    if (cudaMemcpy(k1 + done, gPcBatch.d_k[1], (size_t)n * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    if (cudaMemcpy(k2 + done, gPcBatch.d_k[2], (size_t)n * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    if (cudaMemcpy(k3 + done, gPcBatch.d_k[3], (size_t)n * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    done += n;
  }
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
