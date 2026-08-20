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

static void freeNearfieldStreamPairTable(void) {
  cudaFree(gNear.d_streamPairOffset);
  cudaFree(gNear.d_streamPairSrcStart);
  cudaFree(gNear.d_streamPairSrcCount);
  cudaFree(gNear.d_streamPairDstStart);
  gNear.d_streamPairOffset = NULL;
  gNear.d_streamPairSrcStart = NULL;
  gNear.d_streamPairSrcCount = NULL;
  gNear.d_streamPairDstStart = NULL;
  gNear.streamPairCount = 0;
}

/*
 * Ceiling on the special cache, in bytes. Defaults to 64 GiB, which covers the
 * 51.9 GiB H1N1 sdens=1 special-case cache while leaving room on current
 * capsid workstations. Set
 * FABIPB_GPU_NEARFIELD_SPECIAL_CACHE_MIB=0 to disable caching entirely and go
 * back to reclassifying every matvec.
 */
static size_t nearfieldSpecialCacheBudget(void) {
  const char *env = getenv("FABIPB_GPU_NEARFIELD_SPECIAL_CACHE_MIB");
  double mib = 64.0 * 1024.0;

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
  freeNearfieldStreamPairTable();
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
