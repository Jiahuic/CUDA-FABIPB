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
                             (size_t)nPairs * sizeof(int), "stream pairDstStart") ||
        !cudaMemcpyNearfield(gNear.d_streamPairOffset, &hOffset[0],
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
      if (sys->benchmarkMode > 0) {
        printf("GPU nearfield streaming pair table disabled: %s\n",
               gpuNearfieldLastError());
      }
      freeNearfieldStreamPairTable();
      cudaGetLastError();
      setNearfieldLastError("not attempted");
    } else {
      gNear.streamPairCount = nPairs;
    }
  }

  if (sys->benchmarkMode > 0) {
    printf("GPU nearfield streaming: interactions=%lld chunk-capacity=%lld chunk-device=%.3f GiB qOrd=1\n",
           totalInteractions, capacity,
           bytesToGiB(ni * (2U * sizeof(int) + 4U * sizeof(double))));
    if (gNear.streamPairCount > 0) {
      printf("GPU nearfield streaming pair table: pairs=%d device=%.3f GiB\n",
             gNear.streamPairCount,
             bytesToGiB(((size_t)gNear.streamPairCount + 1U) * sizeof(long long) +
                        3U * (size_t)gNear.streamPairCount * sizeof(int)));
    } else {
      printf("GPU nearfield streaming pair table: disabled\n");
    }
  }
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
