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
