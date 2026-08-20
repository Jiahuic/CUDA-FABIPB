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
