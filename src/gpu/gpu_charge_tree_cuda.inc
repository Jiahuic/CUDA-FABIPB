struct ChargeTreeGpu {
  const ssystem *sys;
  int nNodes;
  int nCharges;
  int nPanels;
  int derivOrderMax;        /* largest order+1 used by any accepted node */
  int idxDim;               /* idx3 flattened as idxDim^3 */
  int nMomMax;              /* nMom[derivOrderMax] */
  /* node arrays */
  double *d_nodeCtr;        /* 3 per node */
  double *d_nodeRad;
  int *d_nodeLevel;
  int *d_nodeKidStart;
  int *d_nodeKidCount;
  int *d_nodeChgStart;
  int *d_nodeChgCount;
  int *d_nodeMomOff;
  int *d_nodeOrder;         /* rhsChargeExpansionOrder for this node's level */
  int *d_kidIdx;
  int *d_chgIdx;
  double *d_moments;
  /* charges and panels */
  double *d_pos;
  double *d_chr;
  NearPanelGeom *d_panels;
  int *d_panelIdx;
  /* small tables */
  int *d_idx3;
  int *d_sgn3;
  int *d_nMom;
  /* results */
  double *d_out;            /* 2 per panel: y0, y1 */
};

static ChargeTreeGpu gCTE = {};

static void freeChargeTreeGpu(void) {
  cudaFree(gCTE.d_nodeCtr);      cudaFree(gCTE.d_nodeRad);
  cudaFree(gCTE.d_nodeLevel);    cudaFree(gCTE.d_nodeKidStart);
  cudaFree(gCTE.d_nodeKidCount); cudaFree(gCTE.d_nodeChgStart);
  cudaFree(gCTE.d_nodeChgCount); cudaFree(gCTE.d_nodeMomOff);
  cudaFree(gCTE.d_nodeOrder);    cudaFree(gCTE.d_kidIdx);
  cudaFree(gCTE.d_chgIdx);       cudaFree(gCTE.d_moments);
  cudaFree(gCTE.d_pos);          cudaFree(gCTE.d_chr);
  cudaFree(gCTE.d_panels);       cudaFree(gCTE.d_panelIdx);
  cudaFree(gCTE.d_idx3);
  cudaFree(gCTE.d_sgn3);         cudaFree(gCTE.d_nMom);
  cudaFree(gCTE.d_out);
  memset(&gCTE, 0, sizeof(gCTE));
}

extern "C" void gpuReleaseChargeTreeCache(void) {
  freeChargeTreeGpu();
}

/* Depth-first flatten of the charge tree into contiguous arrays. */
static int buildChargeTreeGpu(const ssystem *sys) {
  std::vector<const cube *> order;
  std::vector<int> nodeOf;           /* scratch: index of each visited node */
  std::vector<double> ctr, rad, mom;
  std::vector<int> lev, kidStart, kidCount, chgStart, chgCount, momOff, nodeOrd;
  std::vector<int> kidIdx, chgIdx;
  std::vector<const cube *> stack;
  size_t i;
  int derivMax = 0;

  if (sys->chgCubeList == NULL || sys->chgCubeList[0] == NULL) {
    setNearfieldLastError("charge tree not built");
    return 0;
  }

  /* First pass: assign an index to every node, breadth-first from the root. */
  order.push_back(sys->chgCubeList[0]);
  for (i = 0; i < order.size(); i++) {
    const cube *cb = order[i];
    int k;
    for (k = 0; k < cb->nKids; k++) {
      order.push_back(cb->kids[k]);
    }
  }

  /* Map each node pointer to its index so kid lists can be flattened. */
  {
    std::vector<std::pair<const cube *, int> > lookup;
    lookup.reserve(order.size());
    for (i = 0; i < order.size(); i++) lookup.push_back(std::make_pair(order[i], (int)i));
    std::sort(lookup.begin(), lookup.end());

    for (i = 0; i < order.size(); i++) {
      const cube *cb = order[i];
      int ordHere = rhsChargeExpansionOrder((ssystem *)sys, cb->level);
      int nMomHere = sys->nMom[ordHere];
      int k;

      ctr.push_back(cb->x[0]); ctr.push_back(cb->x[1]); ctr.push_back(cb->x[2]);
      rad.push_back(cb->eRad);
      lev.push_back(cb->level);
      nodeOrd.push_back(ordHere);
      if (ordHere + 1 > derivMax) derivMax = ordHere + 1;

      kidStart.push_back((int)kidIdx.size());
      kidCount.push_back(cb->nKids);
      for (k = 0; k < cb->nKids; k++) {
        std::pair<const cube *, int> probe(cb->kids[k], 0);
        std::vector<std::pair<const cube *, int> >::iterator it =
            std::lower_bound(lookup.begin(), lookup.end(), probe);
        if (it == lookup.end() || it->first != cb->kids[k]) {
          setNearfieldLastError("charge tree flatten: child not found");
          return 0;
        }
        kidIdx.push_back(it->second);
      }

      chgStart.push_back((int)chgIdx.size());
      if (cb->level == sys->chgDepth && cb->chgIdx != NULL) {
        chgCount.push_back(cb->nChgs);
        for (k = 0; k < cb->nChgs; k++) chgIdx.push_back(cb->chgIdx[k]);
      } else {
        chgCount.push_back(0);
      }

      momOff.push_back((int)mom.size());
      if (cb->mom_chr != NULL) {
        for (k = 0; k < nMomHere; k++) mom.push_back(cb->mom_chr[k]);
      } else {
        for (k = 0; k < nMomHere; k++) mom.push_back(0.0);
      }
    }
  }

  gCTE.nNodes = (int)order.size();
  gCTE.nCharges = sys->nChar;
  gCTE.nPanels = sys->nPnls;
  gCTE.derivOrderMax = derivMax;
  gCTE.nMomMax = sys->nMom[derivMax];
  /* idx3 was allocated by mkIndex(maxOrder+1), so its valid extent is
   * maxOrder+2 per dimension. The kernel reads it at i1+1 with
   * i1+i2+i3 <= order <= maxOrder, so maxOrder+1 is the largest index needed
   * and that extent is exactly right. Deriving the dimension from derivMax
   * instead would read past the table. */
  gCTE.idxDim = sys->maxOrder + 2;

  /* Flatten idx3 and sgn3 for the device. */
  {
    int D = gCTE.idxDim;
    std::vector<int> fidx((size_t)D * D * D, 0);
    std::vector<int> fsgn((size_t)gCTE.nMomMax, 0);
    int a, b, c;
    for (a = 0; a < D; a++)
      for (b = 0; b < D; b++)
        for (c = 0; c < D; c++)
          if (a + b + c <= sys->maxOrder + 1) fidx[((size_t)a * D + b) * D + c] = idx3[a][b][c];
    for (a = 0; a < gCTE.nMomMax; a++) fsgn[a] = sgn3[a];

    std::vector<int> fnmom((size_t)derivMax + 2, 0);
    for (a = 0; a <= derivMax + 1; a++) fnmom[a] = sys->nMom[a];

    if (!cudaMallocNearfield((void **)&gCTE.d_idx3, fidx.size() * sizeof(int), "cte idx3") ||
        !cudaMallocNearfield((void **)&gCTE.d_sgn3, fsgn.size() * sizeof(int), "cte sgn3") ||
        !cudaMallocNearfield((void **)&gCTE.d_nMom, fnmom.size() * sizeof(int), "cte nMom") ||
        !cudaMemcpyNearfield(gCTE.d_idx3, &fidx[0], fidx.size() * sizeof(int),
                             cudaMemcpyHostToDevice, "cte idx3") ||
        !cudaMemcpyNearfield(gCTE.d_sgn3, &fsgn[0], fsgn.size() * sizeof(int),
                             cudaMemcpyHostToDevice, "cte sgn3") ||
        !cudaMemcpyNearfield(gCTE.d_nMom, &fnmom[0], fnmom.size() * sizeof(int),
                             cudaMemcpyHostToDevice, "cte nMom")) return 0;
  }

#define CTE_UP(dst, vec, type, tag)                                              \
  do {                                                                           \
    if (!cudaMallocNearfield((void **)&(dst), (vec).size() * sizeof(type), tag) ||\
        !cudaMemcpyNearfield((dst), &(vec)[0], (vec).size() * sizeof(type),       \
                             cudaMemcpyHostToDevice, tag)) return 0;              \
  } while (0)

  CTE_UP(gCTE.d_nodeCtr, ctr, double, "cte ctr");
  CTE_UP(gCTE.d_nodeRad, rad, double, "cte rad");
  CTE_UP(gCTE.d_nodeLevel, lev, int, "cte lev");
  CTE_UP(gCTE.d_nodeKidStart, kidStart, int, "cte kidStart");
  CTE_UP(gCTE.d_nodeKidCount, kidCount, int, "cte kidCount");
  CTE_UP(gCTE.d_nodeChgStart, chgStart, int, "cte chgStart");
  CTE_UP(gCTE.d_nodeChgCount, chgCount, int, "cte chgCount");
  CTE_UP(gCTE.d_nodeMomOff, momOff, int, "cte momOff");
  CTE_UP(gCTE.d_nodeOrder, nodeOrd, int, "cte order");
  CTE_UP(gCTE.d_moments, mom, double, "cte moments");
  if (!kidIdx.empty()) CTE_UP(gCTE.d_kidIdx, kidIdx, int, "cte kidIdx");
  if (!chgIdx.empty()) CTE_UP(gCTE.d_chgIdx, chgIdx, int, "cte chgIdx");
#undef CTE_UP

  /* charges */
  if (!cudaMallocNearfield((void **)&gCTE.d_pos, (size_t)3 * sys->nChar * sizeof(double), "cte pos") ||
      !cudaMallocNearfield((void **)&gCTE.d_chr, (size_t)sys->nChar * sizeof(double), "cte chr") ||
      !cudaMemcpyNearfield(gCTE.d_pos, sys->pos, (size_t)3 * sys->nChar * sizeof(double),
                           cudaMemcpyHostToDevice, "cte pos") ||
      !cudaMemcpyNearfield(gCTE.d_chr, sys->chr, (size_t)sys->nChar * sizeof(double),
                           cudaMemcpyHostToDevice, "cte chr")) return 0;

  /* panel geometry: same layout the nearfield already uses */
  {
    std::vector<NearPanelGeom> geom((size_t)sys->nPnls);
    std::vector<int> panelIdx((size_t)sys->nPnls);
    panel *pnl;
    int idx, k;
    for (idx = 0, pnl = sys->pnlLst; pnl != NULL && idx < sys->nPnls; pnl = pnl->nextC, idx++) {
      for (k = 0; k < 3; k++) {
        geom[(size_t)idx].vtx[0][k] = pnl->vtx[0][k];
        geom[(size_t)idx].a0[k] = pnl->a[0][k];
        geom[(size_t)idx].a2[k] = pnl->a[2][k];
        geom[(size_t)idx].normal[k] = pnl->normal[k];
      }
      geom[(size_t)idx].area = pnl->area;
      panelIdx[(size_t)idx] = pnl->idx;
    }
    if (!cudaMallocNearfield((void **)&gCTE.d_panels,
                             (size_t)sys->nPnls * sizeof(NearPanelGeom), "cte panels") ||
        !cudaMemcpyNearfield(gCTE.d_panels, &geom[0],
                             (size_t)sys->nPnls * sizeof(NearPanelGeom),
                             cudaMemcpyHostToDevice, "cte panels") ||
        !cudaMallocNearfield((void **)&gCTE.d_panelIdx,
                             (size_t)sys->nPnls * sizeof(int), "cte panelIdx") ||
        !cudaMemcpyNearfield(gCTE.d_panelIdx, &panelIdx[0],
                             (size_t)sys->nPnls * sizeof(int),
                             cudaMemcpyHostToDevice, "cte panelIdx")) return 0;
  }

  if (!cudaMallocNearfield((void **)&gCTE.d_out,
                           (size_t)2 * sys->nPnls * sizeof(double), "cte out")) return 0;

  gCTE.sys = sys;
  if (sys->benchmarkMode > 0) {
    printf("GPU charge-tree energy: nodes=%d charges=%d panels=%d derivMax=%d "
           "moments=%.3f GiB\n",
           gCTE.nNodes, gCTE.nCharges, gCTE.nPanels, gCTE.derivOrderMax,
           bytesToGiB(mom.size() * sizeof(double)));
  }
  return 1;
}

/* Radial derivatives of G0 and Gk; port of kernelDG0DGk() in kernel.c. */
__device__ __forceinline__ void cteRadialDerivs(double r, int p, double kap,
                                                double fourPiIv,
                                                double *G0, double *Gk) {
  double kappa2 = kap * kap;
  double rexp = exp(-kap * r);
  double r1 = 1.0 / r;
  double r2 = r1 * r1;
  double nr2 = -r2;
  int k;

  G0[0] = r1 * fourPiIv;
  Gk[0] = rexp * G0[0];
  G0[1] = nr2 * G0[0];
  Gk[1] = -Gk[0] * (r2 + kap * r1);
  for (k = 1; k < p; k++) {
    G0[k + 1] = (2 * k + 1) * nr2 * G0[k];
    Gk[k + 1] = r2 * (kappa2 * Gk[k - 1] - (2 * k + 1) * Gk[k]);
  }
}

/*
 * Cartesian derivative tensors of G0/Gk about a cluster centre; port of
 * setupScreenedDerivsLocal() in treecode.c. The recurrence is sequential in
 * order p: each order is built from p+1, and the per-level work is too small
 * to split usefully across CUDA lanes.
 */
__device__ void cteSetupDerivsStrided(int order, const double *x,
                               double kap, double fourPiIv,
                               const int *fidx3, int D,
                               double *gv0, double *gvk,
                               double **dg0, double **dgk, long long S) {
  int p, p1, iRow, iRow1, i1, i2, i3, idx, idx1, idx2;
  double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);

  cteRadialDerivs(r, order, kap, fourPiIv, gv0, gvk);
  for (p = 0; p <= order; p++) { dg0[p][(long long)(0) * S] = gv0[p]; dgk[p][(long long)(0) * S] = gvk[p]; }
  for (p = 0; p < order; p++) {
    p1 = p + 1;
    dg0[p][(long long)(1) * S] = dg0[p1][(long long)(0) * S] * x[2];  dg0[p][(long long)(2) * S] = dg0[p1][(long long)(0) * S] * x[1];  dg0[p][(long long)(3) * S] = dg0[p1][(long long)(0) * S] * x[0];
    dgk[p][(long long)(1) * S] = dgk[p1][(long long)(0) * S] * x[2];  dgk[p][(long long)(2) * S] = dgk[p1][(long long)(0) * S] * x[1];  dgk[p][(long long)(3) * S] = dgk[p1][(long long)(0) * S] * x[0];
  }
#define CTE_IDX(a,b,c) fidx3[(((a) * D) + (b)) * D + (c)]
  for (iRow = 2; iRow <= order; iRow++) {
    for (p = 0; p <= order - iRow; p++) {
      p1 = p + 1;
      idx = CTE_IDX(0, 0, iRow);
      iRow1 = iRow - 1;
      idx1 = CTE_IDX(0, 0, iRow1);
      idx2 = CTE_IDX(0, 0, iRow - 2);
      dg0[p][(long long)(idx) * S] = dg0[p1][(long long)(idx1) * S] * x[2] + dg0[p1][(long long)(idx2) * S] * iRow1;
      dgk[p][(long long)(idx) * S] = dgk[p1][(long long)(idx1) * S] * x[2] + dgk[p1][(long long)(idx2) * S] * iRow1;
      idx++;
      idx1 = CTE_IDX(0, 0, iRow1);
      dg0[p][(long long)(idx) * S] = dg0[p1][(long long)(idx1) * S] * x[1];
      dgk[p][(long long)(idx) * S] = dgk[p1][(long long)(idx1) * S] * x[1];
      idx++;
      for (i2 = 2; i2 <= iRow; i2++, idx++) {
        i3 = iRow - i2;
        idx1 = CTE_IDX(0, i2 - 1, i3);
        idx2 = CTE_IDX(0, i2 - 2, i3);
        dg0[p][(long long)(idx) * S] = dg0[p1][(long long)(idx1) * S] * x[1] + dg0[p1][(long long)(idx2) * S] * (i2 - 1);
        dgk[p][(long long)(idx) * S] = dgk[p1][(long long)(idx1) * S] * x[1] + dgk[p1][(long long)(idx2) * S] * (i2 - 1);
      }
      for (i2 = 0; i2 <= iRow1; i2++, idx++) {
        i3 = iRow1 - i2;
        idx1 = CTE_IDX(0, i2, i3);
        dg0[p][(long long)(idx) * S] = dg0[p1][(long long)(idx1) * S] * x[0];
        dgk[p][(long long)(idx) * S] = dgk[p1][(long long)(idx1) * S] * x[0];
      }
      for (i1 = 2; i1 <= iRow; i1++) {
        for (i2 = 0; i2 <= iRow - i1; i2++, idx++) {
          i3 = iRow - i1 - i2;
          idx1 = CTE_IDX(i1 - 1, i2, i3);
          idx2 = CTE_IDX(i1 - 2, i2, i3);
          dg0[p][(long long)(idx) * S] = dg0[p1][(long long)(idx1) * S] * x[0] + dg0[p1][(long long)(idx2) * S] * (i1 - 1);
          dgk[p][(long long)(idx) * S] = dgk[p1][(long long)(idx1) * S] * x[0] + dgk[p1][(long long)(idx2) * S] * (i1 - 1);
        }
      }
    }
  }
#undef CTE_IDX
}

/*
 * One thread per panel, grid-stride.
 *
 * Each thread owns a panel and its own scratch slab in global memory. The slabs
 * are interleaved -- entry i of thread t lives at [i * stride + t] -- so lanes
 * touching the same tensor entry hit consecutive addresses. Concurrency is
 * capped and panels are taken in a grid-stride loop, which bounds scratch by the
 * number of resident threads instead of the panel count.
 *
 * Storage is a rolling window, not the whole pyramid. The recurrence builds row
 * iRow of level p from rows iRow-1 and iRow-2 of level p+1, and the loop already
 * runs iRow outermost, so only two rows per level ever need to be live. Level 0
 * is the only level the contraction reads, and term n of it needs rows n and
 * n+1, so the contraction is fused into the row loop and level 0 needs no more
 * storage than any other level.
 *
 * That takes the slab from sum_m nMom[m] = C(derivMax+4,4) down to
 * 2*nMom[derivMax] per array -- 715 -> 440 doubles at derivMax=9. It matters
 * because the kernel is footprint-bound: holding threads, arithmetic and
 * coalescing fixed and varying only the footprint gave 248.6 MiB -> 37.17 s
 * against 2.8 MiB -> 14.92 s on 7A6A, so roughly 2.5x is the ceiling for this
 * footprint reduction (see docs/6co8_gpu_fabipb_milestone.md).
 *
 * Row m lives in slot m&1, so writing row iRow reuses the slot holding row
 * iRow-2, which is dead by then. Levels are visited with p ascending, and level
 * p reads only level p+1, so level p's overwrite of its own iRow-2 slot always
 * happens after level p-1 has read it.
 *
 * Every arithmetic operation and its order are unchanged from the pyramid
 * version -- only where the values are stored -- so the output is bit-identical.
 */
__global__ void chargeTreeEnergyKernel(
    int nPanels, int derivMax, int idxDim, int chgDepth, int height,
    double theta, double kap, double eps, double fourPiIv,
    const NearPanelGeom *panels,
    const double *nodeCtr, const double *nodeRad, const int *nodeLevel,
    const int *nodeKidStart, const int *nodeKidCount,
    const int *nodeChgStart, const int *nodeChgCount,
    const int *nodeMomOff, const int *nodeOrder,
    const int *kidIdx, const int *chgIdx, const double *moments,
    const double *pos, const double *chr,
    const int *fidx3, const int *sgn3d, const int *levOff,
    const int *nMomT,
    const int *panelOrigIdx, const double *sgm,
    double *scratch, long long stride,
    double *partial) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long nThreadsTotal = (long long)gridDim.x * blockDim.x;
  /* Base of each level's two-row window; slot m&1 holds row m. */
  double *lev0[16];
  double *levk[16];
  int rowCap[16];            /* allocated row length at this level */
  double gv0[18], gvk[18];
  const long long halfOff = (long long)levOff[derivMax + 1];
  extern __shared__ double blockSum[];
  double threadSum = 0.0;
  int p;
  int panelIdx;

  if (tid >= stride) return;
  for (p = 0; p <= derivMax && p < 16; p++) {
    lev0[p] = scratch + (long long)levOff[p] * stride + tid;
    levk[p] = scratch + (halfOff + levOff[p]) * stride + tid;
    /* Level p never holds a row longer than row(derivMax - p). */
    rowCap[p] = CTE_ROWLEN(nMomT, derivMax - p);
  }

  for (panelIdx = (int)tid; panelIdx < nPanels; panelIdx += (int)nThreadsTotal) {
    const NearPanelGeom *pn = &panels[panelIdx];
    double qpt[3], nrm[3];
    double y0 = 0.0, y1 = 0.0;
    int stack[CTE_MAX_STACK];
    int sp = 0;
    int k;

    for (k = 0; k < 3; k++) {
      qpt[k] = pn->vtx[0][k] + 0.5 * (pn->a2[k] + 0.5 * pn->a0[k]);
      nrm[k] = pn->normal[k];
    }
    stack[sp++] = 0;

    while (sp > 0) {
      int node = stack[--sp];
      double r[3], dist;
      for (k = 0; k < 3; k++) r[k] = qpt[k] - nodeCtr[3 * node + k];
      dist = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

      if (nodeRad[node] < theta * dist && nodeLevel[node] >= height) {
        int ordHere = nodeOrder[node];
        const double *mom = &moments[nodeMomOff[node]];
        int i, i1, i2, n;

        int order = ordHere + 1;
        int iRow;
        double rr = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

#define CTE_IDX(a,b,c) fidx3[(((a) * idxDim) + (b)) * idxDim + (c)]
/* Address of local entry `l` of row `m` at level `q`, in each half. */
#define CTE_A0(q,m,l) (lev0[q] + (long long)(((m) & 1) * rowCap[q] + (l)) * stride)
#define CTE_AK(q,m,l) (levk[q] + (long long)(((m) & 1) * rowCap[q] + (l)) * stride)

        cteRadialDerivs(rr, order, kap, fourPiIv, gv0, gvk);

        /* Row 0 of every level is the radial derivative itself. */
        for (p = 0; p <= order; p++) {
          *CTE_A0(p, 0, 0) = gv0[p];
          *CTE_AK(p, 0, 0) = gvk[p];
        }
        /* Row 1: three entries, each level p from level p+1's single row-0. */
        for (p = 0; p < order; p++) {
          double a0 = *CTE_A0(p + 1, 0, 0), ak = *CTE_AK(p + 1, 0, 0);
          *CTE_A0(p, 1, 0) = a0 * r[2];  *CTE_AK(p, 1, 0) = ak * r[2];
          *CTE_A0(p, 1, 1) = a0 * r[1];  *CTE_AK(p, 1, 1) = ak * r[1];
          *CTE_A0(p, 1, 2) = a0 * r[0];  *CTE_AK(p, 1, 2) = ak * r[0];
        }

        /*
         * Rows 2..order, then the contraction of term n = iRow-1, which needs
         * exactly level 0's rows n and n+1 -- both live at this point. Doing it
         * here keeps the n ordering of the accumulation identical to the
         * unfused version, so y0/y1 come out bit-identical.
         */
        for (iRow = 1; iRow <= order; iRow++) {
          if (iRow >= 2) {
            for (p = 0; p <= order - iRow; p++) {
              int p1 = p + 1;
              int off  = CTE_ROWOFF(nMomT, iRow);
              int off1 = CTE_ROWOFF(nMomT, iRow - 1);
              int off2 = CTE_ROWOFF(nMomT, iRow - 2);
              int iRow1 = iRow - 1;
              int l = CTE_IDX(0, 0, iRow) - off;
              int l1 = CTE_IDX(0, 0, iRow1) - off1;
              int l2 = CTE_IDX(0, 0, iRow - 2) - off2;

              *CTE_A0(p, iRow, l) = *CTE_A0(p1, iRow1, l1) * r[2]
                                  + *CTE_A0(p1, iRow - 2, l2) * iRow1;
              *CTE_AK(p, iRow, l) = *CTE_AK(p1, iRow1, l1) * r[2]
                                  + *CTE_AK(p1, iRow - 2, l2) * iRow1;
              l++;
              l1 = CTE_IDX(0, 0, iRow1) - off1;
              *CTE_A0(p, iRow, l) = *CTE_A0(p1, iRow1, l1) * r[1];
              *CTE_AK(p, iRow, l) = *CTE_AK(p1, iRow1, l1) * r[1];
              l++;
              for (i2 = 2; i2 <= iRow; i2++, l++) {
                int i3 = iRow - i2;
                l1 = CTE_IDX(0, i2 - 1, i3) - off1;
                l2 = CTE_IDX(0, i2 - 2, i3) - off2;
                *CTE_A0(p, iRow, l) = *CTE_A0(p1, iRow1, l1) * r[1]
                                    + *CTE_A0(p1, iRow - 2, l2) * (i2 - 1);
                *CTE_AK(p, iRow, l) = *CTE_AK(p1, iRow1, l1) * r[1]
                                    + *CTE_AK(p1, iRow - 2, l2) * (i2 - 1);
              }
              for (i2 = 0; i2 <= iRow1; i2++, l++) {
                int i3 = iRow1 - i2;
                l1 = CTE_IDX(0, i2, i3) - off1;
                *CTE_A0(p, iRow, l) = *CTE_A0(p1, iRow1, l1) * r[0];
                *CTE_AK(p, iRow, l) = *CTE_AK(p1, iRow1, l1) * r[0];
              }
              for (i1 = 2; i1 <= iRow; i1++) {
                for (i2 = 0; i2 <= iRow - i1; i2++, l++) {
                  int i3 = iRow - i1 - i2;
                  l1 = CTE_IDX(i1 - 1, i2, i3) - off1;
                  l2 = CTE_IDX(i1 - 2, i2, i3) - off2;
                  *CTE_A0(p, iRow, l) = *CTE_A0(p1, iRow1, l1) * r[0]
                                      + *CTE_A0(p1, iRow - 2, l2) * (i1 - 1);
                  *CTE_AK(p, iRow, l) = *CTE_AK(p1, iRow1, l1) * r[0]
                                      + *CTE_AK(p1, iRow - 2, l2) * (i1 - 1);
                }
              }
            }
          }

          n = iRow - 1;
          if (n <= ordHere) {
            int offN  = CTE_ROWOFF(nMomT, n);
            int offN1 = CTE_ROWOFF(nMomT, n + 1);
            i = offN;
            for (i1 = 0; i1 <= n; i1++) {
              for (i2 = 0; i2 <= n - i1; i2++, i++) {
                int i3 = n - i1 - i2;
                int lA = CTE_IDX(i1 + 1, i2, i3) - offN1;
                int lB = CTE_IDX(i1, i2 + 1, i3) - offN1;
                int lC = CTE_IDX(i1, i2, i3 + 1) - offN1;
                int l  = i - offN;
                double dnG0, dnGk;

                dnG0 = nrm[0] * *CTE_A0(0, n + 1, lA)
                     + nrm[1] * *CTE_A0(0, n + 1, lB)
                     + nrm[2] * *CTE_A0(0, n + 1, lC);
                dnGk = nrm[0] * *CTE_AK(0, n + 1, lA)
                     + nrm[1] * *CTE_AK(0, n + 1, lB)
                     + nrm[2] * *CTE_AK(0, n + 1, lC);
                y0 += sgn3d[i] * mom[i] * (*CTE_A0(0, n, l) - *CTE_AK(0, n, l));
                y1 += sgn3d[i] * mom[i] * (eps * dnGk - dnG0);
              }
            }
          }
        }
#undef CTE_A0
#undef CTE_AK
#undef CTE_IDX
        continue;
      }

      if (nodeLevel[node] == chgDepth) {
        int start = nodeChgStart[node];
        int cnt = nodeChgCount[node];
        int i;
        for (i = 0; i < cnt; i++) {
          int j = chgIdx[start + i];
          double x[3], r2, rn, G0, Gk, coef, ip, dG0dn, dGkdn;
          for (k = 0; k < 3; k++) x[k] = qpt[k] - pos[3 * j + k];
          r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
          rn = sqrt(r2);
          G0 = fourPiIv / rn;
          Gk = exp(-kap * rn) * G0;
          coef = (kap * rn + 1.0) * exp(-kap * rn);
          ip = nrm[0] * x[0] + nrm[1] * x[1] + nrm[2] * x[2];
          dG0dn = -ip * G0 / r2;
          dGkdn = coef * dG0dn;
          y0 += chr[j] * (G0 - Gk);
          y1 += chr[j] * (eps * dGkdn - dG0dn);
        }
        continue;
      }

      {
        /* Push children in reverse so they pop in ascending order, matching the
         * order the CPU recursion visits them. Without this the subtrees are
         * summed back-to-front and the result differs at roundoff (4.2e-12
         * measured on the virus RHS) for no reason. */
        int ks = nodeKidStart[node];
        int kc = nodeKidCount[node];
        int i;
        for (i = kc - 1; i >= 0 && sp < CTE_MAX_STACK; i--) stack[sp++] = kidIdx[ks + i];
      }
    }

    {
      int idx = panelOrigIdx[panelIdx];
      /* qOrd==1: weight 0.5 then the 2*area factor, so net = area. */
      threadSum += (y0 * pn->area) * sgm[idx + nPanels] +
                   (y1 * pn->area) * sgm[idx];
    }
  }
  blockSum[threadIdx.x] = threadSum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      blockSum[threadIdx.x] += blockSum[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = blockSum[0];
  }
}

extern "C" double energyTreeTheta(void);

int gpuPanelChargeTreeEnergy(ssystem *sys, const double *sgm, double *pot) {
  int blockSize = 128, gridSize;
  long long stride, slabDoubles;
  size_t scratchBytes;
  cudaError_t err;
  std::vector<double> hostPartial;
  std::vector<int> levOff;
  double total = 0.0;
  int p, i;
  static double *d_scratch = NULL;
  static int *d_levOff = NULL;
  static double *d_sgm = NULL;
  static size_t d_sgmBytes = 0;
  static double *d_partial = NULL;
  static int d_partialCount = 0;
  static long long scratchStride = 0;
  double tBuild0, tBuild1, tScratch0, tScratch1, tUpload0, tUpload1;
  double tKernel0, tKernel1, tCopy0, tCopy1, tReduce0, tReduce1;
  int rebuiltTree = 0;

  if (sys == NULL || sgm == NULL || pot == NULL) return 0;
  if (!gpuBackendAvailable()) return 0;
  if (sys->maxQuadOrder != 1) return 0;   /* kernel assumes the 1-point rule */
  if (sys->chgCubeList == NULL || sys->chgCubeList[0] == NULL) return 0;

  tBuild0 = wall_seconds_cuda_local();
  if (gCTE.sys != sys) {
    freeChargeTreeGpu();
    cudaFree(d_scratch); d_scratch = NULL;
    cudaFree(d_levOff);  d_levOff = NULL;
    scratchStride = 0;
    if (!buildChargeTreeGpu(sys)) { freeChargeTreeGpu(); return 0; }
    rebuiltTree = 1;
  }
  tBuild1 = wall_seconds_cuda_local();
  if (gCTE.derivOrderMax + 1 >= 16) return 0;   /* fixed pointer arrays */
  if (!cteStackDepthOk(sys->chgDepth)) {
    setNearfieldLastError("charge-tree depth %d needs a %d-entry device stack, "
                          "CTE_MAX_STACK is %d", sys->chgDepth,
                          7 * sys->chgDepth + 1, CTE_MAX_STACK);
    return 0;
  }

  /* Two rows per level, each sized for the longest row that level can hold:
   * row(derivMax - p). levOff[derivMax+1] is the per-array total, which also
   * separates the dg0 half from the dgk half. Summed, that is 2*nMom[derivMax]
   * against the old pyramid's sum_m nMom[m]. */
  levOff.resize((size_t)gCTE.derivOrderMax + 2);
  levOff[0] = 0;
  for (p = 0; p <= gCTE.derivOrderMax; p++) {
    int m = gCTE.derivOrderMax - p;
    int rowLen = sys->nMom[m] - ((m <= 0) ? 0 : sys->nMom[m - 1]);
    levOff[(size_t)p + 1] = levOff[(size_t)p] + 2 * rowLen;
  }
  slabDoubles = 2LL * levOff[(size_t)gCTE.derivOrderMax + 1];

  /* Cap concurrency so the scratch stays bounded; panels are then taken in a
   * grid-stride loop. See chargeTreeScratchThreads for how perSm was measured. */
  stride = chargeTreeScratchThreads("FABIPB_ENERGY_GPU_SCRATCH_MIB", 0, 3465.0,
                                    slabDoubles, gCTE.nPanels, blockSize);
  scratchBytes = (size_t)stride * (size_t)slabDoubles * sizeof(double);
  gridSize = (int)(stride / blockSize);

  tScratch0 = wall_seconds_cuda_local();
  if (scratchStride != stride) {
    cudaFree(d_scratch); d_scratch = NULL;
    if (!cudaMallocNearfield((void **)&d_scratch, scratchBytes, "cte scratch")) {
      scratchStride = 0;
      return 0;
    }
    scratchStride = stride;
  }
  if (d_levOff == NULL) {
    if (!cudaMallocNearfield((void **)&d_levOff, levOff.size() * sizeof(int), "cte levOff") ||
        !cudaMemcpyNearfield(d_levOff, &levOff[0], levOff.size() * sizeof(int),
                             cudaMemcpyHostToDevice, "cte levOff")) return 0;
  }
  if (d_sgmBytes < (size_t)2 * sys->nPnls * sizeof(double)) {
    cudaFree(d_sgm);
    d_sgm = NULL;
    d_sgmBytes = (size_t)2 * sys->nPnls * sizeof(double);
    if (!cudaMallocNearfield((void **)&d_sgm, d_sgmBytes, "cte sgm")) return 0;
  }
  if (d_partialCount < gridSize) {
    cudaFree(d_partial);
    d_partial = NULL;
    d_partialCount = gridSize;
    if (!cudaMallocNearfield((void **)&d_partial,
                             (size_t)gridSize * sizeof(double),
                             "cte partial")) return 0;
  }
  tScratch1 = wall_seconds_cuda_local();

  tUpload0 = wall_seconds_cuda_local();
  if (!cudaMemcpyNearfield(d_sgm, sgm, d_sgmBytes,
                           cudaMemcpyHostToDevice, "cte sgm")) return 0;
  tUpload1 = wall_seconds_cuda_local();

  tKernel0 = wall_seconds_cuda_local();
  chargeTreeEnergyKernel<<<gridSize, blockSize, (size_t)blockSize * sizeof(double)>>>(
      gCTE.nPanels, gCTE.derivOrderMax, gCTE.idxDim, sys->chgDepth, sys->height,
      energyTreeTheta(), kappa, epsilon, fourPiI,
      gCTE.d_panels,
      gCTE.d_nodeCtr, gCTE.d_nodeRad, gCTE.d_nodeLevel,
      gCTE.d_nodeKidStart, gCTE.d_nodeKidCount,
      gCTE.d_nodeChgStart, gCTE.d_nodeChgCount,
      gCTE.d_nodeMomOff, gCTE.d_nodeOrder,
      gCTE.d_kidIdx, gCTE.d_chgIdx, gCTE.d_moments,
      gCTE.d_pos, gCTE.d_chr,
      gCTE.d_idx3, gCTE.d_sgn3, d_levOff, gCTE.d_nMom,
      gCTE.d_panelIdx, d_sgm,
      d_scratch, stride,
      d_partial);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    setNearfieldLastError("charge-tree energy launch failed: %s", cudaGetErrorString(err));
    return 0;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    setNearfieldLastError("charge-tree energy kernel failed: %s", cudaGetErrorString(err));
    return 0;
  }
  tKernel1 = wall_seconds_cuda_local();

  tCopy0 = wall_seconds_cuda_local();
  hostPartial.resize((size_t)gridSize);
  if (!cudaMemcpyNearfield(&hostPartial[0], d_partial,
                           (size_t)gridSize * sizeof(double),
                           cudaMemcpyDeviceToHost, "cte partial")) return 0;
  tCopy1 = wall_seconds_cuda_local();

  tReduce0 = wall_seconds_cuda_local();
  for (i = 0; i < gridSize; i++) total += hostPartial[(size_t)i];
  tReduce1 = wall_seconds_cuda_local();
  *pot = total;
  if (sys->benchmarkMode > 0) {
    printf("GPU charge-tree energy: threads=%lld grid=%d scratch=%.3f GiB slab=%lld doubles\n",
           stride, gridSize, bytesToGiB(scratchBytes), slabDoubles);
    printf("GPU charge-tree energy breakdown (s): build=%.6f scratch=%.6f sgm_h2d=%.6f kernel=%.6f partial_d2h=%.6f reduce=%.6f rebuilt=%d\n",
           tBuild1 - tBuild0, tScratch1 - tScratch0, tUpload1 - tUpload0,
           tKernel1 - tKernel0, tCopy1 - tCopy0, tReduce1 - tReduce0,
           rebuiltTree);
  }
  return 1;
}


/* Coulomb-only derivative tensor; port of setupCoulombDerivsLocal(). Half the
 * work and half the scratch of the screened version -- no Gk half. */
__device__ void cteSetupCoulombDerivsStrided(int order, const double *x,
                                             double fourPiIv,
                                             const int *fidx3, int D,
                                             double *gv0, double **dg0,
                                             long long S) {
  int p, p1, iRow, iRow1, i1, i2, i3, idx, idx1, idx2, k;
  double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  double r2;

  gv0[0] = fourPiIv / r;
  r2 = -1.0 / (r * r);
  for (k = 0; k < order; k++) gv0[k + 1] = (2 * k + 1) * r2 * gv0[k];
  for (p = 0; p <= order; p++) dg0[p][0] = gv0[p];
  for (p = 0; p < order; p++) {
    p1 = p + 1;
    dg0[p][(long long)1 * S] = dg0[p1][0] * x[2];
    dg0[p][(long long)2 * S] = dg0[p1][0] * x[1];
    dg0[p][(long long)3 * S] = dg0[p1][0] * x[0];
  }
#define CTE_IDX(a,b,c) fidx3[(((a) * D) + (b)) * D + (c)]
  for (iRow = 2; iRow <= order; iRow++) {
    for (p = 0; p <= order - iRow; p++) {
      p1 = p + 1;
      idx = CTE_IDX(0, 0, iRow);
      iRow1 = iRow - 1;
      idx1 = CTE_IDX(0, 0, iRow1);
      idx2 = CTE_IDX(0, 0, iRow - 2);
      dg0[p][(long long)idx * S] = dg0[p1][(long long)idx1 * S] * x[2]
                                 + dg0[p1][(long long)idx2 * S] * iRow1;
      idx++;
      idx1 = CTE_IDX(0, 0, iRow1);
      dg0[p][(long long)idx * S] = dg0[p1][(long long)idx1 * S] * x[1];
      idx++;
      for (i2 = 2; i2 <= iRow; i2++, idx++) {
        i3 = iRow - i2;
        idx1 = CTE_IDX(0, i2 - 1, i3);
        idx2 = CTE_IDX(0, i2 - 2, i3);
        dg0[p][(long long)idx * S] = dg0[p1][(long long)idx1 * S] * x[1]
                                   + dg0[p1][(long long)idx2 * S] * (i2 - 1);
      }
      for (i2 = 0; i2 <= iRow1; i2++, idx++) {
        i3 = iRow1 - i2;
        idx1 = CTE_IDX(0, i2, i3);
        dg0[p][(long long)idx * S] = dg0[p1][(long long)idx1 * S] * x[0];
      }
      for (i1 = 2; i1 <= iRow; i1++) {
        for (i2 = 0; i2 <= iRow - i1; i2++, idx++) {
          i3 = iRow - i1 - i2;
          idx1 = CTE_IDX(i1 - 1, i2, i3);
          idx2 = CTE_IDX(i1 - 2, i2, i3);
          dg0[p][(long long)idx * S] = dg0[p1][(long long)idx1 * S] * x[0]
                                     + dg0[p1][(long long)idx2 * S] * (i1 - 1);
        }
      }
    }
  }
#undef CTE_IDX
}

/*
 * setupRHS's charge-tree walk. Same traversal as the energy kernel, but the
 * Coulomb-only kernel: no screening, no epsilon, and half the scratch, so more
 * of the working set stays in cache.
 */
__global__ void chargeTreeRhsKernel(
    int nPanels, int derivMax, int idxDim, int chgDepth, int height,
    double theta, double fourPiIv, double fac,
    const NearPanelGeom *panels,
    const double *nodeCtr, const double *nodeRad, const int *nodeLevel,
    const int *nodeKidStart, const int *nodeKidCount,
    const int *nodeChgStart, const int *nodeChgCount,
    const int *nodeMomOff, const int *nodeOrder,
    const int *kidIdx, const int *chgIdx, const double *moments,
    const double *pos, const double *chr,
    const int *fidx3, const int *sgn3d, const int *levOff,
    double *scratch, long long stride,
    double *out) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long nThreadsTotal = (long long)gridDim.x * blockDim.x;
  double *dg0[16];
  double gv0[18];
  int p, panelIdx;

  if (tid >= stride) return;
  for (p = 0; p <= derivMax && p < 16; p++) {
    dg0[p] = scratch + (long long)levOff[p] * stride + tid;
  }

  for (panelIdx = (int)tid; panelIdx < nPanels; panelIdx += (int)nThreadsTotal) {
    const NearPanelGeom *pn = &panels[panelIdx];
    double qpt[3], nrm[3];
    double y0 = 0.0, y1 = 0.0;
    int stack[CTE_MAX_STACK];
    int sp = 0;
    int k;

    for (k = 0; k < 3; k++) {
      qpt[k] = pn->vtx[0][k] + 0.5 * (pn->a2[k] + 0.5 * pn->a0[k]);
      nrm[k] = pn->normal[k];
    }
    stack[sp++] = 0;

    while (sp > 0) {
      int node = stack[--sp];
      double r[3], dist;
      for (k = 0; k < 3; k++) r[k] = qpt[k] - nodeCtr[3 * node + k];
      dist = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

      if (nodeRad[node] < theta * dist && nodeLevel[node] >= height) {
        int ordHere = nodeOrder[node];
        const double *mom = &moments[nodeMomOff[node]];
        double c0 = 0.0, c1 = 0.0;
        int i, i1, i2, n;

        cteSetupCoulombDerivsStrided(ordHere + 1, r, fourPiIv, fidx3, idxDim,
                                     gv0, dg0, stride);
        i = 0;
        for (n = 0; n <= ordHere; n++) {
          for (i1 = 0; i1 <= n; i1++) {
            for (i2 = 0; i2 <= n - i1; i2++, i++) {
              int i3 = n - i1 - i2;
              double dn;
#define CTE_IDX(a,b,c) fidx3[(((a) * idxDim) + (b)) * idxDim + (c)]
              dn = nrm[0] * dg0[0][(long long)CTE_IDX(i1 + 1, i2, i3) * stride]
                 + nrm[1] * dg0[0][(long long)CTE_IDX(i1, i2 + 1, i3) * stride]
                 + nrm[2] * dg0[0][(long long)CTE_IDX(i1, i2, i3 + 1) * stride];
#undef CTE_IDX
              c0 += sgn3d[i] * mom[i] * dg0[0][(long long)i * stride];
              c1 += sgn3d[i] * mom[i] * dn;
            }
          }
        }
        /* chgClusterEvalLocal divides both components by fourPiI */
        y0 += c0 / fourPiIv;
        y1 += c1 / fourPiIv;
        continue;
      }

      if (nodeLevel[node] == chgDepth) {
        int start = nodeChgStart[node];
        int cnt = nodeChgCount[node];
        int i;
        for (i = 0; i < cnt; i++) {
          int j = chgIdx[start + i];
          double x[3], r2, ri, r3i, ip;
          for (k = 0; k < 3; k++) x[k] = qpt[k] - pos[3 * j + k];
          r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
          ri = 1.0 / sqrt(r2);
          r3i = ri / r2;
          ip = nrm[0] * x[0] + nrm[1] * x[1] + nrm[2] * x[2];
          y0 += chr[j] * ri;
          y1 += chr[j] * (-ip * r3i);
        }
        continue;
      }

      {
        /* Push children in reverse so they pop in ascending order, matching the
         * order the CPU recursion visits them. Without this the subtrees are
         * summed back-to-front and the result differs at roundoff (4.2e-12
         * measured on the virus RHS) for no reason. */
        int ks = nodeKidStart[node];
        int kc = nodeKidCount[node];
        int i;
        for (i = kc - 1; i >= 0 && sp < CTE_MAX_STACK; i--) stack[sp++] = kidIdx[ks + i];
      }
    }

    /* qOrd==1: weight 0.5 then 2*area gives area; setupRHS then applies fac. */
    out[panelIdx] = y0 * pn->area * fac;
    out[nPanels + panelIdx] = y1 * pn->area * fac;
  }
}

extern "C" double rhsTreeTheta(void);

int gpuChargeTreeRHS(ssystem *sys, int qOrder, double fac, double *sgm) {
  int blockSize = 128, gridSize;
  long long stride, slabDoubles;
  size_t scratchBytes;
  cudaError_t err;
  std::vector<double> host;
  std::vector<int> levOff;
  int p, i;
  static double *d_scratch = NULL;
  static int *d_levOff = NULL;
  static long long scratchStride = 0;
  static const ssystem *scratchSys = NULL;

  if (sys == NULL || sgm == NULL) return 0;
  if (!gpuBackendAvailable()) return 0;
  if (qOrder != 1) return 0;              /* kernel assumes the 1-point rule */
  if (sys->chgCubeList == NULL || sys->chgCubeList[0] == NULL) return 0;
  if (envFlagEnabled("FABIPB_RHS_GPU_DISABLE")) return 0;

  if (gCTE.sys != sys) {
    freeChargeTreeGpu();
    if (!buildChargeTreeGpu(sys)) { freeChargeTreeGpu(); return 0; }
  }
  if (gCTE.derivOrderMax + 1 >= 16) return 0;
  if (!cteStackDepthOk(sys->chgDepth)) {
    setNearfieldLastError("charge-tree depth %d needs a %d-entry device stack, "
                          "CTE_MAX_STACK is %d", sys->chgDepth,
                          7 * sys->chgDepth + 1, CTE_MAX_STACK);
    return 0;
  }

  if (scratchSys != sys) {
    cudaFree(d_scratch); d_scratch = NULL;
    cudaFree(d_levOff);  d_levOff = NULL;
    scratchStride = 0;
    scratchSys = sys;
  }

  /* Only the Coulomb half is needed here, so the slab is half the energy
   * kernel's and more of it stays resident. */
  levOff.resize((size_t)gCTE.derivOrderMax + 2);
  levOff[0] = 0;
  for (p = 0; p <= gCTE.derivOrderMax; p++) {
    levOff[(size_t)p + 1] = levOff[(size_t)p] + sys->nMom[gCTE.derivOrderMax - p];
  }
  slabDoubles = levOff[(size_t)gCTE.derivOrderMax + 1];

  /* Lower perSm than the energy kernel: this one steps up sharply in time past
   * roughly 16k threads on the measurement card. */
  stride = chargeTreeScratchThreads("FABIPB_RHS_GPU_SCRATCH_MIB", 112, 0.0,
                                    slabDoubles, gCTE.nPanels, blockSize);
  scratchBytes = (size_t)stride * (size_t)slabDoubles * sizeof(double);

  if (scratchStride != stride) {
    cudaFree(d_scratch); d_scratch = NULL;
    if (!cudaMallocNearfield((void **)&d_scratch, scratchBytes, "rhs scratch")) {
      scratchStride = 0; return 0;
    }
    scratchStride = stride;
  }
  if (d_levOff == NULL) {
    if (!cudaMallocNearfield((void **)&d_levOff, levOff.size() * sizeof(int), "rhs levOff") ||
        !cudaMemcpyNearfield(d_levOff, &levOff[0], levOff.size() * sizeof(int),
                             cudaMemcpyHostToDevice, "rhs levOff")) return 0;
    }

  gridSize = (int)(stride / blockSize);
  chargeTreeRhsKernel<<<gridSize, blockSize>>>(
      gCTE.nPanels, gCTE.derivOrderMax, gCTE.idxDim, sys->chgDepth, sys->height,
      rhsTreeTheta(), fourPiI, fac,
      gCTE.d_panels,
      gCTE.d_nodeCtr, gCTE.d_nodeRad, gCTE.d_nodeLevel,
      gCTE.d_nodeKidStart, gCTE.d_nodeKidCount,
      gCTE.d_nodeChgStart, gCTE.d_nodeChgCount,
      gCTE.d_nodeMomOff, gCTE.d_nodeOrder,
      gCTE.d_kidIdx, gCTE.d_chgIdx, gCTE.d_moments,
      gCTE.d_pos, gCTE.d_chr,
      gCTE.d_idx3, gCTE.d_sgn3, d_levOff,
      d_scratch, stride,
      gCTE.d_out);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    setNearfieldLastError("charge-tree RHS launch failed: %s", cudaGetErrorString(err));
    return 0;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    setNearfieldLastError("charge-tree RHS kernel failed");
    return 0;
  }

  /* d_out is sized 2*nPanels, which is exactly the RHS layout. */
  host.resize((size_t)2 * gCTE.nPanels);
  if (!cudaMemcpyNearfield(&host[0], gCTE.d_out,
                           (size_t)2 * gCTE.nPanels * sizeof(double),
                           cudaMemcpyDeviceToHost, "rhs out")) return 0;

  /* The kernel indexes panels in pnlLst order; sgm is indexed by pnl->idx. */
  for (i = 0; i < gCTE.nPanels; i++) {
    panel *pnl = sys->panelByIdx[i];
    sgm[pnl->idx] = host[i];
    sgm[pnl->idx + sys->nPnls] = host[gCTE.nPanels + i];
  }
  if (sys->benchmarkMode > 0) {
    printf("GPU charge-tree RHS: threads=%lld scratch=%.3f GiB slab=%lld doubles theta=%g\n",
           stride, bytesToGiB(scratchBytes), slabDoubles, rhsTreeTheta());
  }
  return 1;
}
