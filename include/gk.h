/*
 * gk.h
 * copyright   Johannes Tausch
 */
struct panel {          /* panel */
  double vtx[3][3];     /* vertices */
  double a[3][3];       /* sides */
  double x[3];          /* centroid */
  double normal[3];     /* normal, determined by right-hand rule */
  double nrm[3][3];     /* normals at vertices */
  double area;          /* area  */
  double area2;         /* 2*sqrt(area) */
  int shape;            /* 3=triangle, everything else generates an error */
  int nSurf;            /* number of face on surface (ith face) */
  int idx;              /* index in vector */
  struct cube *cube;    /* cube in which panel is located */
  struct panel *next;   /* next panel in input list. */
  struct panel *nextC;  /* next panel in linked list (Contiguous wn cubes) */
};
typedef struct panel panel;


struct edge {            /* edge */
  double v0[3];          /* startpoint */
  double v1[3];          /* endpoint */
};
typedef struct edge edge;


struct cube {               /* cube, actually a cluster of panels */
  int level;                /* 0 => root */
  int i, j, k;              /* cube is cubes[level][j][k][l] */
  int flatIdx;              /* flattened active-FMM cube index */
  int leafFlatIdx;          /* flattened finest-level cube index */
  double x[3];              /* Chebychev center */
  int nPnls;                /* number of panels in cube */
  panel *pnls;              /* linked list of panels with center in cube */
  int nNbrs;                /* Nbr of non-empty neighbors */
  int n2Nbrs;               /* Nbr of non-empty 2nd neighbors*/
  struct cube **nbrs;       /* Nbrs and 2nd nbrs with Panels */
  int nKids;                /* Number of kids */
  int nthKid;               /* index of this cube among parent's kids */
  struct cube *kids[8];     /* Array of kids ptrs. */
  struct cube *parent;      /* parent cube */
  double *mom_pot;          /* moments for potential */
  double *mom_dpdn;         /* moments for potential derivative */
  double *lec_k1;           /* local expansion coefficients of kernel1 */
  double *lec_k2;           /* local expansion coefficients of kernel2 */
  double *lec_k3;           /* local expansion coefficients of kernel3 */
  double *lec_k4;           /* local expansion coefficients of kernel4 */
  double eBoxLo[3];         /* lower corner of the enclosing box */
  double eBoxUp[3];         /* upper corner of the enclosing box */
  double eRad;              /* half-diameter of enclosing box */
  struct cube *next;        /* Ptr to next nonempty cube with panels */
  double *mom_chr;          /* multipole moments of enclosed charges (charge-tree only) */
  int *chgIdx;              /* indices into sys->pos/sys->chr for this leaf (charge-tree only) */
  int nChgs;                /* number of charges in/under this cube (charge-tree only) */
};
typedef struct cube cube;



struct ssystem {
  int depth;                /* # of levels of cubes. */
  int height;               /* highest level to throw out terms */
  int maxOrder;             /* maximal order */
  int nPnls;                /* nr of panels */
  int nChar;                /* nr of charges */
  int nVtxs;                /* nr of vertices */
//  int nSurf;                /* nr of surfaces */
  int nKerl;                /* nr of kernels */
  int layer;                /* single layer=0; double layer=1; adjoint=2 */
  int *ordMom;              /* order of moments to compute per level */
  int *ordM2L;              /* order used in M2L's */
  int *nMom;                /* number of moments (as a function of order) */
  int maxSngs;              /* max number of panels in a finest level cube */
  int max1Nbrs;             /* max number of frist neighbors that a cube can have */
  int maxQuadOrder;         /* maximal order for panel interactions */
  double maxSepRatio;       /* maximal separation ratio for cubes to be neighbors */
  double *pos, *chr;        /* charge position and charges for rhs */
  int mesh_flag;            /* mesh format flag */
  int benchmarkMode;        /* 0=quiet default, 1=print profiling/benchmark details */
  int gpuMode;              /* 0=CPU only, >0 request GPU backend */
  int debugCompareApply;    /* 0=off, >0 compare one CPU/GPU applyFMM call */
  int debugComparePrecond;  /* 0=off, >0 compare original/cached PtVfmm once */
  int matvecMode;           /* 0=FMM, 1=direct GPU baseline */
  int gpuQ2MMode;           /* 1=GPU Q2M, 0=CPU dgemv loop; default resolved after loadPanel */
  int gpuNearfieldMode;     /* 0=interaction+atomics, 1=thread-per-destination,
                              2=warp-per-destination (see gpu_nearfield_cuda.inc) */
  /*
   * 0=original, 1=cached local blocks, 2=cached LU, 3=diagonal/Jacobi.
   *
   * Mode 3 is faster than mode 2 in nearly everything measured, despite
   * usually costing one more iteration: mode 2's setup and heavier solve
   * outweigh the better convergence. Wall clock at tol 1e-4, R=1.0, q=1,
   * a=30, mode 2 against mode 3:
   *
   *   1cbn eps1=1   8 its 0.381 s  |  9 its 0.326 s
   *   1cbn eps1=4   8 its 0.505 s  |  9 its 0.381 s
   *   1ajj eps1=1   8 its 0.478 s  |  9 its 0.371 s
   *   1ajj eps1=4   8 its 0.486 s  |  9 its 0.360 s
   *   1a63 eps1=1  29 its 1.047 s  | 83 its 1.359 s   <- the one mode-2 win
   *   1a63 eps1=4  20 its 0.949 s  | 20 its 0.757 s
   *
   * The exception is driven by dielectric contrast, not problem size: at
   * eps1=1/eps2=80 the diagonal preconditioner needs 83 iterations on 1a63,
   * and at eps1=4/eps2=80 it needs 20, the same as mode 2. Since the capsid
   * runs use eps1=4, that regime is the relevant one, and there mode 2 is
   * strictly worse -- at sdens=2 it needed 51 iterations and 356.7 s of
   * preconditioner solve against mode 3's 37 and 0.34 s, and at sdens=1 it
   * stalled at the 100-iteration cap with residual 1.24e-1 where mode 3
   * converged in 87. (Those capsid figures predate the transL2L fix and other
   * changes; the direction is solid, the numbers are stale.)
   *
   * The default is resolved after loadPanel: huge capsids use mode 3 to avoid
   * memory and solve-time pressure, small/medium high-contrast dielectric cases
   * use mode 2, and other cases use mode 3.
   */
  int precondCacheMode;
  int nLeafCubesFlat;       /* flattened finest-level cube count */
  int *leafPanelStart;      /* flattened per-leaf panel start index */
  int *leafPanelCount;      /* flattened per-leaf panel count */
  int nNearPairsFlat;       /* flattened leaf-neighbor pair count */
  int *nearPairSrc;         /* flattened nearfield source leaf index */
  int *nearPairDst;         /* flattened nearfield target leaf index */
  /*
   * First near-pair index of each destination leaf, nLeafCubesFlat+1 entries.
   * buildApplyLayout emits pairs destination-major -- the destination cube is
   * the outer loop and leafFlatIdx is assigned in the same cubeList order -- so
   * this is a running offset, not a sort. It lets the host near-field apply be
   * split across threads by destination leaf, giving disjoint pot[] output.
   */
  int *nearLeafPairOffset;
  int nFmmCubesFlat;        /* flattened FMM cube count across active levels */
  struct cube **fmmCubeByIdx; /* flattened cube lookup by active FMM index */
  int nM2LPairsFlat;        /* flattened M2L interaction count */
  int *m2lPairSrc;          /* flattened M2L source cube index */
  int *m2lPairDst;          /* flattened M2L destination cube index */
  int *m2lPairOrder;        /* flattened M2L order by interaction */
  int nM2LDstGroups;        /* number of destination-grouped M2L ranges */
  int *m2lDstGroupStart;    /* start offset of each destination-grouped M2L range */
  int *m2lDstGroupCount;    /* interaction count of each destination-grouped M2L range */
  panel **panelByIdx;       /* direct panel lookup by contiguous index */
  double *panelArea;        /* pnl->area by contiguous index; see buildPanelIndex */
  int *fmmLevelStart;       /* first fmmCubeByIdx entry of each level */
  int *fmmLevelCount;       /* cube count of each level */
  int maxlevCudes;          /* max cubes at finest level */
  int maxlevnPnls;          /* max panels in a cube at finest level */
  panel *pnlLst;            /* linked list of panels (Contiguous wn cubes) */
  panel *pnlOLst;           /* linked list of original order panels */
  cube **cubeList;          /* heads of lists of cubes for each level */
  int chgDepth;             /* depth of the charge-only tree (v1: equals depth) */
  cube **chgCubeList;       /* heads of per-level charge-tree cube lists */
};
typedef struct ssystem ssystem;

typedef struct {
  int maxOrder;
  int derivOrder;
  double *gvals0;
  double *gvalsk;
  double **dg0;
  double **dgk;
} RhsTreeWorkspace;

/*
 * Per-thread scratch for transM2M/transL2L.
 *
 * Both used only the file-scope buffers fcnBuf1/2/3 (moments.c) and
 * convVecR/convVec1/convVec2 (expan.c), which made them unsafe to run on more
 * than one thread. The workspace-taking variants take these explicitly so the
 * upward and downward passes can be split across cubes; the original
 * signatures remain as wrappers over the shared buffers for the single-threaded
 * callers in moments.c and chargeTree.c.
 */
typedef struct {
  int order;                /* fcn buffers hold order+1 entries */
  int nMoments;             /* conv vectors hold nMoments entries */
  double *fcn1, *fcn2, *fcn3;
  double *convR, *conv1, *conv2;
} TransWorkspace;

typedef void (*KernelFn)(double *x, double *y);
typedef void (*KernelDerivFn)(double r, int p, double *G);
typedef void (*MatVecFn)(ssystem *sys, double *sgm, double *pot);
