/*
 * chargeTree.c
 *   builds an independent octree over the point charges, and computes
 *   multipole moments of the enclosed charges per cube. This is used by
 *   the tree-accelerated setupRHS path (see rhsTreeWalk in treecode.c)
 *   for problem sizes where the direct panel-charge loop is intractable.
 *
 *   Mirrors the panel cube-tree construction in gkSetup.c, but cannot
 *   reuse it directly: linkcubes() assumes every finest-level cube it
 *   links has a non-empty panel list (cb->pnls), which a charge-only
 *   leaf cube would not have.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include "gkGlobal.h"
#include "gk.h"

/* globals owned by gkSetup.c; temporarily repointed at the charge
 * bounding box/depth while the charge tree is built, then restored. */
extern double *sideLen;
extern int maxLev;
extern double minx, miny, minz;
extern cube *firstCube, *lastCube;

extern int ***idx3, *sgn3;
extern double *ifact3;
extern double *fcnBuf1, *fcnBuf2, *fcnBuf3, *convVecR;

cube *goDown(double *refPt, cube *cb);
void setupConVect(int order, double *trns, double *convVec);

/*
 * Taylor order for the charge-tree clusters used by setupRHS and the post-solve
 * energy.
 *
 * FABIPB_CHARGE_TREE_ORDER=<n> overrides the default set below. It exists so
 * the charge-tree order can be varied without touching -pm, which would change
 * the matvec accuracy at the same time and confound the comparison. Still
 * capped by sys->maxOrder, since the moment tables are only allocated that far.
 */
int chargeTreeOrderOverride(void) {
  static int initialized = 0;
  static int override = 0;    /* 0 = follow the FMM order */

  if (!initialized) {
    const char *env = getenv("FABIPB_CHARGE_TREE_ORDER");
    initialized = 1;
    if (env != NULL) {
      char *endptr = NULL;
      long value;

      errno = 0;
      value = strtol(env, &endptr, 10);
      if (errno == 0 && endptr != env && *endptr == '\0' &&
          value >= 1 && value <= 32) {
        override = (int)value;
      } else {
        fprintf(stderr,
                "Warning: ignoring invalid FABIPB_CHARGE_TREE_ORDER='%s'\n", env);
      }
    }
  }
  return override;
}

/*
 * Default Taylor order for charge-tree clusters: a flat 3.
 *
 * This used to follow the FMM's per-level ordMom, giving p=4 at the fine levels
 * and rising to p=7 near the root. That adaptivity is a good trade on a CPU,
 * where each cluster pays for its own order, but it works against the GPU
 * kernels: the per-thread derivative scratch slab is sized by derivMax, the
 * *maximum* order over all levels, so a handful of near-root clusters at p=7
 * imposed a 660-double slab on every thread. A flat lower order shrinks the
 * slab for everyone.
 *
 * Measured on H1N1 sdens=0.5 at theta=0.8, whole-solve time and energy drift
 * against theta=0.2/adaptive (results/paper/benchmarks/t19,t20):
 *
 *   p        total     rhs+energy   share of run   drift
 *   1        290.5 s      17.6 s        6.1%       1.031%
 *   3        303.9 s      29.2 s        9.6%       0.191%
 *   5        339.7 s      66.2 s       19.5%       0.052%
 *   adaptive 374.0 s     103.3 s       27.6%       0.076%
 *
 * p=3 is the floor of the useful range. Below it the walk is bound by traversal
 * and direct evaluation rather than by coefficient count, so p=1 buys only 4.4%
 * more time for 5.4x the error. Above it the extra accuracy is invisible next
 * to the mesh discretization error, which exceeds 10% at these densities.
 *
 * As with the acceptance ratio, mesh-convergence series and cross-solver
 * comparisons need more than this -- they resolve differences smaller than the
 * tree error. Set FABIPB_CHARGE_TREE_ORDER=7 (or higher) alongside a tighter
 * FABIPB_*_TREE_THETA for those.
 */
/* Policy default, set by resolveAutoSolverPolicy(); 0 means "follow the FMM's
 * per-level order", which is the conservative choice for small problems. An
 * explicit FABIPB_CHARGE_TREE_ORDER always wins over this. */
static int gChargeTreeOrderPolicy = 0;

void setChargeTreeOrderPolicy(int order) {
  if (order >= 1 && order <= 32) {
    gChargeTreeOrderPolicy = order;
  }
}

/* The order that will actually be used, for logging: env override if set,
 * otherwise the size-gated policy, otherwise 0 meaning "follow the FMM". */
int rhsChargeExpansionOrderReport(void) {
  int override = chargeTreeOrderOverride();
  if (override > 0) return override;
  return gChargeTreeOrderPolicy;   /* 0 = adaptive, follows the FMM order */
}

int rhsChargeExpansionOrder(ssystem *sys, int level) {
  int override = chargeTreeOrderOverride();
  int order;

  if (override > 0) {
    order = override;                       /* explicit env wins */
  } else if (gChargeTreeOrderPolicy > 0) {
    order = gChargeTreeOrderPolicy;         /* size-gated policy, flat */
  } else {
    order = MAX(sys->ordMom[level], 4);     /* conservative: follow the FMM */
  }
  return MIN(order, sys->maxOrder);
}

/*
 * trimmed analogue of linkcubes() in gkSetup.c: links cubes into
 * per-level lists and accumulates nChgs upward. No panel-list
 * stitching, since charge-tree leaves carry no panels.
 */
static void linkChgCubes(cube *cb, cube **firstLst, cube **lastLst) {
  int i1, i2;
  int lev = cb->level;

  /* goDown() threads newly-created maxLev cubes onto its own
   * chronological firstCube/lastCube chain (see chargeTree.c's
   * buildChargeTree), leaving a stale ->next on this cube from that
   * chain. We're about to make cb the new tail of its level's
   * DFS-order sibling list, so cb->next must start NULL here or the
   * list can splice into an earlier part of itself and cycle. */
  if (lev == maxLev) {
    cb->next = NULL;
  }

  if (lastLst[lev] == NULL) {
    firstLst[lev] = lastLst[lev] = cb;
  } else {
    lastLst[lev]->next = cb;
    lastLst[lev] = cb;
  }

  if (lev < maxLev) {
    for (i1 = 0; i1 < 8; i1++) {
      if (cb->kids[i1] != NULL) cb->nKids++;
    }
    for (i1 = i2 = 0; i1 < 8; i1++) {
      if (cb->kids[i1] != NULL) cb->kids[i2++] = cb->kids[i1];
    }
    for (i1 = cb->nKids; i1 < 8; i1++) {
      cb->kids[i1] = NULL;
    }
    for (i2 = 0; i2 < cb->nKids; i2++) {
      linkChgCubes(cb->kids[i2], firstLst, lastLst);
      cb->nChgs += cb->kids[i2]->nChgs;
    }
  }
} /* linkChgCubes */

/*
 * trimmed analogue of getEnclBoxs() in gkSetup.c, driven by charge
 * positions (via each leaf's chgIdx) instead of panel vertices.
 */
static void getChgEnclBoxs(ssystem *sys, cube **firstLst) {
  double *up, *lo;
  cube *cb, *kid;
  int i, k, lev;
  int depth = sys->chgDepth;

  for (cb = firstLst[depth]; cb != NULL; cb = cb->next) {
    up = cb->eBoxUp;
    lo = cb->eBoxLo;
    for (k = 0; k < 3; k++) {
      up[k] = lo[k] = sys->pos[3*cb->chgIdx[0]+k];
    }
    for (i = 1; i < cb->nChgs; i++) {
      for (k = 0; k < 3; k++) {
        double v = sys->pos[3*cb->chgIdx[i]+k];
        up[k] = v > up[k] ? v : up[k];
        lo[k] = v < lo[k] ? v : lo[k];
      }
    }
    cb->x[0] = 0.5*(lo[0]+up[0]);
    cb->x[1] = 0.5*(lo[1]+up[1]);
    cb->x[2] = 0.5*(lo[2]+up[2]);
    cb->eRad = 0.5*sqrt(SQR(up[0]-lo[0])+SQR(up[1]-lo[1])+SQR(up[2]-lo[2]));
  }

  for (lev = depth-1; lev >= sys->height; lev--) {
    for (cb = firstLst[lev]; cb != NULL; cb = cb->next) {
      up = cb->eBoxUp;
      lo = cb->eBoxLo;
      kid = cb->kids[0];
      for (k = 0; k < 3; k++) { up[k] = kid->eBoxUp[k]; lo[k] = kid->eBoxLo[k]; }
      for (i = 1; i < cb->nKids; i++) {
        kid = cb->kids[i];
        for (k = 0; k < 3; k++) {
          up[k] = kid->eBoxUp[k] > up[k] ? kid->eBoxUp[k] : up[k];
          lo[k] = kid->eBoxLo[k] < lo[k] ? kid->eBoxLo[k] : lo[k];
        }
      }
      cb->x[0] = 0.5*(lo[0]+up[0]);
      cb->x[1] = 0.5*(lo[1]+up[1]);
      cb->x[2] = 0.5*(lo[2]+up[2]);
      cb->eRad = 0.5*sqrt(SQR(up[0]-lo[0])+SQR(up[1]-lo[1])+SQR(up[2]-lo[2]));
    }
  }
} /* getChgEnclBoxs */

/*
 * builds sys->chgCubeList: an octree over sys->pos[0..nChar-1], with
 * each leaf's chgIdx listing the charge indices it encloses.
 */
void buildChargeTree(ssystem *sys) {
  double savedMinx, savedMiny, savedMinz;
  double *savedSideLen;
  int savedMaxLev;
  double cminx, cmaxx, cminy, cmaxy, cminz, cmaxz, length;
  cube *topCube, **firstLst = NULL, **lastLst = NULL, **leafOf = NULL;
  int j, lev, nLeaves;
  int *cursor = NULL;
  cube *cb;

  ASSERT(sys->nChar > 0);

  savedMinx = minx; savedMiny = miny; savedMinz = minz;
  savedSideLen = sideLen; savedMaxLev = maxLev;
  /* goDown() only threads firstCube/lastCube for finest-level cubes it
   * newly creates; that chain isn't consumed by linkChgCubes below (it
   * walks cb->kids[] instead), but it must be reset here so goDown()
   * doesn't write into the panel tree's already-linked leaf cubes. */
  firstCube = NULL;
  lastCube = NULL;

  sys->chgDepth = sys->depth;

  cminx = cmaxx = sys->pos[0];
  cminy = cmaxy = sys->pos[1];
  cminz = cmaxz = sys->pos[2];
  for (j = 1; j < sys->nChar; j++) {
    double x = sys->pos[3*j], y = sys->pos[3*j+1], z = sys->pos[3*j+2];
    cminx = MIN(cminx, x); cmaxx = MAX(cmaxx, x);
    cminy = MIN(cminy, y); cmaxy = MAX(cmaxy, y);
    cminz = MIN(cminz, z); cmaxz = MAX(cmaxz, z);
  }

  minx = cminx; miny = cminy; minz = cminz;
  maxLev = sys->chgDepth;

  length = MAX(cmaxx-cminx, cmaxy-cminy);
  length = MAX(cmaxz-cminz, length);
  if (length <= 0.0) length = 1.0; /* guard a degenerate/point-like charge set */

  CALLOC(sideLen, maxLev+1, double);
  for (lev = 0; lev <= maxLev; lev++) {
    sideLen[lev] = length;
    length *= 0.5;
  }

  CALLOC(topCube, 1, cube);
  topCube->level = 0;

  CALLOC(leafOf, sys->nChar, cube*);
  for (j = 0; j < sys->nChar; j++) {
    leafOf[j] = goDown(&sys->pos[3*j], topCube);
    leafOf[j]->nChgs++;
  }

  CALLOC(firstLst, maxLev+1, cube*);
  CALLOC(lastLst, maxLev+1, cube*);
  linkChgCubes(topCube, firstLst, lastLst);
  sys->chgCubeList = firstLst;
  free(lastLst);

  nLeaves = 0;
  for (cb = firstLst[sys->chgDepth]; cb != NULL; cb = cb->next) {
    CALLOC(cb->chgIdx, cb->nChgs, int);
    cb->leafFlatIdx = nLeaves++;
  }
  CALLOC(cursor, nLeaves, int);
  for (j = 0; j < sys->nChar; j++) {
    cube *lf = leafOf[j];
    lf->chgIdx[cursor[lf->leafFlatIdx]++] = j;
  }
  free(cursor);
  free(leafOf);

  getChgEnclBoxs(sys, firstLst);

  free(sideLen);
  minx = savedMinx; miny = savedMiny; minz = savedMinz;
  sideLen = savedSideLen; maxLev = savedMaxLev;
} /* buildChargeTree */

/*
 * direct-sum multipole moment of a point charge about a cube center:
 * setupConVect() already computes exactly the monomial basis
 * m^alpha = (dx)^i1(dy)^i2(dz)^i3 / (i1!i2!i3!) needed here.
 */
static void computeChgLeafMoments(ssystem *sys, cube *cb) {
  int order = rhsChargeExpansionOrder(sys, cb->level);
  int nMom = sys->nMom[order];
  double trns[3];
  double *convVec = NULL;
  int i, k;

  CALLOC(cb->mom_chr, nMom, double);
  CALLOC(convVec, nMom, double);

  for (k = 0; k < cb->nChgs; k++) {
    int j = cb->chgIdx[k];
    trns[0] = sys->pos[3*j]   - cb->x[0];
    trns[1] = sys->pos[3*j+1] - cb->x[1];
    trns[2] = sys->pos[3*j+2] - cb->x[2];
    setupConVect(order, trns, convVec);
    for (i = 0; i < nMom; i++) {
      cb->mom_chr[i] += sys->chr[j]*convVec[i];
    }
  }
  free(convVec);
} /* computeChgLeafMoments */

/*
 * upward translation of a single charge-moment array to a coarser
 * cube's center. Mechanically the same convolution as convM2M() in
 * expan.c, restricted to one array instead of the (pot,dpdn) pair.
 */
static void transM2Mchr(ssystem *sys, cube *cbIn, cube *cbOut) {
  int i, n, i1, i2, i3, k1, k2, k3, j1, j2, j3;
  int ordIn = rhsChargeExpansionOrder(sys, cbIn->level);
  int ordOut = rhsChargeExpansionOrder(sys, cbOut->level);
  int nMomIn = sys->nMom[ordIn];
  double trns[3];

  for (i = 0; i < 3; i++) trns[i] = cbIn->x[i] - cbOut->x[i];

  fcnBuf1[0] = fcnBuf2[0] = fcnBuf3[0] = 1.0;
  for (i = 1; i <= ordOut; i++) {
    fcnBuf1[i] = fcnBuf1[i-1]*trns[0];
    fcnBuf2[i] = fcnBuf2[i-1]*trns[1];
    fcnBuf3[i] = fcnBuf3[i-1]*trns[2];
  }
  for (i = n = 0; n <= ordOut; n++) {
    for (i1 = 0; i1 <= n; i1++) {
      for (i2 = 0; i2 <= n-i1; i2++, i++) {
        i3 = n-i1-i2;
        convVecR[i] = fcnBuf1[i1]*fcnBuf2[i2]*fcnBuf3[i3];
        convVecR[i] *= ifact3[i];
      }
    }
  }

  for (i = n = 0; n <= ordOut; n++) {
    for (i1 = 0; i1 <= n; i1++) {
      for (i2 = 0; i2 <= n-i1; i2++, i++) {
        double tmp = 0.0;
        i3 = n-i1-i2;
        for (k1 = 0; k1 <= i1; k1++) {
          j1 = i1-k1;
          for (k2 = 0; k2 <= i2; k2++) {
            j2 = i2-k2;
            for (k3 = 0; k3 <= i3; k3++) {
              int bIdx;
              j3 = i3-k3;
              bIdx = idx3[k1][k2][k3];
              if (bIdx < nMomIn) {
                tmp += convVecR[idx3[j1][j2][j3]]*cbIn->mom_chr[bIdx];
              }
            }
          }
        }
        cbOut->mom_chr[i] += tmp;
      }
    }
  }
} /* transM2Mchr */

/*
 * computes charge-cluster multipole moments for every cube in the
 * charge tree, from the leaves (direct sum over enclosed charges)
 * up through the coarser levels (translation of children's moments).
 */
void computeChgMoments(ssystem *sys) {
  int lev, order, nMom;
  cube *cb;

  for (cb = sys->chgCubeList[sys->chgDepth]; cb != NULL; cb = cb->next) {
    computeChgLeafMoments(sys, cb);
  }

  for (lev = sys->chgDepth-1; lev >= sys->height; lev--) {
    order = rhsChargeExpansionOrder(sys, lev);
    nMom = sys->nMom[order];
    for (cb = sys->chgCubeList[lev]; cb != NULL; cb = cb->next) {
      int i;
      CALLOC(cb->mom_chr, nMom, double);
      for (i = 0; i < cb->nKids; i++) {
        transM2Mchr(sys, cb->kids[i], cb);
      }
    }
  }
} /* computeChgMoments */
