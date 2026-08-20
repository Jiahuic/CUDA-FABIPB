#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include "gkGlobal.h"
#include "gk.h"
#include "gpu_backend.h"

/* blas: matrix times vector */
void dgemv_(char *tr, int *m, int *n, double *alpha, double *A, int *lda,
          double *x, int *incx, double *beta, double *y, int *incy);

void setupDerivs(int order, double *x );
double *panelPotential(int qOrder, double *r, panel *pnl);
void transM2M(ssystem *sys, cube *cbIn, cube *cbOut);
void kernelRHS(double *x, double *y);
int rhsChargeExpansionOrder(ssystem *sys, int level);

extern double **dG0;     /* workspace for setupDerivs */
extern double **dGk;     /* workspace for setupDerivs */
extern double kappa;
extern double epsilon;
extern void (*kernel)();
extern void (*kernelD)(double r, int p, double *G0, double *Gk);
extern double **Q2M0, **Q2M1;   /* moments */
extern double *ifact3;
extern int *sgn3;
extern int ***idx3;
extern double fourPiI;
extern double **tLegA, **wLegA;

double rhsTreeTheta(void) {
  static int initialized = 0;
  static double theta = 0.3;

  if (!initialized) {
    const char *env = getenv("FABIPB_RHS_TREE_THETA");
    initialized = 1;
    if (env != NULL) {
      char *endptr = NULL;
      double value;

      errno = 0;
      value = strtod(env, &endptr);
      if (errno == 0 && endptr != env && *endptr == '\0' &&
          value > 0.0 && value <= 1.0) {
        theta = value;
      } else {
        fprintf(stderr,
                "Warning: ignoring invalid FABIPB_RHS_TREE_THETA='%s'; using 0.3\n",
                env);
      }
    }
  }
  return theta;
}

/*
 * Acceptance ratio for the post-solve panel-charge energy treecode.
 *
 * This is deliberately separate from rhsTreeTheta() so accuracy/runtime can be
 * tuned independently. For the virus-scale production runs we use 0.3 by
 * default: it keeps the post-solve energy evaluator from dominating runtime,
 * while the observed H1N1/ZIKV energy drift remains within the accepted
 * capsid-scale tolerance. Set FABIPB_ENERGY_TREE_THETA=0.2 for the stricter
 * historical setting.
 */
double energyTreeTheta(void) {
  static int initialized = 0;
  static double theta = 0.3;

  if (!initialized) {
    const char *env = getenv("FABIPB_ENERGY_TREE_THETA");
    initialized = 1;
    if (env != NULL) {
      char *endptr = NULL;
      double value;

      errno = 0;
      value = strtod(env, &endptr);
      if (errno == 0 && endptr != env && *endptr == '\0' &&
          value > 0.0 && value <= 1.0) {
        theta = value;
      } else {
        fprintf(stderr,
                "Warning: ignoring invalid FABIPB_ENERGY_TREE_THETA='%s'; using 0.3\n",
                env);
      }
    }
  }
  return theta;
}

void initRhsTreeWorkspace(ssystem *sys, RhsTreeWorkspace *ws) {
  int k, p;

  if (ws == NULL) {
    return;
  }
  memset(ws, 0, sizeof(*ws));
  if (sys == NULL) {
    return;
  }

  ws->maxOrder = sys->maxOrder;
  ws->derivOrder = sys->maxOrder + 1;
  CALLOC(ws->gvals0, ws->derivOrder + 1, double);
  CALLOC(ws->gvalsk, ws->derivOrder + 1, double);
  CALLOC(ws->dg0, ws->derivOrder + 1, double*);
  CALLOC(ws->dgk, ws->derivOrder + 1, double*);
  for (k = 0; k <= ws->derivOrder; k++) {
    p = ws->derivOrder - k;
    CALLOC(ws->dg0[k], sys->nMom[p], double);
    CALLOC(ws->dgk[k], sys->nMom[p], double);
  }
}

void freeRhsTreeWorkspace(RhsTreeWorkspace *ws) {
  int k;

  if (ws == NULL) {
    return;
  }
  if (ws->dg0 != NULL) {
    for (k = 0; k <= ws->derivOrder; k++) {
      free(ws->dg0[k]);
    }
    free(ws->dg0);
  }
  if (ws->dgk != NULL) {
    for (k = 0; k <= ws->derivOrder; k++) {
      free(ws->dgk[k]);
    }
    free(ws->dgk);
  }
  free(ws->gvals0);
  free(ws->gvalsk);
  memset(ws, 0, sizeof(*ws));
}

/*
 * Thread-safe setupDerivs: identical recurrence, but the workspace is passed
 * in rather than taken from the file-scope dG0/dGk. Exposed because the M2L
 * streaming path needs to run this concurrently across threads.
 */
void setupDerivsWorkspace(ssystem *sys, RhsTreeWorkspace *ws,
                                     int order, const double *x) {
  int p, p1, iRow, iRow1, i1, i2, i3, idx, idx1, idx2;
  double r;

  ASSERT(sys != NULL && ws != NULL);
  ASSERT(order >= 0 && order <= ws->derivOrder);

  r = sqrt(SQR(x[0]) + SQR(x[1]) + SQR(x[2]));
  kernelD(r, order, ws->gvals0, ws->gvalsk);

  for (p = 0; p <= order; p++) {
    ws->dg0[p][0] = ws->gvals0[p];
    ws->dgk[p][0] = ws->gvalsk[p];
  }

  for (p = 0; p < order; p++) {
    p1 = p + 1;
    ws->dg0[p][1] = ws->dg0[p1][0] * x[2];
    ws->dg0[p][2] = ws->dg0[p1][0] * x[1];
    ws->dg0[p][3] = ws->dg0[p1][0] * x[0];
    ws->dgk[p][1] = ws->dgk[p1][0] * x[2];
    ws->dgk[p][2] = ws->dgk[p1][0] * x[1];
    ws->dgk[p][3] = ws->dgk[p1][0] * x[0];
  }

  for (iRow = 2; iRow <= order; iRow++) {
    for (p = 0; p <= order - iRow; p++) {
      p1 = p + 1;
      idx = idx3[0][0][iRow];
      iRow1 = iRow - 1;

      idx1 = idx3[0][0][iRow1];
      idx2 = idx3[0][0][iRow - 2];
      ws->dg0[p][idx] = ws->dg0[p1][idx1] * x[2] + ws->dg0[p1][idx2] * iRow1;
      ws->dgk[p][idx] = ws->dgk[p1][idx1] * x[2] + ws->dgk[p1][idx2] * iRow1;
      idx++;

      idx1 = idx3[0][0][iRow1];
      ws->dg0[p][idx] = ws->dg0[p1][idx1] * x[1];
      ws->dgk[p][idx] = ws->dgk[p1][idx1] * x[1];
      idx++;

      for (i2 = 2; i2 <= iRow; i2++, idx++) {
        i3 = iRow - i2;
        idx1 = idx3[0][i2 - 1][i3];
        idx2 = idx3[0][i2 - 2][i3];
        ws->dg0[p][idx] = ws->dg0[p1][idx1] * x[1] + ws->dg0[p1][idx2] * (i2 - 1);
        ws->dgk[p][idx] = ws->dgk[p1][idx1] * x[1] + ws->dgk[p1][idx2] * (i2 - 1);
      }

      for (i2 = 0; i2 <= iRow1; i2++, idx++) {
        i3 = iRow1 - i2;
        idx1 = idx3[0][i2][i3];
        ws->dg0[p][idx] = ws->dg0[p1][idx1] * x[0];
        ws->dgk[p][idx] = ws->dgk[p1][idx1] * x[0];
      }

      for (i1 = 2; i1 <= iRow; i1++) {
        for (i2 = 0; i2 <= iRow - i1; i2++, idx++) {
          i3 = iRow - i1 - i2;
          idx1 = idx3[i1 - 1][i2][i3];
          idx2 = idx3[i1 - 2][i2][i3];
          ws->dg0[p][idx] = ws->dg0[p1][idx1] * x[0] + ws->dg0[p1][idx2] * (i1 - 1);
          ws->dgk[p][idx] = ws->dgk[p1][idx1] * x[0] + ws->dgk[p1][idx2] * (i1 - 1);
        }
      }
    }
  }
}

static void setupCoulombDerivsLocal(ssystem *sys, RhsTreeWorkspace *ws,
                                    int order, const double *x) {
  double r, r2;
  int p, p1, iRow, iRow1, i1, i2, i3, idx, idx1, idx2, k;

  ASSERT(sys != NULL && ws != NULL);
  ASSERT(order >= 0 && order <= ws->derivOrder);

  r = sqrt(SQR(x[0]) + SQR(x[1]) + SQR(x[2]));
  ws->gvals0[0] = fourPiI / r;
  r2 = -1.0 / (r * r);
  for (k = 0; k < order; k++) {
    ws->gvals0[k + 1] = (2 * k + 1) * r2 * ws->gvals0[k];
  }

  for (p = 0; p <= order; p++) {
    ws->dg0[p][0] = ws->gvals0[p];
  }

  for (p = 0; p < order; p++) {
    p1 = p + 1;
    ws->dg0[p][1] = ws->dg0[p1][0] * x[2];
    ws->dg0[p][2] = ws->dg0[p1][0] * x[1];
    ws->dg0[p][3] = ws->dg0[p1][0] * x[0];
  }

  for (iRow = 2; iRow <= order; iRow++) {
    for (p = 0; p <= order - iRow; p++) {
      p1 = p + 1;
      idx = idx3[0][0][iRow];
      iRow1 = iRow - 1;

      idx1 = idx3[0][0][iRow1];
      idx2 = idx3[0][0][iRow - 2];
      ws->dg0[p][idx] = ws->dg0[p1][idx1] * x[2] + ws->dg0[p1][idx2] * iRow1;
      idx++;

      idx1 = idx3[0][0][iRow1];
      ws->dg0[p][idx] = ws->dg0[p1][idx1] * x[1];
      idx++;

      for (i2 = 2; i2 <= iRow; i2++, idx++) {
        i3 = iRow - i2;
        idx1 = idx3[0][i2 - 1][i3];
        idx2 = idx3[0][i2 - 2][i3];
        ws->dg0[p][idx] = ws->dg0[p1][idx1] * x[1] + ws->dg0[p1][idx2] * (i2 - 1);
      }

      for (i2 = 0; i2 <= iRow1; i2++, idx++) {
        i3 = iRow1 - i2;
        idx1 = idx3[0][i2][i3];
        ws->dg0[p][idx] = ws->dg0[p1][idx1] * x[0];
      }

      for (i1 = 2; i1 <= iRow; i1++) {
        for (i2 = 0; i2 <= iRow - i1; i2++, idx++) {
          i3 = iRow - i1 - i2;
          idx1 = idx3[i1 - 1][i2][i3];
          idx2 = idx3[i1 - 2][i2][i3];
          ws->dg0[p][idx] = ws->dg0[p1][idx1] * x[0] + ws->dg0[p1][idx2] * (i1 - 1);
        }
      }
    }
  }
}

static void chgClusterEvalLocal(ssystem *sys, cube *chgCb, panel *pnlX,
                                RhsTreeWorkspace *ws, double *y) {
  int order = rhsChargeExpansionOrder(sys, chgCb->level);
  int nMom = sys->nMom[order];
  double *mom = chgCb->mom_chr;
  double *dg = ws->dg0[0];
  double *nrm = pnlX->normal;
  double y0 = 0.0, y1 = 0.0;
  int i, i1, i2, i3, n;

  ASSERT(chgCb->level >= 0 && chgCb->level <= sys->chgDepth);
  ASSERT(order >= 0 && order <= sys->maxOrder);
  ASSERT(nMom > 0 && nMom < 1000000);
  ASSERT(mom != NULL);

  for (i = n = 0; n <= order; n++) {
    for (i1 = 0; i1 <= n; i1++) {
      for (i2 = 0; i2 <= n - i1; i2++, i++) {
        double dnDeriv;
        i3 = n - i1 - i2;
        y0 += sgn3[i] * mom[i] * dg[i];
        dnDeriv = nrm[0] * dg[idx3[i1 + 1][i2][i3]]
                + nrm[1] * dg[idx3[i1][i2 + 1][i3]]
                + nrm[2] * dg[idx3[i1][i2][i3 + 1]];
        y1 += sgn3[i] * mom[i] * dnDeriv;
      }
    }
  }
  y[0] = y0 / fourPiI;
  y[1] = y1 / fourPiI;
}

static void rhsTreeWalkLocal(ssystem *sys, cube *chgCb, const double *quadPt,
                             panel *pnlX, RhsTreeWorkspace *ws, double *y) {
  double theta = rhsTreeTheta();
  double dist, r[3], yFar[2];
  int k, i;

  for (k = 0; k < 3; k++) r[k] = quadPt[k] - chgCb->x[k];
  dist = sqrt(SQR(r[0]) + SQR(r[1]) + SQR(r[2]));

  if (chgCb->eRad < theta * dist && chgCb->level >= sys->height) {
    setupCoulombDerivsLocal(sys, ws, rhsChargeExpansionOrder(sys, chgCb->level) + 1, r);
    chgClusterEvalLocal(sys, chgCb, pnlX, ws, yFar);
    y[0] += yFar[0];
    y[1] += yFar[1];
    return;
  }

  if (chgCb->level == sys->chgDepth) {
    for (i = 0; i < chgCb->nChgs; i++) {
      int j = chgCb->chgIdx[i];
      double x[3], r2, ri, r3i, ip;
      for (k = 0; k < 3; k++) x[k] = quadPt[k] - sys->pos[3 * j + k];
      r2 = SQR(x[0]) + SQR(x[1]) + SQR(x[2]);
      ri = 1.0 / sqrt(r2);
      r3i = ri / r2;
      ip = pnlX->normal[0] * x[0] + pnlX->normal[1] * x[1] + pnlX->normal[2] * x[2];
      y[0] += sys->chr[j] * ri;
      y[1] += sys->chr[j] * (-ip * r3i);
    }
    return;
  }

  for (i = 0; i < chgCb->nKids; i++) {
    rhsTreeWalkLocal(sys, chgCb->kids[i], quadPt, pnlX, ws, y);
  }
}

void panelRHSTreeWorkspace(ssystem *sys, int qOrder, panel *pnlX, cube *chgRoot,
                           RhsTreeWorkspace *ws, double out[2]) {
  int ix, jx, k;
  double r0[3], r[3], *ax2, *ax0;
  double *tLeg, *wLeg;
  double y[2];

  tLeg = tLegA[qOrder];
  wLeg = wLegA[qOrder];
  ax2 = pnlX->a[2];
  ax0 = pnlX->a[0];

  for (k = 0; k < 3; k++) {
    r0[k] = pnlX->vtx[0][k];
  }

  out[0] = 0.0;
  out[1] = 0.0;
  for (ix = 0; ix < qOrder; ix++) {
    for (jx = 0; jx < qOrder; jx++) {
      for (k = 0; k < 3; k++) {
        r[k] = r0[k] + tLeg[ix] * (ax2[k] + tLeg[jx] * ax0[k]);
      }
      y[0] = 0.0;
      y[1] = 0.0;
      rhsTreeWalkLocal(sys, chgRoot, r, pnlX, ws, y);
      out[0] += y[0] * tLeg[ix] * wLeg[ix] * wLeg[jx];
      out[1] += y[1] * tLeg[ix] * wLeg[ix] * wLeg[jx];
    }
  }
  out[0] *= 2.0 * pnlX->area;
  out[1] *= 2.0 * pnlX->area;
}

static void partclusterComponents(ssystem *sys, double *G0, double *Gk,
                                  cube *cb, double *pot0, double *pot1) {
  int i, k1, k2, k3, k, j;
  int order = sys->ordM2L[cb->level];
  int nMom  = sys->nMom[order];
  double tmp, tmp1, tmp2;
  double *mom_pot, *mom_dpdn, *lec_k1, *lec_k2;

  mom_pot = cb->mom_pot; mom_dpdn = cb->mom_dpdn;
  for ( tmp1=tmp2=0.,i=0; i<nMom; i++ ) {
    tmp = sgn3[i];
    tmp1 += tmp*mom_dpdn[i]*(G0[i]-Gk[i]);
    tmp2 += tmp*mom_pot[i] *(epsilon*Gk[i]-G0[i]);
    //printf("%f %f %f %e %e\n",ifact3[i],mom_dpdn[i],mom_pot[i],G0[i],Gk[i]);
  }
  *pot0 = tmp2;
  *pot1 = tmp1;
}

double partcluster( ssystem *sys, double *G0, double *Gk, cube *cb ) {
  double pot0, pot1;
  partclusterComponents(sys, G0, Gk, cb, &pot0, &pot1);
  return pot0 + pot1;
}

double Treecode( ssystem *sys, cube *cb, double *pos, double *sgm ) {
  int lev = cb->level, order=sys->ordM2L[lev];
  int depth=sys->depth, height=sys->height;
  int nPnls=sys->nPnls, qOrder=sys->maxQuadOrder;
  int i, k;
  double theta = sys->maxSepRatio;
  double dist, pot=0., r[3], *intgr;
  cube *cbKid;
  panel *pnl;
  theta = 0.2;

  for (k=0; k<3; k++) r[k] = pos[k]-cb->x[k];
  dist = sqrt(SQR(r[0])+SQR(r[1])+SQR(r[2]));

  if ( cb->eRad < theta*dist & cb->level>=height ) {
    //printf("target: %f %f %f\n", pos[0], pos[1], pos[2]);
    //printf("cubect: %f %f %f\n", cb->x[0], cb->x[1], cb->x[2]);
    //printf("dist: %f, radius: %f\n", dist, cb->eRad);
    //printf("lev=%d (i,j,k)=(%d,%d,%d)\n", lev, cb->i, cb->j, cb->k);
    //printf("#pnls: %d\n", cb->nPnls);

    //double pot1 = 0.;
    //for ( i=0,pnl=cb->pnls; i<cb->nPnls; i++,pnl=pnl->nextC ) {
    //  intgr = panelPotential(qOrder, pos, pnl);
    //  pot1 += intgr[0]*sgm[pnl->idx+sys->nPnls]+intgr[1]*sgm[pnl->idx];
    //}

    // particle-cluster interaction
    setupDerivs(order, r);
    pot = partcluster(sys, dG0[0], dGk[0], cb);
    //printf("%e %e\n\n",pot1, pot);
    //exit(0);
    //if ( fabs(pot1-pot) > 1e-4 ) exit(0);
    return pot;
  } else {
    if ( cb->level == sys->depth ) {
      //printf("cube lvl: %d, pnl #:%d\n", cb->level, cb->nPnls);
      // direct summation
      for ( i=0, pnl=cb->pnls; i<cb->nPnls; i++,pnl=pnl->nextC ) {
        intgr = panelPotential(qOrder, pos, pnl);
        pot += intgr[0]*sgm[pnl->idx+sys->nPnls]+intgr[1]*sgm[pnl->idx];
      }
      return pot;
    } else {
      for ( i=0; i<cb->nKids; i++ ) {
        cbKid = cb->kids[i];
        pot += Treecode(sys, cbKid, pos, sgm);
      }
      return pot;
    }
  }
}

static void buildTreecodeMoments(ssystem *sys, double *sgm) {
  cube *cb;
  int depth=sys->depth, height=sys->height, nPnls=sys->nPnls;
  int nKid, idx, nMom, lev, k, n, inc=1;
  double *x, *y;

  for (lev=depth; lev>=height; lev--) {
    nMom = sys->nMom[sys->ordMom[lev]];
    for (cb=sys->cubeList[lev]; cb != NULL; cb=cb->next) {
      for (k=0; k<nMom; k++) {
        cb->mom_pot[k] = 0.;
        cb->mom_dpdn[k] = 0.;
      }
    }
  }

  nMom = sys->nMom[sys->ordMom[depth]];
  for (idx=0, cb=sys->cubeList[depth]; cb != NULL; cb=cb->next, idx++) {
    x = &(sgm[cb->pnls->idx]);
    n = cb->nPnls;
    y = cb->mom_pot;
    dgemv_(&nChr, &nMom, &n, &one, Q2M1[idx], &nMom, x, &inc, &one, y, &inc);
    x = &(sgm[cb->pnls->idx+nPnls]);
    y = cb->mom_dpdn;
    dgemv_(&nChr, &nMom, &n, &one, Q2M0[idx], &nMom, x, &inc, &one, y, &inc);
  }
  for (lev=depth-1; lev>=height; lev--) {
    for (cb=sys->cubeList[lev]; cb != NULL; cb=cb->next) {
      for (nKid=0; nKid<cb->nKids; nKid++) {
        transM2M(sys, cb->kids[nKid], cb);
      }
    }
  }
}

/*
 * this subroutine apply the particle-cluster interaction
 * work on source term and solvation energy
 * input should specify the kernels (can't apply it on RHS)
*/
//void applyTreecode( ssystem *sys, double *sgm, double *pot, void (*kernelType)() ) {
void applyTreecode( ssystem *sys, double *sgm, double *pot ) {
  cube *Topcb=sys->cubeList[0];
  int i;

  buildTreecodeMoments(sys, sgm);
  for (*pot=0., i=0; i<sys->nChar; i++) {
    *pot += sys->chr[i]*Treecode(sys, Topcb, &sys->pos[3*i], sgm);
  }
}

static void chgClusterEnergyEval(ssystem *sys, cube *chgCb, panel *pnlX,
                                 RhsTreeWorkspace *ws, double *y) {
  int order = rhsChargeExpansionOrder(sys, chgCb->level);
  int nMom = sys->nMom[order];
  double *mom = chgCb->mom_chr;
  double *nrm = pnlX->normal;
  double *dg0 = ws->dg0[0];
  double *dgk = ws->dgk[0];
  double y0 = 0.0, y1 = 0.0;
  int i, i1, i2, i3, n;

  ASSERT(chgCb->level >= 0 && chgCb->level <= sys->chgDepth);
  ASSERT(order >= 0 && order <= sys->maxOrder);
  ASSERT(nMom > 0 && nMom < 1000000);
  ASSERT(mom != NULL);

  for (i = n = 0; n <= order; n++) {
    for (i1 = 0; i1 <= n; i1++) {
      for (i2 = 0; i2 <= n - i1; i2++, i++) {
        double dnG0, dnGk;
        i3 = n - i1 - i2;
        y0 += sgn3[i] * mom[i] * (dg0[i] - dgk[i]);
        dnG0 = nrm[0] * dg0[idx3[i1 + 1][i2][i3]]
             + nrm[1] * dg0[idx3[i1][i2 + 1][i3]]
             + nrm[2] * dg0[idx3[i1][i2][i3 + 1]];
        dnGk = nrm[0] * dgk[idx3[i1 + 1][i2][i3]]
             + nrm[1] * dgk[idx3[i1][i2 + 1][i3]]
             + nrm[2] * dgk[idx3[i1][i2][i3 + 1]];
        y1 += sgn3[i] * mom[i] * (epsilon * dnGk - dnG0);
      }
    }
  }
  y[0] = y0;
  y[1] = y1;
}

static void energyTreeWalk(ssystem *sys, cube *chgCb, double *quadPt,
                           panel *pnlX, RhsTreeWorkspace *ws, double *y) {
  double theta = energyTreeTheta();
  double dist, r[3], yFar[2];
  int k, i;

  for (k = 0; k < 3; k++) {
    r[k] = quadPt[k] - chgCb->x[k];
  }
  dist = sqrt(SQR(r[0]) + SQR(r[1]) + SQR(r[2]));

  if (chgCb->eRad < theta * dist && chgCb->level >= sys->height) {
    setupDerivsWorkspace(sys, ws, rhsChargeExpansionOrder(sys, chgCb->level) + 1, r);
    chgClusterEnergyEval(sys, chgCb, pnlX, ws, yFar);
    y[0] += yFar[0];
    y[1] += yFar[1];
    return;
  }

  if (chgCb->level == sys->chgDepth) {
    double *nrm = pnlX->normal;
    for (i = 0; i < chgCb->nChgs; i++) {
      int j = chgCb->chgIdx[i];
      double x[3], r2, rnorm, G0, Gk, coef, ip, dG0dn, dGkdn;
      for (k = 0; k < 3; k++) {
        x[k] = quadPt[k] - sys->pos[3 * j + k];
      }
      r2 = SQR(x[0]) + SQR(x[1]) + SQR(x[2]);
      rnorm = sqrt(r2);
      G0 = fourPiI / rnorm;
      Gk = exp(-kappa * rnorm) * G0;
      coef = (kappa * rnorm + 1.0) * exp(-kappa * rnorm);
      ip = nrm[0] * x[0] + nrm[1] * x[1] + nrm[2] * x[2];
      dG0dn = -ip * G0 / r2;
      dGkdn = coef * dG0dn;
      y[0] += sys->chr[j] * (G0 - Gk);
      y[1] += sys->chr[j] * (epsilon * dGkdn - dG0dn);
    }
    return;
  }

  for (i = 0; i < chgCb->nKids; i++) {
    energyTreeWalk(sys, chgCb->kids[i], quadPt, pnlX, ws, y);
  }
}

static void panelEnergyTree(ssystem *sys, int qOrder, panel *pnlX,
                            cube *chgRoot, RhsTreeWorkspace *ws, double out[2]) {
  int ix, jx, k;
  double r0[3], r[3], *ax2, *ax0;
  double *tLeg, *wLeg;
  double y[2];

  tLeg = tLegA[qOrder];
  wLeg = wLegA[qOrder];
  ax2 = pnlX->a[2];
  ax0 = pnlX->a[0];

  for (k = 0; k < 3; k++) {
    r0[k] = pnlX->vtx[0][k];
  }

  out[0] = 0.0;
  out[1] = 0.0;
  for (ix = 0; ix < qOrder; ix++) {
    for (jx = 0; jx < qOrder; jx++) {
      for (k = 0; k < 3; k++) {
        r[k] = r0[k] + tLeg[ix] * (ax2[k] + tLeg[jx] * ax0[k]);
      }
      y[0] = 0.0;
      y[1] = 0.0;
      energyTreeWalk(sys, chgRoot, r, pnlX, ws, y);
      out[0] += y[0] * tLeg[ix] * wLeg[ix] * wLeg[jx];
      out[1] += y[1] * tLeg[ix] * wLeg[ix] * wLeg[jx];
    }
  }
  out[0] *= 2.0 * pnlX->area;
  out[1] *= 2.0 * pnlX->area;
}

typedef struct {
  ssystem *sys;
  panel **panels;
  const double *sgm;
  int begin;
  int end;
  double sum;
} PanelEnergyTask;

static int energyThreadCount(int nTasks) {
  const char *env = getenv("FABIPB_ENERGY_THREADS");
  long hc;
  int threads;

  if (env != NULL && env[0] != '\0') {
    threads = atoi(env);
  } else {
    hc = sysconf(_SC_NPROCESSORS_ONLN);
    threads = (hc > 0) ? (int)hc : 1;
  }
  if (threads < 1) {
    threads = 1;
  }
  if (threads > nTasks) {
    threads = nTasks;
  }
  if (threads > 128) {
    threads = 128;
  }
  return threads;
}

static panel **panelEnergyArray(ssystem *sys, int *owned) {
  panel **panels;
  panel *pnl;
  int i;

  *owned = 0;
  if (sys->panelByIdx != NULL) {
    return sys->panelByIdx;
  }

  CALLOC(panels, sys->nPnls, panel*);
  for (i = 0, pnl = sys->pnlLst; pnl != NULL && i < sys->nPnls; pnl = pnl->nextC, i++) {
    panels[i] = pnl;
  }
  ASSERT(i == sys->nPnls);
  *owned = 1;
  return panels;
}

static void *panelEnergyWorker(void *arg) {
  PanelEnergyTask *task = (PanelEnergyTask *)arg;
  ssystem *sys = task->sys;
  RhsTreeWorkspace ws;
  double local = 0.0;
  double intgr[2];
  int i;

  initRhsTreeWorkspace(sys, &ws);
  for (i = task->begin; i < task->end; i++) {
    panel *pnl = task->panels[i];
    panelEnergyTree(sys, sys->maxQuadOrder, pnl, sys->chgCubeList[0], &ws, intgr);
    local += intgr[0] * task->sgm[pnl->idx + sys->nPnls] +
             intgr[1] * task->sgm[pnl->idx];
  }
  freeRhsTreeWorkspace(&ws);
  task->sum = local;
  return NULL;
}

void applyPanelChargeTreeEnergy( ssystem *sys, double *sgm, double *pot ) {
  int ownPanels = 0;
  int nThreads, t, created = 0, failed = 0;
  panel **panels;
  PanelEnergyTask *tasks;
  pthread_t *threads;

  ASSERT(sys->chgCubeList != NULL && sys->chgCubeList[0] != NULL);

  *pot = 0.0;
  if (sys->nPnls <= 0) {
    return;
  }

  /*
   * Try the device first. It walks the same tree with the same acceptance
   * rule, one warp per panel; if it is unavailable, or the quadrature order is
   * not the single-point rule it assumes, it reports failure and the threaded
   * CPU evaluator below runs instead.
   */
  {
    /* FABIPB_ENERGY_GPU=0 forces the CPU evaluator while leaving the rest of
     * the solve on the GPU, so the two evaluators can be compared on one
     * identical solution vector. */
    const char *envGpu = getenv("FABIPB_ENERGY_GPU");
    int wantGpu = !(envGpu != NULL && atoi(envGpu) == 0);
    if (wantGpu && sys->gpuMode > 0) {
      gpuReleaseMatvecCaches();
      if (gpuPanelChargeTreeEnergy(sys, sgm, pot)) {
        return;
      }
      if (sys->benchmarkMode > 0) {
        printf("GPU panel-tree energy unavailable: %s; using CPU fallback.\n",
               gpuNearfieldLastError());
      }
    }
  }

  panels = panelEnergyArray(sys, &ownPanels);
  nThreads = energyThreadCount(sys->nPnls);
  if (sys->benchmarkMode > 0) {
    printf("panel-tree energy evaluator: threads=%d panels=%d theta=%g\n",
           nThreads, sys->nPnls, energyTreeTheta());
  }

  CALLOC(tasks, nThreads, PanelEnergyTask);
  CALLOC(threads, nThreads, pthread_t);
  for (t = 0; t < nThreads; t++) {
    int begin = (int)(((long long)sys->nPnls * t) / nThreads);
    int end = (int)(((long long)sys->nPnls * (t + 1)) / nThreads);
    tasks[t].sys = sys;
    tasks[t].panels = panels;
    tasks[t].sgm = sgm;
    tasks[t].begin = begin;
    tasks[t].end = end;
  }

  if (nThreads == 1) {
    panelEnergyWorker(&tasks[0]);
    created = 1;
  } else {
    for (t = 0; t < nThreads; t++) {
      if (pthread_create(&threads[t], NULL, panelEnergyWorker, &tasks[t]) != 0) {
        failed = 1;
        break;
      }
      created++;
    }
    for (t = 0; t < created; t++) {
      pthread_join(threads[t], NULL);
    }
    if (failed) {
      fprintf(stderr,
              "Warning: pthread_create failed for panel-tree energy; recomputing serially\n");
      tasks[0].begin = 0;
      tasks[0].end = sys->nPnls;
      panelEnergyWorker(&tasks[0]);
      created = 1;
    }
  }

  for (t = 0; t < created; t++) {
    *pot += tasks[t].sum;
  }

  free(threads);
  free(tasks);
  if (ownPanels) {
    free(panels);
  }
}

/*
 * evaluates a charge cluster's multipole expansion (built by
 * computeChgMoments() in chargeTree.c) at a target panel quadrature
 * point, producing the same two RHS quantities as kernelRHS(): the
 * potential and its derivative in the panel's own normal direction.
 * Caller must have already called setupDerivs(order, r) for the
 * relative vector r = quadPt - chgCb->x.
 *
 * kernelDC0() (used by setupDerivs -> dG0) bakes fourPiI into G[0],
 * while kernelRHS() (used by the near-field direct sum below) does
 * not; dividing by fourPiI here keeps both branches on the same
 * scale, since setupRHS applies fourPiI/epsilon1 once at the end.
 */
static void chgClusterEval(ssystem *sys, cube *chgCb, panel *pnlX, double *y) {
  int order = rhsChargeExpansionOrder(sys, chgCb->level);
  int nMom = sys->nMom[order];
  double *mom = chgCb->mom_chr;
  double *nrm = pnlX->normal;
  double y0 = 0.0, y1 = 0.0;
  int i, i1, i2, i3, n;

  ASSERT(chgCb->level >= 0 && chgCb->level <= sys->chgDepth);
  ASSERT(order >= 0 && order <= sys->maxOrder);
  ASSERT(nMom > 0 && nMom < 1000000);
  ASSERT(mom != NULL);

  /*
   * y0 = Phi(target) = sum_alpha sgn3[alpha]*mom_chr[alpha]*D^alpha_r G(r),
   * the standard multipole-to-local evaluation (mirrors partcluster()'s
   * combination of mom_pot/mom_dpdn with G0/Gk above, minus the
   * screened-Coulomb term since RHS charges are pure point monopoles).
   *
   * y1 = d/dn_target Phi(target) = n . grad_target Phi(target). Since
   * D^alpha_r G is already a derivative w.r.t. r = target - center,
   * grad_target D^alpha_r G = D^(alpha+e_k)_r G for each axis k -- one
   * order higher in dG0, NOT a calcOneNoment-style reweighting of the
   * moments (that function instead builds a *source*'s own dipole
   * moments from its monopole moments via its own normal, which is a
   * different operation from a target-side directional derivative of
   * the evaluated field).
   */
  for (i = n = 0; n <= order; n++) {
    for (i1 = 0; i1 <= n; i1++) {
      for (i2 = 0; i2 <= n-i1; i2++, i++) {
        double dnDeriv;
        i3 = n-i1-i2;
        y0 += sgn3[i]*mom[i]*dG0[0][i];
        dnDeriv = nrm[0]*dG0[0][idx3[i1+1][i2][i3]]
                + nrm[1]*dG0[0][idx3[i1][i2+1][i3]]
                + nrm[2]*dG0[0][idx3[i1][i2][i3+1]];
        y1 += sgn3[i]*mom[i]*dnDeriv;
      }
    }
  }
  y[0] = y0/fourPiI;
  y[1] = y1/fourPiI;
} /* chgClusterEval */

/*
 * particle(target quad point)-cluster(charge tree) walk: the mirror
 * image of Treecode() above, which walks the panel tree per charge
 * to get the solvation energy. Here we walk the charge tree
 * (built by buildChargeTree()/computeChgMoments() in chargeTree.c)
 * per panel quadrature point to accumulate the RHS contribution of
 * every charge, near or far, into y[0] (potential) and y[1] (its
 * derivative in the panel's normal direction). y must be zeroed by
 * the caller before the top-level call.
 */
void rhsTreeWalk(ssystem *sys, cube *chgCb, double *quadPt, panel *pnlX, double *y) {
  /*
   * A particle(single target point)-cluster evaluation needs a
   * tighter admissibility ratio than panel-panel M2L: at
   * sys->maxSepRatio (0.8, tuned for applyFMM()'s panel-cluster to
   * panel-cluster interactions), direct-reference sweeps showed unacceptable
   * error at the FMM separation ratio. With the charge tree's minimum
   * fourth-order expansion, 0.2 keeps both RHS components below 0.1%
   * relative L2 error on the scale-1 1a63 reference while preserving
   * far-cluster acceptance for virus-scale inputs.
   */
  double theta = rhsTreeTheta();
  double dist, r[3], yFar[2];
  int k, i;

  for (k = 0; k < 3; k++) r[k] = quadPt[k]-chgCb->x[k];
  dist = sqrt(SQR(r[0])+SQR(r[1])+SQR(r[2]));

  if (chgCb->eRad < theta*dist && chgCb->level >= sys->height) {
    setupDerivs(rhsChargeExpansionOrder(sys, chgCb->level) + 1, r);
    chgClusterEval(sys, chgCb, pnlX, yFar);
    y[0] += yFar[0];
    y[1] += yFar[1];
    return;
  }

  if (chgCb->level == sys->chgDepth) {
    for (i = 0; i < chgCb->nChgs; i++) {
      int j = chgCb->chgIdx[i];
      double x[3], fcn[2];
      for (k = 0; k < 3; k++) x[k] = quadPt[k]-sys->pos[3*j+k];
      kernelRHS(x, fcn);
      y[0] += sys->chr[j]*fcn[0];
      y[1] += sys->chr[j]*fcn[1];
    }
    return;
  }

  for (i = 0; i < chgCb->nKids; i++) {
    rhsTreeWalk(sys, chgCb->kids[i], quadPt, pnlX, y);
  }
} /* rhsTreeWalk */
