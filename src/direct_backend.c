#include "direct_backend.h"
#include "gk.h"

#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

extern double *panelIA0(panel *pnlX, panel *pnlY);
extern void kernelKER4(double *x, double *y);
extern void (*kernel)(double *x, double *y);

typedef struct {
  const ssystem *sys;
  int nPnls;
  long long nInteractions;
  double *k0;
  double *k1;
  double *k2;
  double *k3;
} CpuDirectCache;

static CpuDirectCache gCpuDirect = {0};

static void freeCpuDirectCache(void) {
  free(gCpuDirect.k0);
  free(gCpuDirect.k1);
  free(gCpuDirect.k2);
  free(gCpuDirect.k3);
  gCpuDirect.k0 = NULL;
  gCpuDirect.k1 = NULL;
  gCpuDirect.k2 = NULL;
  gCpuDirect.k3 = NULL;
  gCpuDirect.sys = NULL;
  gCpuDirect.nPnls = 0;
  gCpuDirect.nInteractions = 0;
}

static int buildCpuDirectTables(const ssystem *sys) {
  int i, j;
  long long nInteractions;
  size_t coeffBytes;
  size_t totalCoeffBytes;

  nInteractions = (long long)sys->nPnls * (long long)sys->nPnls;
  if (nInteractions <= 0) {
    return 0;
  }
  if ((unsigned long long)nInteractions > (unsigned long long)(SIZE_MAX / sizeof(double))) {
    return 0;
  }

  coeffBytes = (size_t)nInteractions * sizeof(double);
  totalCoeffBytes = 4U * coeffBytes;

  gCpuDirect.k0 = (double *)malloc(coeffBytes);
  gCpuDirect.k1 = (double *)malloc(coeffBytes);
  gCpuDirect.k2 = (double *)malloc(coeffBytes);
  gCpuDirect.k3 = (double *)malloc(coeffBytes);
  if (gCpuDirect.k0 == NULL || gCpuDirect.k1 == NULL ||
      gCpuDirect.k2 == NULL || gCpuDirect.k3 == NULL) {
    freeCpuDirectCache();
    if (sys->benchmarkMode > 0) {
      printf("CPU direct cache unavailable: need %.3f GB host coefficients\n",
             (double)totalCoeffBytes / (1024.0 * 1024.0 * 1024.0));
    }
    return 0;
  }

  kernel = kernelKER4;
  for (i = 0; i < sys->nPnls; i++) {
    panel *pnlX = sys->panelByIdx[i];
    long long base = (long long)i * (long long)sys->nPnls;
    for (j = 0; j < sys->nPnls; j++) {
      panel *pnlY = sys->panelByIdx[j];
      double *KER = panelIA0(pnlX, pnlY);
      long long idx = base + (long long)j;

      gCpuDirect.k0[idx] = KER[0];
      gCpuDirect.k1[idx] = KER[1];
      gCpuDirect.k2[idx] = KER[2];
      gCpuDirect.k3[idx] = KER[3];
    }
  }

  gCpuDirect.sys = sys;
  gCpuDirect.nPnls = sys->nPnls;
  gCpuDirect.nInteractions = nInteractions;
  if (sys->benchmarkMode > 0) {
    printf("CPU direct cache: panel-pairs=%lld host-coeff=%.3f GB\n",
           nInteractions,
           (double)totalCoeffBytes / (1024.0 * 1024.0 * 1024.0));
  }
  return 1;
}

int cpuDirectApply(ssystem *sys, double alpha, double beta,
                   const double *sgm, double *pot) {
  int d, s;
  int nPnls;

  if (sys == NULL || sgm == NULL || pot == NULL) {
    return 0;
  }

  if (gCpuDirect.sys != sys) {
    freeCpuDirectCache();
    if (!buildCpuDirectTables(sys)) {
      freeCpuDirectCache();
      return 0;
    }
  }

  nPnls = gCpuDirect.nPnls;
  for (d = 0; d < nPnls; d++) {
    long long base = (long long)d * (long long)nPnls;
    double sumPot = 0.0;
    double sumDpdn = 0.0;

    for (s = 0; s < nPnls; s++) {
      long long idx = base + (long long)s;
      double x_pot = sgm[s];
      double x_dpdn = sgm[s + nPnls];
      sumPot += gCpuDirect.k0[idx] * x_dpdn + gCpuDirect.k1[idx] * x_pot;
      sumDpdn += gCpuDirect.k2[idx] * x_dpdn + gCpuDirect.k3[idx] * x_pot;
    }

    pot[d] = beta * pot[d] + alpha * sumPot;
    pot[d + nPnls] = beta * pot[d + nPnls] + alpha * sumDpdn;
  }

  return 1;
}
