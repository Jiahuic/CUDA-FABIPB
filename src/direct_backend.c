#include "direct_backend.h"
#include "gk.h"
#include "fabipb_system.h"

#include <limits.h>
#include <stddef.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

typedef struct {
  const ssystem *sys;
  int begin;
  int end;
} CpuDirectBuildTask;

static int directThreadCount(int nTasks) {
  const char *env = getenv("FABIPB_DIRECT_THREADS");
  int threads;

  if (env != NULL) {
    threads = atoi(env);
  } else {
    threads = fabipb_online_cpu_count();
  }
  if (threads < 1) {
    threads = 1;
  }
  if (threads > nTasks) {
    threads = nTasks;
  }
  if (threads > 32) {
    threads = 32;
  }
  return threads;
}

static void *cpuDirectBuildWorker(void *arg) {
  CpuDirectBuildTask *task = (CpuDirectBuildTask *)arg;
  int i, j;

  for (i = task->begin; i < task->end; i++) {
    panel *pnlX = task->sys->panelByIdx[i];
    long long base = (long long)i * (long long)task->sys->nPnls;
    for (j = 0; j < task->sys->nPnls; j++) {
      panel *pnlY = task->sys->panelByIdx[j];
      long long idx = base + (long long)j;
      double *KER;

      KER = panelIA0(pnlX, pnlY);
      gCpuDirect.k0[idx] = KER[0];
      gCpuDirect.k1[idx] = KER[1];
      gCpuDirect.k2[idx] = KER[2];
      gCpuDirect.k3[idx] = KER[3];
    }
  }
  return NULL;
}

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
  int i;
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
  {
    int nThreads = directThreadCount(sys->nPnls);
    if (nThreads <= 1) {
      CpuDirectBuildTask task;
      task.sys = sys;
      task.begin = 0;
      task.end = sys->nPnls;
      cpuDirectBuildWorker(&task);
    } else {
      pthread_t *threads;
      CpuDirectBuildTask *tasks;

      threads = (pthread_t *)calloc((size_t)nThreads, sizeof(pthread_t));
      tasks = (CpuDirectBuildTask *)calloc((size_t)nThreads, sizeof(CpuDirectBuildTask));
      if (threads == NULL || tasks == NULL) {
        free(threads);
        free(tasks);
        freeCpuDirectCache();
        return 0;
      }
      for (i = 0; i < nThreads; i++) {
        tasks[i].sys = sys;
        /* (long long) so the product cannot overflow int at virus scale and
         * hand a worker a negative range; see setupRHSTreeParallel. */
        tasks[i].begin = (int)(((long long)sys->nPnls * i) / nThreads);
        tasks[i].end = (int)(((long long)sys->nPnls * (i + 1)) / nThreads);
        pthread_create(&threads[i], NULL, cpuDirectBuildWorker, &tasks[i]);
      }
      for (i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
      }
      free(tasks);
      free(threads);
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
