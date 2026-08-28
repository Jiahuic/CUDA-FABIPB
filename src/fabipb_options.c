#define _POSIX_C_SOURCE 200809L

#include "fabipb_options.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gk.h"
#include "gkGlobal.h"
#include "gpu_backend.h"

#define DEFAULT_MAX_DIRECT_RHS_PAIRS 5000000000ULL
#define DEFAULT_MAX_GPU_DIRECT_RHS_PAIRS 200000000000ULL
#define DEFAULT_HUGE_CAPSID_ATOMS 5000000ULL
#define DEFAULT_HUGE_CAPSID_PANELS 5000000ULL

#define HUGE_CAPSID_SEP_RATIO 1.2
#define HUGE_CAPSID_TREE_THETA 0.8
#define HUGE_CAPSID_TREE_ORDER 3

extern double epsilon1, epsilon2;
double rhsTreeTheta(void);
double energyTreeTheta(void);
int rhsChargeExpansionOrderReport(void);
void setChargeTreeOrderPolicy(int order);

unsigned long long fabipb_parse_unsigned_long_long_env(const char *name,
                                                       unsigned long long defaultValue) {
  const char *env = getenv(name);
  char *endptr = NULL;
  unsigned long long value;

  if (env == NULL || env[0] == '\0') {
    return defaultValue;
  }

  errno = 0;
  value = strtoull(env, &endptr, 10);
  if (errno != 0 || endptr == env || *endptr != '\0') {
    fprintf(stderr, "Warning: ignoring invalid %s='%s'\n", name, env);
    return defaultValue;
  }

  return value;
}

unsigned long long fabipb_get_max_direct_rhs_pairs(void) {
  return fabipb_parse_unsigned_long_long_env("FABIPB_MAX_DIRECT_RHS_PAIRS",
                                            DEFAULT_MAX_DIRECT_RHS_PAIRS);
}

static unsigned long long get_max_gpu_direct_rhs_pairs(void) {
  return fabipb_parse_unsigned_long_long_env("FABIPB_MAX_GPU_DIRECT_RHS_PAIRS",
                                            DEFAULT_MAX_GPU_DIRECT_RHS_PAIRS);
}

int fabipb_allow_large_direct_rhs(void) {
  const char *env = getenv("FABIPB_ALLOW_LARGE_DIRECT_RHS");
  return env != NULL && strcmp(env, "1") == 0;
}

static int force_tree_rhs(void) {
  const char *env = getenv("FABIPB_FORCE_TREE_RHS");
  return env != NULL && strcmp(env, "1") == 0;
}

static int can_use_gpu_direct_rhs(int gpuMode) {
  return gpuMode > 0 && gpuBackendAvailable();
}

static unsigned long long get_huge_capsid_atom_threshold(void) {
  const char *env = getenv("FABIPB_HUGE_CAPSID_ATOMS");

  if (env != NULL && env[0] != '\0') {
    return fabipb_parse_unsigned_long_long_env("FABIPB_HUGE_CAPSID_ATOMS",
                                              DEFAULT_HUGE_CAPSID_ATOMS);
  }
  return fabipb_parse_unsigned_long_long_env("FABIPB_Q2M_HUGE_CAPSID_ATOMS",
                                            DEFAULT_HUGE_CAPSID_ATOMS);
}

static int is_high_contrast_dielectric(void) {
  double ratio = epsilon2 / epsilon1;
  return epsilon1 > 0.0 && epsilon1 <= 2.0 && ratio >= 40.0;
}

static int is_capsid_scale(const ssystem *sys, int nPnls) {
  unsigned long long atomThreshold = get_huge_capsid_atom_threshold();
  unsigned long long panelThreshold =
      fabipb_parse_unsigned_long_long_env("FABIPB_HUGE_CAPSID_PANELS",
                                          DEFAULT_HUGE_CAPSID_PANELS);
  return (unsigned long long)sys->nChar > atomThreshold ||
         (nPnls > 0 && (unsigned long long)nPnls > panelThreshold);
}

void fabipb_set_benchmark_thread_defaults(void) {
  const char *vars[] = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS"
  };
  size_t i;

  for (i = 0; i < sizeof(vars) / sizeof(vars[0]); i++) {
    if (getenv(vars[i]) == NULL) {
      setenv(vars[i], "1", 0);
    }
  }
}

int fabipb_missing_external_thread_env(void) {
  const char *vars[] = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS"
  };
  size_t i;

  for (i = 0; i < sizeof(vars) / sizeof(vars[0]); i++) {
    if (getenv(vars[i]) == NULL) {
      return 1;
    }
  }
  return 0;
}

void fabipb_ensure_startup_thread_env(int nargs, char *argv[]) {
  const char *vars[] = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS"
  };
  int needsExec = 0;
  size_t i;

  (void)nargs;
  for (i = 0; i < sizeof(vars) / sizeof(vars[0]); i++) {
    if (getenv(vars[i]) == NULL) {
      setenv(vars[i], "1", 0);
      needsExec = 1;
    }
  }

  if (!needsExec || getenv("FABIPB_THREAD_ENV_REEXEC") != NULL) {
    return;
  }

  setenv("FABIPB_THREAD_ENV_REEXEC", "1", 1);
  execvp(argv[0], argv);
  fprintf(stderr,
          "Warning: failed to restart fabipb with pinned thread env (%s). "
          "Continuing in-process; benchmark reproducibility may be affected.\n",
          strerror(errno));
}

unsigned long long fabipb_count_direct_rhs_pairs(int nPnls, int nChar) {
  return (unsigned long long)nPnls * (unsigned long long)nChar;
}

unsigned long long fabipb_get_active_direct_rhs_pair_limit(int gpuMode) {
  return can_use_gpu_direct_rhs(gpuMode) ? get_max_gpu_direct_rhs_pairs()
                                         : fabipb_get_max_direct_rhs_pairs();
}

int fabipb_should_use_tree_rhs(int nPnls, int nChar, int gpuMode) {
  unsigned long long rhsPairs = fabipb_count_direct_rhs_pairs(nPnls, nChar);
  unsigned long long maxDirectRhsPairs =
      fabipb_get_active_direct_rhs_pair_limit(gpuMode);

  if (force_tree_rhs()) {
    return 1;
  }
  return rhsPairs > maxDirectRhsPairs && !fabipb_allow_large_direct_rhs();
}

void fabipb_report_direct_rhs_limit(int nPnls, int nChar, int gpuMode) {
  unsigned long long rhsPairs = fabipb_count_direct_rhs_pairs(nPnls, nChar);
  unsigned long long maxDirectRhsPairs =
      fabipb_get_active_direct_rhs_pair_limit(gpuMode);

  if (rhsPairs > maxDirectRhsPairs && !fabipb_allow_large_direct_rhs() &&
      !force_tree_rhs()) {
    printf("setupRHS: %llu panel-charge interactions (panels=%d charges=%d) exceeds "
           "FABIPB_MAX_DIRECT_RHS_PAIRS=%llu; using tree-accelerated RHS.\n",
           rhsPairs, nPnls, nChar, maxDirectRhsPairs);
  }
}

void fabipb_resolve_auto_solver_policy(ssystem *sys, int q2mExplicit,
                                       int precondExplicit, int sepExplicit,
                                       int nPnls) {
  unsigned long long threshold = get_huge_capsid_atom_threshold();
  int hugeCapsid = (unsigned long long)sys->nChar > threshold;
  int capsidScale = is_capsid_scale(sys, nPnls);
  int highContrast = is_high_contrast_dielectric();

  if (!q2mExplicit) {
    if (sys->gpuMode <= 0) {
      sys->gpuQ2MMode = 0;
    } else if (hugeCapsid) {
      sys->gpuQ2MMode = 0;
    } else {
      sys->gpuQ2MMode = 1;
    }
  }

  if (!sepExplicit && hugeCapsid) {
    sys->maxSepRatio = HUGE_CAPSID_SEP_RATIO;
  }

  if (capsidScale) {
    setChargeTreeOrderPolicy(HUGE_CAPSID_TREE_ORDER);
    if (sys->benchmarkMode > 0) {
      printf("Resolved charge-tree theta=%g/%g order=%d (capsid-scale policy "
             "requested %g/%d; %d panels, %llu atoms)\n",
             rhsTreeTheta(), energyTreeTheta(), rhsChargeExpansionOrderReport(),
             (double)HUGE_CAPSID_TREE_THETA, HUGE_CAPSID_TREE_ORDER,
             nPnls, (unsigned long long)sys->nChar);
    }
  }

  if (!precondExplicit) {
    if (hugeCapsid) {
      sys->precondCacheMode = 3;
    } else if (highContrast) {
      sys->precondCacheMode = 2;
    } else {
      sys->precondCacheMode = 3;
    }
  }

  if (sys->benchmarkMode > 0) {
    if (sys->matvecMode == 0) {
      const char *q2mReason = q2mExplicit ? "explicit -Q"
          : (sys->gpuMode <= 0) ? "auto: GPU disabled"
          : hugeCapsid ? "auto: huge capsid, preserve GPU memory for nearfield"
                       : "auto: small/medium case, use GPU Q2M/L2P";
      printf("Resolved GPU Q2M mode=%d (%s; huge-capsid-threshold=%llu atoms, charges=%d)\n",
             sys->gpuQ2MMode, q2mReason, threshold, sys->nChar);
    }
    {
      const char *precondReason = precondExplicit ? "explicit -P"
          : hugeCapsid ? "auto: huge capsid, use diagonal preconditioner"
          : highContrast ? "auto: high dielectric contrast, use cached block-LU"
                         : "auto: default diagonal preconditioner";
      printf("Resolved preconditioner mode=%d (%s; eps2/eps1=%g)\n",
             sys->precondCacheMode, precondReason, epsilon2 / epsilon1);
    }
    {
      const char *sepReason = sepExplicit ? "explicit -S"
          : hugeCapsid ? "auto: huge capsid, wider separation ratio"
                       : "auto: default";
      printf("Resolved separation ratio=%g (%s)\n", sys->maxSepRatio, sepReason);
    }
  }
}

const char *fabipb_auto_or_explicit_label(int explicitFlag, int value,
                                          char *buf, size_t bufSize) {
  if (explicitFlag) {
    snprintf(buf, bufSize, "%d", value);
  } else {
    snprintf(buf, bufSize, "auto");
  }
  return buf;
}
