#define _POSIX_C_SOURCE 200809L
/*
 * fabipb.c: main driver
 * This program computes the boundary integral PB equation with fmm method
 * usage:
 *   fabipb [options] panelfile [options]
 *
 * Copyright: Jiahui Chen, Weihua Geng, Johannes Tausch
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include "gkGlobal.h"
#include "gk.h"
#include "gmres.h"
#include "direct_backend.h"
#include "gpu_backend.h"

#define DEFAULT_MAX_DIRECT_RHS_PAIRS 5000000000ULL
#define DEFAULT_MAX_GPU_DIRECT_RHS_PAIRS 200000000000ULL

#if defined(__GNUC__) || defined(__clang__)
#define FABIPB_THREAD_LOCAL __thread
#else
#define FABIPB_THREAD_LOCAL
#endif

/* global variables */
int orderMom=0;
double kappa, epsilon, epsilon1=1.0, epsilon2=80.0;

ssystem *sys;

/* function pointers to kernel routines */
void (*kernel)(double *x, double *y);
void (*kernelD)(double r, int p, double *G0, double *Gk);
void (*kernelDC)(double r, int p, double *G);
void (*kernelDS)(double r, int p, double *G);
int (*MtV)(), (*PtV)();

/* routines used by the main routine */
panel *loadPanel(char *panelfile, const char *meshParam, int *numSing, ssystem *sys,
                 const char *meshControlName, double meshControlValue,
                 const char *backendParamName, double backendParamValue);
void gkInit(ssystem *sys, panel *pnlList, int order, int orderMom);
void setupFMM(ssystem *sys);
void applyFMM( ssystem *sys, double *alpha, double *sgm, double *beta, double *pot );
void setupPreconditioning(ssystem *sys);
int cpuDirectApply(ssystem *sys, double alpha, double beta,
                   const double *sgm, double *pot);

double *panelRHS(int qOrder, panel *pnlX, double *chrY );
double *panelRHSTree(ssystem *sys, int qOrder, panel *pnlX, cube *chgRoot);
void initRhsTreeWorkspace(ssystem *sys, RhsTreeWorkspace *ws);
void freeRhsTreeWorkspace(RhsTreeWorkspace *ws);
void panelRHSTreeWorkspace(ssystem *sys, int qOrder, panel *pnlX, cube *chgRoot,
                           RhsTreeWorkspace *ws, double out[2]);
double *panelIA0(panel *pnlX, panel *pnlY);
int nrCommonVtx(panel *p, panel *q, int *idxX, int *idxY);
void kernelKER4(double *x, double *y);
void buildChargeTree(ssystem *sys);
void computeChgMoments(ssystem *sys);
double rhsTreeTheta(void);
extern double **tLegA, **wLegA;
extern FABIPB_THREAD_LOCAL double *nrmX, *nrmY;
extern int maxQuadOrder;

int MtVmain(double *alpha, double *sgm, double *beta, double *pot);
int PtVfmm(double *pot, double *sgm);
int PtVfmmCached(double *pot, double *sgm);
int PtVfmmCachedLU(double *pot, double *sgm);
int PtVfmmDiagonal(double *pot, double *sgm);
int PtVmain(double *pot, double *sgm);

void applyTreecode( ssystem *sys, double *sgm, double *pot );
void applyTreecodeComponents(ssystem *sys, double *sgm,
                             double *pot0, double *pot1);

static void set_benchmark_thread_defaults(void) {
  const char *vars[] = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "FABIPB_SETUP_THREADS",
    "FABIPB_PRECOND_APPLY_THREADS",
    "FABIPB_NEARFIELD_BUILD_THREADS",
    "FABIPB_DIRECT_THREADS"
  };
  size_t i;

  for (i = 0; i < sizeof(vars) / sizeof(vars[0]); i++) {
    if (getenv(vars[i]) == NULL) {
      setenv(vars[i], "1", 0);
    }
  }
}

static int missing_external_thread_env(void) {
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

static unsigned long long parseUnsignedLongLongEnv(const char *name,
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

static unsigned long long getMaxDirectRhsPairs(void) {
  return parseUnsignedLongLongEnv("FABIPB_MAX_DIRECT_RHS_PAIRS",
                                  DEFAULT_MAX_DIRECT_RHS_PAIRS);
}

static unsigned long long getMaxGpuDirectRhsPairs(void) {
  return parseUnsignedLongLongEnv("FABIPB_MAX_GPU_DIRECT_RHS_PAIRS",
                                  DEFAULT_MAX_GPU_DIRECT_RHS_PAIRS);
}

static int allowLargeDirectRhs(void) {
  const char *env = getenv("FABIPB_ALLOW_LARGE_DIRECT_RHS");
  return env != NULL && strcmp(env, "1") == 0;
}

static int forceTreeRhs(void) {
  const char *env = getenv("FABIPB_FORCE_TREE_RHS");
  return env != NULL && strcmp(env, "1") == 0;
}

static unsigned long long countDirectRhsPairs(int nPnls, int nChar) {
  return (unsigned long long)nPnls * (unsigned long long)nChar;
}

static int canUseGpuDirectRhs(int gpuMode) {
  return gpuMode > 0 && gpuBackendAvailable();
}

static unsigned long long getActiveDirectRhsPairLimit(int gpuMode) {
  return canUseGpuDirectRhs(gpuMode) ? getMaxGpuDirectRhsPairs()
                                     : getMaxDirectRhsPairs();
}

/*
 * Decides whether setupRHS should use the tree-accelerated charge-to-panel
 * path instead of the direct panel-charge double loop. FABIPB_FORCE_TREE_RHS
 * forces the tree path regardless of size (used to validate the tree path
 * against the direct path on small meshes). FABIPB_ALLOW_LARGE_DIRECT_RHS
 * preserves its original meaning: force the direct path even above the active
 * pair limit, for an intentional long benchmark. GPU runs use a separate,
 * larger cap because the CUDA direct RHS is practical for medium-large cases
 * where the CPU direct loop is not.
 */
static int shouldUseTreeRhs(int nPnls, int nChar, int gpuMode) {
  unsigned long long rhsPairs = countDirectRhsPairs(nPnls, nChar);
  unsigned long long maxDirectRhsPairs = getActiveDirectRhsPairLimit(gpuMode);

  if (forceTreeRhs()) {
    return 1;
  }
  return rhsPairs > maxDirectRhsPairs && !allowLargeDirectRhs();
}

static void reportDirectRhsLimit(int nPnls, int nChar, int gpuMode) {
  unsigned long long rhsPairs = countDirectRhsPairs(nPnls, nChar);
  unsigned long long maxDirectRhsPairs = getActiveDirectRhsPairLimit(gpuMode);

  if (rhsPairs > maxDirectRhsPairs && !allowLargeDirectRhs() && !forceTreeRhs()) {
    printf("setupRHS: %llu panel-charge interactions (panels=%d charges=%d) exceeds "
           "FABIPB_MAX_DIRECT_RHS_PAIRS=%llu; using tree-accelerated RHS.\n",
           rhsPairs, nPnls, nChar, maxDirectRhsPairs);
  }
}

static void ensure_startup_thread_env(int nargs, char *argv[]) {
  const char *vars[] = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "FABIPB_SETUP_THREADS",
    "FABIPB_PRECOND_APPLY_THREADS",
    "FABIPB_NEARFIELD_BUILD_THREADS",
    "FABIPB_DIRECT_THREADS"
  };
  int needsExec = 0;
  size_t i;

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

static void buildPanelIndexDirect(ssystem *sys) {
  int idx = 0;
  panel *pnl;

  if (sys->panelByIdx != NULL) {
    return;
  }
  CALLOC(sys->panelByIdx, sys->nPnls, panel *);
  for (pnl = sys->pnlLst; pnl != NULL; pnl = pnl->nextC, idx++) {
    ASSERT(idx < sys->nPnls);
    sys->panelByIdx[idx] = pnl;
  }
}

static void compareApplyFMMOnce(ssystem *sys, double *sgm) {
  double *potCpu, *potGpu;
  double alpha = 1.0, beta = 0.0;
  double maxAbs = 0.0, l2Diff = 0.0, l2Ref = 0.0;
  double savedQ2M, savedM2M, savedM2L, savedL2L, savedL2P, savedNear;
  int oldGpuMode = sys->gpuMode;
  int maxIdx = -1;
  int i, n = 2 * sys->nPnls;

  if (oldGpuMode <= 0) {
    printf("applyFMM debug compare skipped: run with -g=1 to compare CPU and GPU paths.\n");
    return;
  }
  if (!gpuBackendAvailable()) {
    printf("applyFMM debug compare skipped: GPU backend unavailable.\n");
    return;
  }

  CALLOC(potCpu, n, double);
  CALLOC(potGpu, n, double);

  savedQ2M = fmmQ2MTime;
  savedM2M = fmmM2MTime;
  savedM2L = fmmM2LTime;
  savedL2L = fmmL2LTime;
  savedL2P = fmmL2PTime;
  savedNear = fmmNearTime;

  sys->gpuMode = 0;
  applyFMM(sys, &alpha, sgm, &beta, potCpu);
  sys->gpuMode = oldGpuMode;
  applyFMM(sys, &alpha, sgm, &beta, potGpu);

  fmmQ2MTime = savedQ2M;
  fmmM2MTime = savedM2M;
  fmmM2LTime = savedM2L;
  fmmL2LTime = savedL2L;
  fmmL2PTime = savedL2P;
  fmmNearTime = savedNear;

  for (i = 0; i < n; i++) {
    double diff = fabs(potCpu[i] - potGpu[i]);
    if (diff > maxAbs) {
      maxAbs = diff;
      maxIdx = i;
    }
    l2Diff += diff * diff;
    l2Ref += potCpu[i] * potCpu[i];
  }

  printf("applyFMM debug compare: max_abs=%e rel_l2=%e max_idx=%d cpu=%e gpu=%e\n",
         maxAbs,
         (l2Ref > 0.0) ? sqrt(l2Diff / l2Ref) : 0.0,
         maxIdx,
         (maxIdx >= 0) ? potCpu[maxIdx] : 0.0,
         (maxIdx >= 0) ? potGpu[maxIdx] : 0.0);

  free(potCpu);
  free(potGpu);
}

static void comparePrecondOnce(ssystem *sys, double *sgm) {
  double *potOrig, *potCached;
  double maxAbs = 0.0, l2Diff = 0.0, l2Ref = 0.0;
  const char *modeLabel = (sys->precondCacheMode > 1) ? "cached-lu" : "cached-blocks";
  int maxIdx = -1;
  int i, n = 2 * sys->nPnls;

  CALLOC(potOrig, n, double);
  CALLOC(potCached, n, double);

  PtVfmm(potOrig, sgm);
  if (sys->precondCacheMode > 1) {
    PtVfmmCachedLU(potCached, sgm);
  } else {
    PtVfmmCached(potCached, sgm);
  }

  for (i = 0; i < n; i++) {
    double diff = fabs(potOrig[i] - potCached[i]);
    if (diff > maxAbs) {
      maxAbs = diff;
      maxIdx = i;
    }
    l2Diff += diff * diff;
    l2Ref += potOrig[i] * potOrig[i];
  }

  printf("PtVfmm debug compare (%s): max_abs=%e rel_l2=%e max_idx=%d orig=%e test=%e\n",
         modeLabel,
         maxAbs,
         (l2Ref > 0.0) ? sqrt(l2Diff / l2Ref) : 0.0,
         maxIdx,
         (maxIdx >= 0) ? potOrig[maxIdx] : 0.0,
         (maxIdx >= 0) ? potCached[maxIdx] : 0.0);

  free(potOrig);
  free(potCached);
}

static double wall_seconds(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

static int parse_double_arg(const char *arg, double *value) {
  char *endptr;

  errno = 0;
  *value = strtod(arg, &endptr);
  if (errno != 0 || endptr == arg || *endptr != '\0') {
    return 0;
  }
  return 1;
}

static int resolve_mesh_parameter(int meshFlag,
                                  int meshOverrideSet,
                                  double meshOverrideValue,
                                  double meshResolution,
                                  double *backendParamValue,
                                  const char **meshControlName,
                                  double *meshControlValue,
                                  const char **backendParamName) {
  if (meshOverrideSet) {
    *backendParamValue = meshOverrideValue;
    *meshControlName = "backend_override";
    *meshControlValue = meshOverrideValue;
  } else {
    /*
     * Backend-neutral mesh control:
     *   mesh_resolution ~ target spacing in Angstrom.
     * Resolve this to the backend-specific knobs used by the meshers:
     *   MSMS density      ~ 1 / R^2
     *   NanoShaper scale  ~ 1 / R
     * The two backends will still produce different meshes for the same R,
     * but this gives one monotone control surface for branch experiments.
     */
    *meshControlName = "mesh_resolution";
    *meshControlValue = meshResolution;
    if (meshFlag == 1) {
      *backendParamValue = 1.0 / (meshResolution * meshResolution);
    } else if (meshFlag == 2) {
      *backendParamValue = 1.0 / meshResolution;
    } else {
      return 0;
    }
  }

  if (meshFlag == 1) {
    *backendParamName = "msms_density";
  } else if (meshFlag == 2) {
    *backendParamName = "nanoshaper_grid_scale";
  } else {
    return 0;
  }

  return 1;
}

static void print_usage(const char *prog) {
  printf("Usage: %s [options] <panel-base-or-pqr-path>\n", prog);
  printf("Core options:\n");
  printf("  -g=0|1    CPU only or request GPU (default: auto)\n");
  printf("  -m=1|2    regenerate with MSMS or NanoShaper (default: 1)\n");
  printf("  -B=0|1    quiet default or benchmark/profiling output (default: 0)\n");
  printf("  -R=<A>    backend-neutral mesh resolution in angstroms (default: 1)\n");
  printf("            MSMS uses 1/R^2, NanoShaper uses 1/R\n");
  printf("  -d=<val>  backend-specific override: MSMS density or NanoShaper Grid_scale\n");
  printf("  -M=0|1    full solve or mesh-only calibration run (default: 0)\n");
  printf("  -r=0|1|2  FMM, direct GPU, or direct CPU matvec (default: 0)\n");
  printf("  -Q=0|1    CPU-default or GPU-debug Q2M path (default: 0)\n");
  printf("  -G=0|1    interaction or destination-leaf GPU nearfield (default: 1)\n");
  printf("  -P=-1|0|1|2|3  disabled, original, cached-block, cached-LU, or diagonal/Jacobi preconditioner (default: 2)\n");
  printf("  -t=<lev>  tree depth\n");
  printf("  -H=<lev>  coarsest active FMM level (default: 2)\n");
  printf("  -q=<ord>  panel quadrature order\n");
  printf("  -k=<val>  Debye-Huckel kappa\n");
  printf("  -e1=<val> solvent epsilon 1\n");
  printf("  -e2=<val> solvent epsilon 2\n");
  printf("  -S=<val>  FMM separation ratio\n");
  printf("  -o=<val>  GMRES tolerance\n");
  printf("  -a=<val>  GMRES Arnoldi/restart dimension (default: 30)\n");
  printf("  -i=<val>  GMRES maximum iterations (default: 100)\n");
  printf("  -p=<val>  FMM order\n");
  printf("  -pm=<val> moment order override\n");
  printf("Development/debug options:\n");
  printf("  -c=1      compare one CPU/GPU applyFMM call before GMRES\n");
  printf("  -C=1      compare one original/cached preconditioner apply before GMRES\n");
  printf("Environment guards:\n");
  printf("  FABIPB_MAX_DIRECT_RHS_PAIRS=<n>      cap CPU direct setupRHS work\n");
  printf("  FABIPB_MAX_GPU_DIRECT_RHS_PAIRS=<n>  cap GPU direct setupRHS work\n");
  printf("  FABIPB_ALLOW_LARGE_DIRECT_RHS=1  bypass the active direct setupRHS cap\n");
  printf("  FABIPB_FORCE_TREE_RHS=1   force the tree-accelerated setupRHS path\n");
  printf("  FABIPB_RHS_TREE_THETA=<x> charge-tree acceptance ratio (default 0.2)\n");
  printf("  FABIPB_RHS_THREADS=<n> setupRHS tree worker threads (default: online CPUs, max 128)\n");
  printf("  FABIPB_RHS_SUMMARY_PATH=<path> write raw and TABI-style RHS summary CSV\n");
  printf("  FABIPB_RHS_SAMPLE_PATH=<path> write sampled RHS rows after setupRHS\n");
  printf("  FABIPB_RHS_SAMPLE_STRIDE=<n> sample every nth RHS row (default: 1000)\n");
  printf("  FABIPB_DEBUG_COMPARE_RHS=1  also run the other RHS path and report the diff\n");
  printf("  FABIPB_DEBUG_RHS_NORMS=1  report RHS/source-term norms and equation checks\n");
  printf("  FABIPB_DEBUG_OPERATOR_NORMS=1  compare direct operator with TABI-style collocation\n");
  printf("  FABIPB_DEBUG_PANEL_CASES=1  report nearfield panelIA0 topological case counts\n");
  printf("  FABIPB_STOP_AFTER_PANEL_CASES=1  stop after panel case diagnostic\n");
  printf("  FABIPB_DEBUG_PANEL_ALL_TYPES=1  compare panelIA0 topological cases with product quadrature\n");
  printf("  FABIPB_STOP_AFTER_PANEL_ALL_TYPES=1  stop after all-types panel diagnostic\n");
  printf("  FABIPB_DEBUG_PANEL_ONE_COMMON=1  legacy alias for FABIPB_DEBUG_PANEL_ALL_TYPES\n");
  printf("  FABIPB_STOP_AFTER_PANEL_ONE_COMMON=1  legacy stop alias for all-types panel diagnostic\n");
  printf("  FABIPB_DEBUG_DUMP_DENSE_SYSTEM=1  dump dense panelIA0-built system matrix + leaf-cube ids\n");
  printf("  FABIPB_DEBUG_DUMP_DENSE_SYSTEM_MAX=<n>  cap panels for dense dump (default 6000)\n");
  printf("  FABIPB_DENSE_DUMP_PATH=<prefix>  output path prefix for dense dump (default dense_system)\n");
  printf("  FABIPB_STOP_AFTER_DENSE_DUMP=1  stop after dense system dump\n");
  printf("  FABIPB_DEBUG_TABI_NODEPATCH=1  run TABI-style nodepatch collocation diagnostic\n");
  printf("  FABIPB_TABI_NODEPATCH_MAX_VERTS=<n>  cap dense nodepatch diagnostic solve\n");
  printf("  FABIPB_TABI_NODEPATCH_SAMPLE_VERTS=<n>  sample vertices before dense nodepatch checks\n");
  printf("  FABIPB_TABI_NODEPATCH_TREE=1  use approximate tree matvec for full nodepatch diagnostic\n");
  printf("  FABIPB_TABI_NODEPATCH_TREE_THETA=<x>  nodepatch tree acceptance ratio (default 0.35)\n");
  printf("  FABIPB_GMRES_INITIAL=rhs|zero  choose GMRES initial guess (default: rhs)\n");
  printf("  FABIPB_GMRES_DUMP_PREFIX=<path>  dump GMRES setup/first-iteration vectors\n");
  printf("  FABIPB_GMRES_DUMP_STRIDE=<n>  dump every nth GMRES vector row (default: 1)\n");
  printf("  FABIPB_GMRES_STOP_AFTER_ITER=<n>  debug stop after n GMRES iterations\n");
  printf("  FABIPB_DEBUG_SOLUTION_NORMS=1  report post-GMRES solution norms\n");
  printf("  FABIPB_DEBUG_ENERGY_COMPONENTS=1  report both energy-functional terms\n");
  printf("  FABIPB_WRITE_SOLUTION=<path>  write post-GMRES panel solution CSV\n");
  printf("  FABIPB_OPERATOR_VECTOR=ones|ones0|ones1  choose operator diagnostic vector\n");
  printf("  FABIPB_SKIP_SELF_PANEL=1|all|k0|k1|k2|k3  diagnostic: skip same-panel operator coefficients\n");
  printf("  FABIPB_SKIP_PANEL_CASES=one,two,self  diagnostic: skip selected singular panel topologies\n");
  printf("  FABIPB_REUSE_MESH=1       preserve and reuse existing .face/.vert files\n");
  printf("  FABIPB_STOP_AFTER_OPERATOR=1  stop after operator diagnostic\n");
  printf("  FABIPB_STOP_AFTER_TABI_NODEPATCH=1  stop after nodepatch diagnostic\n");
  printf("  FABIPB_STOP_AFTER_RHS=1   stop after setupRHS, skipping preconditioner/GMRES\n");
  printf("  FABIPB_STOP_AFTER_GMRES=1 stop after GMRES, skipping post-solve treecode energy\n");
  printf("  -h        show this help\n");
}
/*
 *  setup right hand side (exterior Neumann problem)
 */
static void ensureChargeTreeBuilt(ssystem *sys) {
  double t0, t1;

  if (sys->chgCubeList != NULL) {
    return;
  }
  t0 = wall_seconds();
  buildChargeTree(sys);
  computeChgMoments(sys);
  t1 = wall_seconds();
  if (sys->benchmarkMode > 0) {
    printf("Charge tree build: %f s (charges=%d depth=%d)\n",
           t1 - t0, sys->nChar, sys->chgDepth);
  }
}

static int debugCompareRhs(void) {
  const char *env = getenv("FABIPB_DEBUG_COMPARE_RHS");
  return env != NULL && strcmp(env, "1") == 0;
}

static int debugRhsNorms(void) {
  const char *env = getenv("FABIPB_DEBUG_RHS_NORMS");
  return env != NULL && strcmp(env, "1") == 0;
}

static int debugOperatorNorms(void) {
  const char *env = getenv("FABIPB_DEBUG_OPERATOR_NORMS");
  return env != NULL && strcmp(env, "1") == 0;
}

static int debugTabiNodepatch(void) {
  const char *env = getenv("FABIPB_DEBUG_TABI_NODEPATCH");
  return env != NULL && strcmp(env, "1") == 0;
}

static int debugPanelCases(void) {
  const char *env = getenv("FABIPB_DEBUG_PANEL_CASES");
  return env != NULL && strcmp(env, "1") == 0;
}

static int debugPanelAllTypes(void) {
  const char *env = getenv("FABIPB_DEBUG_PANEL_ALL_TYPES");
  const char *legacy = getenv("FABIPB_DEBUG_PANEL_ONE_COMMON");
  return (env != NULL && strcmp(env, "1") == 0) ||
         (legacy != NULL && strcmp(legacy, "1") == 0);
}

static int debugSolutionNorms(void) {
  const char *env = getenv("FABIPB_DEBUG_SOLUTION_NORMS");
  return env != NULL && strcmp(env, "1") == 0;
}

static const char *gmresInitialMode(void) {
  const char *env = getenv("FABIPB_GMRES_INITIAL");
  if (env == NULL || env[0] == '\0') {
    return "rhs";
  }
  if (strcmp(env, "zero") == 0 || strcmp(env, "rhs") == 0) {
    return env;
  }
  fprintf(stderr, "Warning: ignoring invalid FABIPB_GMRES_INITIAL='%s'; using rhs\n", env);
  return "rhs";
}

static void writeSolutionIfRequested(ssystem *sys, double *sgm) {
  const char *path = getenv("FABIPB_WRITE_SOLUTION");
  FILE *fp;
  int i;
  panel *pnl;

  if (path == NULL || path[0] == '\0') {
    return;
  }
  fp = fopen(path, "w");
  if (fp == NULL) {
    fprintf(stderr, "Warning: cannot write solution dump '%s'\n", path);
    return;
  }
  fprintf(fp, "idx,mesh_face_idx,x,y,z,area,component0,component1\n");
  for (i = 0, pnl = sys->pnlLst; pnl != NULL; pnl = pnl->nextC, i++) {
    int idx = pnl->idx;
    fprintf(fp, "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            idx, pnl->nSurf - 1, pnl->x[0], pnl->x[1], pnl->x[2], pnl->area,
            sgm[idx], sgm[sys->nPnls + idx]);
  }
  fclose(fp);
  if (sys->benchmarkMode > 0) {
    printf("Wrote solution dump: %s\n", path);
  }
}

typedef struct {
  int count;
  double sum;
  double l1;
  double l2;
  double inf;
  double areaWeightedSum;
  double areaWeightedL1;
  double areaWeightedL2;
} RhsSummaryStats;

static unsigned long long rhsSampleStride(void) {
  unsigned long long stride =
      parseUnsignedLongLongEnv("FABIPB_RHS_SAMPLE_STRIDE", 1000ULL);
  return (stride > 0ULL) ? stride : 1000ULL;
}

static void rhsSummaryAdd(RhsSummaryStats *stats, double value, double area) {
  double av = fabs(value);
  double weighted = value * area;
  double wav = fabs(weighted);

  stats->count++;
  stats->sum += value;
  stats->l1 += av;
  stats->l2 += value * value;
  if (av > stats->inf) {
    stats->inf = av;
  }
  stats->areaWeightedSum += weighted;
  stats->areaWeightedL1 += wav;
  stats->areaWeightedL2 += weighted * weighted;
}

static void writeRhsSummaryRow(FILE *fp, const char *quantity, int component,
                               const RhsSummaryStats *stats,
                               double totalArea) {
  fprintf(fp,
          "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
          quantity, component, stats->count, totalArea, stats->sum, stats->l1,
          sqrt(stats->l2), stats->inf,
          (stats->count > 0) ? stats->sum / (double)stats->count : 0.0,
          stats->areaWeightedSum, stats->areaWeightedL1,
          sqrt(stats->areaWeightedL2),
          (totalArea > 0.0) ? stats->areaWeightedSum / totalArea : 0.0);
}

static void writeRhsSummaryIfRequested(ssystem *sys, double *sgm) {
  const char *summaryPath = getenv("FABIPB_RHS_SUMMARY_PATH");
  const char *samplePath = getenv("FABIPB_RHS_SAMPLE_PATH");
  FILE *summary = NULL;
  FILE *sample = NULL;
  RhsSummaryStats integrated[2];
  RhsSummaryStats tabiAvg[2];
  double totalArea = 0.0;
  unsigned long long sampleStride = rhsSampleStride();
  int i;
  panel *pnl;

  if ((summaryPath == NULL || summaryPath[0] == '\0') &&
      (samplePath == NULL || samplePath[0] == '\0')) {
    return;
  }

  memset(integrated, 0, sizeof(integrated));
  memset(tabiAvg, 0, sizeof(tabiAvg));

  if (summaryPath != NULL && summaryPath[0] != '\0') {
    summary = fopen(summaryPath, "w");
    if (summary == NULL) {
      fprintf(stderr, "Warning: cannot write RHS summary '%s'\n", summaryPath);
    }
  }
  if (samplePath != NULL && samplePath[0] != '\0') {
    sample = fopen(samplePath, "w");
    if (sample == NULL) {
      fprintf(stderr, "Warning: cannot write RHS sample '%s'\n", samplePath);
    } else {
      fprintf(sample,
              "idx,mesh_face_idx,x,y,z,nx,ny,nz,area,integrated0,integrated1,tabi_area_avg0,tabi_area_avg1\n");
    }
  }

  for (i = 0, pnl = sys->pnlLst; pnl != NULL; pnl = pnl->nextC, i++) {
    int idx = pnl->idx;
    double area = (pnl->area > 0.0) ? pnl->area : 1.0;
    double integrated0 = sgm[idx];
    double integrated1 = sgm[sys->nPnls + idx];
    double avg0 = integrated0 / area;
    double avg1 = integrated1 / area;

    totalArea += (pnl->area > 0.0) ? pnl->area : 0.0;
    rhsSummaryAdd(&integrated[0], integrated0, 1.0);
    rhsSummaryAdd(&integrated[1], integrated1, 1.0);
    rhsSummaryAdd(&tabiAvg[0], avg0, area);
    rhsSummaryAdd(&tabiAvg[1], avg1, area);

    if (sample != NULL && (((unsigned long long)idx % sampleStride) == 0ULL)) {
      fprintf(sample,
              "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
              idx, pnl->nSurf - 1, pnl->x[0], pnl->x[1], pnl->x[2],
              pnl->normal[0], pnl->normal[1], pnl->normal[2], pnl->area,
              integrated0, integrated1, avg0, avg1);
    }
  }

  if (summary != NULL) {
    fprintf(summary,
            "quantity,component,count,total_area,sum,l1,l2,inf,mean,area_weighted_sum,area_weighted_l1,area_weighted_l2,area_weighted_mean\n");
    writeRhsSummaryRow(summary, "integrated", 0, &integrated[0], totalArea);
    writeRhsSummaryRow(summary, "integrated", 1, &integrated[1], totalArea);
    writeRhsSummaryRow(summary, "tabi_area_avg", 0, &tabiAvg[0], totalArea);
    writeRhsSummaryRow(summary, "tabi_area_avg", 1, &tabiAvg[1], totalArea);
    fclose(summary);
    if (sys->benchmarkMode > 0) {
      printf("Wrote RHS summary: %s\n", summaryPath);
    }
  }
  if (sample != NULL) {
    fclose(sample);
    if (sys->benchmarkMode > 0) {
      printf("Wrote RHS sample: %s stride=%llu\n", samplePath, sampleStride);
    }
  }
}

static int debugEnergyComponents(void) {
  const char *env = getenv("FABIPB_DEBUG_ENERGY_COMPONENTS");
  return env != NULL && strcmp(env, "1") == 0;
}

static void compareRange(const char *label, double *sgmDirect, double *sgmTree, int lo, int hi) {
  double maxAbs = 0.0, l2Diff = 0.0, l2Ref = 0.0;
  int maxIdx = -1, i;

  for (i = lo; i < hi; i++) {
    double diff = fabs(sgmDirect[i] - sgmTree[i]);
    if (diff > maxAbs) { maxAbs = diff; maxIdx = i; }
    l2Diff += diff*diff;
    l2Ref += sgmDirect[i]*sgmDirect[i];
  }
  printf("setupRHS debug compare (%s): max_abs=%e rel_l2=%e max_idx=%d direct=%e tree=%e\n",
         label, maxAbs, (l2Ref > 0.0) ? sqrt(l2Diff/l2Ref) : 0.0, maxIdx,
         (maxIdx >= 0) ? sgmDirect[maxIdx] : 0.0,
         (maxIdx >= 0) ? sgmTree[maxIdx] : 0.0);
}

static void compareSetupRhsOnce(ssystem *sys, double *sgmDirect, double *sgmTree) {
  int nPnls = sys->nPnls;
  compareRange("y0/potential", sgmDirect, sgmTree, 0, nPnls);
  compareRange("y1/normal-deriv", sgmDirect, sgmTree, nPnls, 2*nPnls);
}

static void printRhsStats(const char *label, double *values, int n) {
  double l1 = 0.0, l2 = 0.0, inf = 0.0, sum = 0.0;
  int i;

  for (i = 0; i < n; i++) {
    double av = fabs(values[i]);
    l1 += av;
    l2 += values[i] * values[i];
    sum += values[i];
    if (av > inf) {
      inf = av;
    }
  }
  printf("RHS norm (%s): n=%d l1=%e l2=%e inf=%e mean=%e\n",
         label, n, l1, sqrt(l2), inf, (n > 0) ? sum / n : 0.0);
}

static void compareRhsToCentroidSource(ssystem *sys, double *sgm) {
  int nPnls = sys->nPnls;
  int nChar = sys->nChar;
  int qOrder = sys->maxQuadOrder;
  double *avg0, *avg1, *pt0, *pt1, *quad0, *quad1;
  double fac = fourPiI / epsilon1;
  double maxAbs0 = 0.0, maxAbs1 = 0.0, l2Diff0 = 0.0, l2Diff1 = 0.0;
  double maxAbsQ0 = 0.0, maxAbsQ1 = 0.0, l2DiffQ0 = 0.0, l2DiffQ1 = 0.0;
  double l2Ref0 = 0.0, l2Ref1 = 0.0;
  int maxIdx0 = -1, maxIdx1 = -1, maxIdxQ0 = -1, maxIdxQ1 = -1;
  int i, j, k, ix, jx;
  panel *pnl;
  double *tLeg = tLegA[qOrder], *wLeg = wLegA[qOrder];

  CALLOC(avg0, nPnls, double);
  CALLOC(avg1, nPnls, double);
  CALLOC(pt0, nPnls, double);
  CALLOC(pt1, nPnls, double);
  CALLOC(quad0, nPnls, double);
  CALLOC(quad1, nPnls, double);

  for (i = 0, pnl = sys->pnlLst; pnl != NULL; pnl = pnl->nextC, i++) {
    double area = (pnl->area > 0.0) ? pnl->area : 1.0;
    avg0[i] = sgm[i] / area;
    avg1[i] = sgm[nPnls + i] / area;

    for (j = 0; j < nChar; j++) {
      double r[3], r2, ri, r3i, ip;
      for (k = 0; k < 3; k++) {
        r[k] = pnl->x[k] - sys->pos[3*j + k];
      }
      r2 = SQR(r[0]) + SQR(r[1]) + SQR(r[2]);
      if (r2 <= 0.0) {
        continue;
      }
      ri = 1.0 / sqrt(r2);
      r3i = ri / r2;
      ip = pnl->normal[0]*r[0] + pnl->normal[1]*r[1] + pnl->normal[2]*r[2];
      pt0[i] += sys->chr[j] * fac * ri;
      pt1[i] += sys->chr[j] * fac * (-ip * r3i);
    }

    for (ix = 0; ix < qOrder; ix++) {
      for (jx = 0; jx < qOrder; jx++) {
        double qpt[3], weight;
        double v0 = 0.0, v1 = 0.0;
        for (k = 0; k < 3; k++) {
          qpt[k] = pnl->vtx[0][k] + tLeg[ix] * (pnl->a[2][k] + tLeg[jx] * pnl->a[0][k]);
        }
        weight = 2.0 * tLeg[ix] * wLeg[ix] * wLeg[jx];
        for (j = 0; j < nChar; j++) {
          double r[3], r2, ri, r3i, ip;
          for (k = 0; k < 3; k++) {
            r[k] = qpt[k] - sys->pos[3*j + k];
          }
          r2 = SQR(r[0]) + SQR(r[1]) + SQR(r[2]);
          if (r2 <= 0.0) {
            continue;
          }
          ri = 1.0 / sqrt(r2);
          r3i = ri / r2;
          ip = pnl->normal[0]*r[0] + pnl->normal[1]*r[1] + pnl->normal[2]*r[2];
          v0 += sys->chr[j] * fac * ri;
          v1 += sys->chr[j] * fac * (-ip * r3i);
        }
        quad0[i] += weight * v0;
        quad1[i] += weight * v1;
      }
    }

    {
      double diff0 = fabs(avg0[i] - pt0[i]);
      double diff1 = fabs(avg1[i] - pt1[i]);
      double diffQ0 = fabs(avg0[i] - quad0[i]);
      double diffQ1 = fabs(avg1[i] - quad1[i]);
      if (diff0 > maxAbs0) { maxAbs0 = diff0; maxIdx0 = i; }
      if (diff1 > maxAbs1) { maxAbs1 = diff1; maxIdx1 = i; }
      if (diffQ0 > maxAbsQ0) { maxAbsQ0 = diffQ0; maxIdxQ0 = i; }
      if (diffQ1 > maxAbsQ1) { maxAbsQ1 = diffQ1; maxIdxQ1 = i; }
      l2Diff0 += diff0 * diff0;
      l2Diff1 += diff1 * diff1;
      l2DiffQ0 += diffQ0 * diffQ0;
      l2DiffQ1 += diffQ1 * diffQ1;
      l2Ref0 += avg0[i] * avg0[i];
      l2Ref1 += avg1[i] * avg1[i];
    }
  }

  printRhsStats("integrated potential", sgm, nPnls);
  printRhsStats("integrated normal-deriv", sgm + nPnls, nPnls);
  printRhsStats("area-avg potential", avg0, nPnls);
  printRhsStats("area-avg normal-deriv", avg1, nPnls);
  printRhsStats("centroid TABI-style potential", pt0, nPnls);
  printRhsStats("centroid TABI-style normal-deriv", pt1, nPnls);
  printRhsStats("quadrature TABI-style potential", quad0, nPnls);
  printRhsStats("quadrature TABI-style normal-deriv", quad1, nPnls);

  printf("RHS quadrature equation compare (potential): max_abs=%e rel_l2=%e max_idx=%d area_avg=%e quadrature=%e\n",
         maxAbsQ0, (l2Ref0 > 0.0) ? sqrt(l2DiffQ0 / l2Ref0) : 0.0, maxIdxQ0,
         (maxIdxQ0 >= 0) ? avg0[maxIdxQ0] : 0.0,
         (maxIdxQ0 >= 0) ? quad0[maxIdxQ0] : 0.0);
  printf("RHS quadrature equation compare (normal-deriv): max_abs=%e rel_l2=%e max_idx=%d area_avg=%e quadrature=%e\n",
         maxAbsQ1, (l2Ref1 > 0.0) ? sqrt(l2DiffQ1 / l2Ref1) : 0.0, maxIdxQ1,
         (maxIdxQ1 >= 0) ? avg1[maxIdxQ1] : 0.0,
         (maxIdxQ1 >= 0) ? quad1[maxIdxQ1] : 0.0);

  printf("RHS centroid equation compare (potential): max_abs=%e rel_l2=%e max_idx=%d area_avg=%e centroid=%e\n",
         maxAbs0, (l2Ref0 > 0.0) ? sqrt(l2Diff0 / l2Ref0) : 0.0, maxIdx0,
         (maxIdx0 >= 0) ? avg0[maxIdx0] : 0.0,
         (maxIdx0 >= 0) ? pt0[maxIdx0] : 0.0);
  printf("RHS centroid equation compare (normal-deriv): max_abs=%e rel_l2=%e max_idx=%d area_avg=%e centroid=%e\n",
         maxAbs1, (l2Ref1 > 0.0) ? sqrt(l2Diff1 / l2Ref1) : 0.0, maxIdx1,
         (maxIdx1 >= 0) ? avg1[maxIdx1] : 0.0,
         (maxIdx1 >= 0) ? pt1[maxIdx1] : 0.0);

  free(avg0);
  free(avg1);
  free(pt0);
  free(pt1);
  free(quad0);
  free(quad1);
}

static double deterministicPotential0(int idx) {
  return sin(0.173 * (double)(idx + 1)) + 0.25 * cos(0.071 * (double)(idx + 3));
}

static double deterministicPotential1(int idx) {
  return cos(0.137 * (double)(idx + 1)) - 0.20 * sin(0.113 * (double)(idx + 5));
}

static const char *operatorVectorMode(void) {
  const char *env = getenv("FABIPB_OPERATOR_VECTOR");
  return env != NULL ? env : "deterministic";
}

static void printOperatorBlockCompare(const char *label, double *root, double *rootOff,
                                      double *tabi, int n) {
  double maxAbs = 0.0, maxAbsOff = 0.0;
  double l2Diff = 0.0, l2DiffOff = 0.0, l2Flip = 0.0, l2FlipOff = 0.0;
  double l2Ref = 0.0, l2RefOff = 0.0;
  int maxIdx = -1, maxIdxOff = -1, i;

  for (i = 0; i < n; i++) {
    double diff = fabs(root[i] - tabi[i]);
    double diffOff = fabs(rootOff[i] - tabi[i]);
    if (diff > maxAbs) { maxAbs = diff; maxIdx = i; }
    if (diffOff > maxAbsOff) { maxAbsOff = diffOff; maxIdxOff = i; }
    l2Diff += diff * diff;
    l2DiffOff += diffOff * diffOff;
    l2Flip += (root[i] + tabi[i]) * (root[i] + tabi[i]);
    l2FlipOff += (rootOff[i] + tabi[i]) * (rootOff[i] + tabi[i]);
    l2Ref += root[i] * root[i];
    l2RefOff += rootOff[i] * rootOff[i];
  }

  printf("Operator block %s full-vs-TABI: max_abs=%e rel_l2=%e max_idx=%d root=%e tabi=%e\n",
         label, maxAbs, (l2Ref > 0.0) ? sqrt(l2Diff / l2Ref) : 0.0, maxIdx,
         (maxIdx >= 0) ? root[maxIdx] : 0.0,
         (maxIdx >= 0) ? tabi[maxIdx] : 0.0);
  printf("Operator block %s offdiag-vs-TABI: max_abs=%e rel_l2=%e max_idx=%d root_off=%e tabi=%e\n",
         label, maxAbsOff, (l2RefOff > 0.0) ? sqrt(l2DiffOff / l2RefOff) : 0.0, maxIdxOff,
         (maxIdxOff >= 0) ? rootOff[maxIdxOff] : 0.0,
         (maxIdxOff >= 0) ? tabi[maxIdxOff] : 0.0);
  printf("Operator block %s sign-flip check: full_rel_l2=%e offdiag_rel_l2=%e\n",
         label,
         (l2Ref > 0.0) ? sqrt(l2Flip / l2Ref) : 0.0,
         (l2RefOff > 0.0) ? sqrt(l2FlipOff / l2RefOff) : 0.0);
}

static void compareOperatorToTabiCollocation(ssystem *sys) {
  int nPnls = sys->nPnls;
  const char *vecMode = operatorVectorMode();
  double coeff0 = 0.5 * (1.0 + epsilon);
  double coeff1 = 0.5 * (1.0 + 1.0 / epsilon);
  double *x, *rootIntegral, *rootA0, *rootA1, *rootOff0, *rootOff1, *tabiA0, *tabiA1;
  double *rootBlock[4], *rootOffBlock[4], *tabiBlock[4];
  double maxAbs0 = 0.0, maxAbs1 = 0.0, l2Diff0 = 0.0, l2Diff1 = 0.0;
  double maxAbsOff0 = 0.0, maxAbsOff1 = 0.0, l2DiffOff0 = 0.0, l2DiffOff1 = 0.0;
  double l2Ref0 = 0.0, l2Ref1 = 0.0;
  double l2RefOff0 = 0.0, l2RefOff1 = 0.0;
  double selfL20 = 0.0, selfL21 = 0.0, diagL20 = 0.0, diagL21 = 0.0;
  double selfInf0 = 0.0, selfInf1 = 0.0, diagInf0 = 0.0, diagInf1 = 0.0;
  double selfCoeffL2[4] = {0.0, 0.0, 0.0, 0.0};
  double selfCoeffInf[4] = {0.0, 0.0, 0.0, 0.0};
  double selfCoeffSum[4] = {0.0, 0.0, 0.0, 0.0};
  int maxIdx0 = -1, maxIdx1 = -1, maxIdxOff0 = -1, maxIdxOff1 = -1;
  int i, j, k;

  CALLOC(x, 2*nPnls, double);
  CALLOC(rootIntegral, 2*nPnls, double);
  CALLOC(rootA0, nPnls, double);
  CALLOC(rootA1, nPnls, double);
  CALLOC(rootOff0, nPnls, double);
  CALLOC(rootOff1, nPnls, double);
  CALLOC(tabiA0, nPnls, double);
  CALLOC(tabiA1, nPnls, double);
  for (k = 0; k < 4; k++) {
    CALLOC(rootBlock[k], nPnls, double);
    CALLOC(rootOffBlock[k], nPnls, double);
    CALLOC(tabiBlock[k], nPnls, double);
  }

  buildPanelIndexDirect(sys);

  for (i = 0; i < nPnls; i++) {
    if (strcmp(vecMode, "ones") == 0) {
      x[i] = 1.0;
      x[nPnls + i] = 1.0;
    } else if (strcmp(vecMode, "ones0") == 0) {
      x[i] = 1.0;
      x[nPnls + i] = 0.0;
    } else if (strcmp(vecMode, "ones1") == 0) {
      x[i] = 0.0;
      x[nPnls + i] = 1.0;
    } else {
      x[i] = deterministicPotential0(i);
      x[nPnls + i] = deterministicPotential1(i);
    }
  }
  printf("Operator diagnostic vector: %s\n", vecMode);

  if (!cpuDirectApply(sys, 1.0, 0.0, x, rootIntegral)) {
    printf("Operator equation compare skipped: CPU direct operator unavailable.\n");
    free(x); free(rootIntegral); free(rootA0); free(rootA1);
    free(rootOff0); free(rootOff1); free(tabiA0); free(tabiA1);
    return;
  }

  for (i = 0; i < nPnls; i++) {
    panel *target = sys->panelByIdx[i];
    double area = (target->area > 0.0) ? target->area : 1.0;
    double *selfKer = panelIA0(target, target);
    double self0 = selfKer[0] * x[nPnls + i] + selfKer[1] * x[i];
    double self1 = selfKer[2] * x[nPnls + i] + selfKer[3] * x[i];
    double selfAvg0 = self0 / area;
    double selfAvg1 = self1 / area;
    double diag0 = coeff0 * x[i];
    double diag1 = coeff1 * x[nPnls + i];

    for (k = 0; k < 4; k++) {
      double coeffAvg = selfKer[k] / area;
      selfCoeffL2[k] += coeffAvg * coeffAvg;
      selfCoeffSum[k] += coeffAvg;
      if (fabs(coeffAvg) > selfCoeffInf[k]) {
        selfCoeffInf[k] = fabs(coeffAvg);
      }
    }

    selfL20 += selfAvg0 * selfAvg0;
    selfL21 += selfAvg1 * selfAvg1;
    diagL20 += diag0 * diag0;
    diagL21 += diag1 * diag1;
    if (fabs(selfAvg0) > selfInf0) { selfInf0 = fabs(selfAvg0); }
    if (fabs(selfAvg1) > selfInf1) { selfInf1 = fabs(selfAvg1); }
    if (fabs(diag0) > diagInf0) { diagInf0 = fabs(diag0); }
    if (fabs(diag1) > diagInf1) { diagInf1 = fabs(diag1); }

    rootA0[i] = diag0 - rootIntegral[i] / area;
    rootA1[i] = diag1 - rootIntegral[nPnls + i] / area;
    rootOff0[i] = diag0 - (rootIntegral[i] - self0) / area;
    rootOff1[i] = diag1 - (rootIntegral[nPnls + i] - self1) / area;
    tabiA0[i] = diag0;
    tabiA1[i] = diag1;

    for (j = 0; j < nPnls; j++) {
      panel *source = sys->panelByIdx[j];
      double rvec[3], r2, r, oneOverR, kappaR, expKappaR;
      double G0, Gk, sourceCos, targetCos, tp1, tp2;
      double dot, G3, G4, L1, L2, L3, L4;

      for (k = 0; k < 3; k++) {
        rvec[k] = source->x[k] - target->x[k];
      }
      r2 = SQR(rvec[0]) + SQR(rvec[1]) + SQR(rvec[2]);
      if (r2 <= 0.0) {
        continue;
      }
      r = sqrt(r2);
      oneOverR = 1.0 / r;
      G0 = fourPiI * oneOverR;
      kappaR = kappa * r;
      expKappaR = exp(-kappaR);
      Gk = expKappaR * G0;
      sourceCos = (source->normal[0] * rvec[0] +
                   source->normal[1] * rvec[1] +
                   source->normal[2] * rvec[2]) * oneOverR;
      targetCos = (target->normal[0] * rvec[0] +
                   target->normal[1] * rvec[1] +
                   target->normal[2] * rvec[2]) * oneOverR;
      tp1 = G0 * oneOverR;
      tp2 = (1.0 + kappaR) * expKappaR;
      dot = source->normal[0] * target->normal[0] +
            source->normal[1] * target->normal[1] +
            source->normal[2] * target->normal[2];
      G3 = (dot - 3.0 * targetCos * sourceCos) * oneOverR * tp1;
      G4 = tp2 * G3 - kappa * kappa * targetCos * sourceCos * Gk;

      L1 = sourceCos * tp1 * (1.0 - tp2 * epsilon);
      L2 = G0 - Gk;
      L3 = G4 - G3;
      L4 = targetCos * tp1 * (1.0 - tp2 / epsilon);

      tabiA0[i] -= (L1 * x[j] + L2 * x[nPnls + j]) * source->area;
      tabiA1[i] -= (L3 * x[j] + L4 * x[nPnls + j]) * source->area;
      tabiBlock[0][i] -= L2 * source->area;
      tabiBlock[1][i] -= L1 * source->area;
      tabiBlock[2][i] -= L4 * source->area;
      tabiBlock[3][i] -= L3 * source->area;
    }

    for (j = 0; j < nPnls; j++) {
      panel *source = sys->panelByIdx[j];
      double *ker = panelIA0(target, source);
      rootBlock[0][i] -= ker[0] / area;
      rootBlock[1][i] -= ker[1] / area;
      rootBlock[2][i] -= ker[2] / area;
      rootBlock[3][i] -= ker[3] / area;
      if (j != i) {
        rootOffBlock[0][i] -= ker[0] / area;
        rootOffBlock[1][i] -= ker[1] / area;
        rootOffBlock[2][i] -= ker[2] / area;
        rootOffBlock[3][i] -= ker[3] / area;
      }
    }

    {
      double diff0 = fabs(rootA0[i] - tabiA0[i]);
      double diff1 = fabs(rootA1[i] - tabiA1[i]);
      double diffOff0 = fabs(rootOff0[i] - tabiA0[i]);
      double diffOff1 = fabs(rootOff1[i] - tabiA1[i]);
      if (diff0 > maxAbs0) { maxAbs0 = diff0; maxIdx0 = i; }
      if (diff1 > maxAbs1) { maxAbs1 = diff1; maxIdx1 = i; }
      if (diffOff0 > maxAbsOff0) { maxAbsOff0 = diffOff0; maxIdxOff0 = i; }
      if (diffOff1 > maxAbsOff1) { maxAbsOff1 = diffOff1; maxIdxOff1 = i; }
      l2Diff0 += diff0 * diff0;
      l2Diff1 += diff1 * diff1;
      l2DiffOff0 += diffOff0 * diffOff0;
      l2DiffOff1 += diffOff1 * diffOff1;
      l2Ref0 += rootA0[i] * rootA0[i];
      l2Ref1 += rootA1[i] * rootA1[i];
      l2RefOff0 += rootOff0[i] * rootOff0[i];
      l2RefOff1 += rootOff1[i] * rootOff1[i];
    }
  }

  printRhsStats("operator root area-avg component-0", rootA0, nPnls);
  printRhsStats("operator root area-avg component-1", rootA1, nPnls);
  printRhsStats("operator root offdiag area-avg component-0", rootOff0, nPnls);
  printRhsStats("operator root offdiag area-avg component-1", rootOff1, nPnls);
  printRhsStats("operator TABI collocation component-0", tabiA0, nPnls);
  printRhsStats("operator TABI collocation component-1", tabiA1, nPnls);
  printf("Operator self/diag norms (component-0): self_l2=%e diag_l2=%e self_over_diag=%e self_inf=%e diag_inf=%e\n",
         sqrt(selfL20), sqrt(diagL20), (diagL20 > 0.0) ? sqrt(selfL20 / diagL20) : 0.0,
         selfInf0, diagInf0);
  printf("Operator self/diag norms (component-1): self_l2=%e diag_l2=%e self_over_diag=%e self_inf=%e diag_inf=%e\n",
         sqrt(selfL21), sqrt(diagL21), (diagL21 > 0.0) ? sqrt(selfL21 / diagL21) : 0.0,
         selfInf1, diagInf1);
  for (k = 0; k < 4; k++) {
    printf("Operator self coefficient k%d/area: l2=%e inf=%e mean=%e\n",
           k, sqrt(selfCoeffL2[k]), selfCoeffInf[k],
           (nPnls > 0) ? selfCoeffSum[k] / (double)nPnls : 0.0);
  }
  printOperatorBlockCompare("k0(component1->component0)", rootBlock[0], rootOffBlock[0], tabiBlock[0], nPnls);
  printOperatorBlockCompare("k1(component0->component0)", rootBlock[1], rootOffBlock[1], tabiBlock[1], nPnls);
  printOperatorBlockCompare("k2(component1->component1)", rootBlock[2], rootOffBlock[2], tabiBlock[2], nPnls);
  printOperatorBlockCompare("k3(component0->component1)", rootBlock[3], rootOffBlock[3], tabiBlock[3], nPnls);
  printf("Operator equation compare (component-0): max_abs=%e rel_l2=%e max_idx=%d root=%e tabi=%e\n",
         maxAbs0, (l2Ref0 > 0.0) ? sqrt(l2Diff0 / l2Ref0) : 0.0, maxIdx0,
         (maxIdx0 >= 0) ? rootA0[maxIdx0] : 0.0,
         (maxIdx0 >= 0) ? tabiA0[maxIdx0] : 0.0);
  printf("Operator equation compare (component-1): max_abs=%e rel_l2=%e max_idx=%d root=%e tabi=%e\n",
         maxAbs1, (l2Ref1 > 0.0) ? sqrt(l2Diff1 / l2Ref1) : 0.0, maxIdx1,
         (maxIdx1 >= 0) ? rootA1[maxIdx1] : 0.0,
         (maxIdx1 >= 0) ? tabiA1[maxIdx1] : 0.0);
  printf("Operator offdiag equation compare (component-0): max_abs=%e rel_l2=%e max_idx=%d root=%e tabi=%e\n",
         maxAbsOff0, (l2RefOff0 > 0.0) ? sqrt(l2DiffOff0 / l2RefOff0) : 0.0, maxIdxOff0,
         (maxIdxOff0 >= 0) ? rootOff0[maxIdxOff0] : 0.0,
         (maxIdxOff0 >= 0) ? tabiA0[maxIdxOff0] : 0.0);
  printf("Operator offdiag equation compare (component-1): max_abs=%e rel_l2=%e max_idx=%d root=%e tabi=%e\n",
         maxAbsOff1, (l2RefOff1 > 0.0) ? sqrt(l2DiffOff1 / l2RefOff1) : 0.0, maxIdxOff1,
         (maxIdxOff1 >= 0) ? rootOff1[maxIdxOff1] : 0.0,
         (maxIdxOff1 >= 0) ? tabiA1[maxIdxOff1] : 0.0);

  free(x);
  free(rootIntegral);
  free(rootA0);
  free(rootA1);
  free(rootOff0);
  free(rootOff1);
  free(tabiA0);
  free(tabiA1);
  for (k = 0; k < 4; k++) {
    free(rootBlock[k]);
    free(rootOffBlock[k]);
    free(tabiBlock[k]);
  }
}

static void reportPanelCaseDistribution(ssystem *sys) {
  long long disjoint = 0;
  long long oneCommon = 0;
  long long twoCommon = 0;
  long long twoCommonRev = 0;
  long long self = 0;
  long long other = 0;
  long long total = 0;
  long long leafPairsSelf = 0;
  long long leafPairsOther = 0;
  int pairIdx;

  buildPanelIndexDirect(sys);
  for (pairIdx = 0; pairIdx < sys->nNearPairsFlat; pairIdx++) {
    int srcLeaf = sys->nearPairSrc[pairIdx];
    int dstLeaf = sys->nearPairDst[pairIdx];
    int srcStart = sys->leafPanelStart[srcLeaf];
    int srcCount = sys->leafPanelCount[srcLeaf];
    int dstStart = sys->leafPanelStart[dstLeaf];
    int dstCount = sys->leafPanelCount[dstLeaf];
    int i, j;

    if (srcLeaf == dstLeaf) {
      leafPairsSelf++;
    } else {
      leafPairsOther++;
    }

    for (i = 0; i < dstCount; i++) {
      panel *pnlX = sys->panelByIdx[dstStart + i];
      for (j = 0; j < srcCount; j++) {
        panel *pnlY = sys->panelByIdx[srcStart + j];
        int idxX[3], idxY[3];
        int nVtx = nrCommonVtx(pnlX, pnlY, idxX, idxY);
        if (nVtx == 0) {
          disjoint++;
        } else if (nVtx == 1) {
          oneCommon++;
        } else if (nVtx == 2) {
          twoCommon++;
        } else if (nVtx == -2) {
          twoCommonRev++;
        } else if (nVtx == 3) {
          self++;
        } else {
          other++;
        }
        total++;
      }
    }
  }

  printf("Panel nearfield case distribution: leaf_pairs=%d self_leaf_pairs=%lld other_leaf_pairs=%lld panel_pairs=%lld\n",
         sys->nNearPairsFlat, leafPairsSelf, leafPairsOther, total);
  printf("Panel nearfield cases: disjoint=%lld one-common=%lld two-common=%lld two-common-rev=%lld self=%lld other=%lld\n",
         disjoint, oneCommon, twoCommon, twoCommonRev, self, other);
  if (total > 0) {
    printf("Panel nearfield case fractions: disjoint=%.6f one-common=%.6f two-common=%.6f two-common-rev=%.6f self=%.6f other=%.6f\n",
           (double)disjoint / (double)total,
           (double)oneCommon / (double)total,
           (double)twoCommon / (double)total,
           (double)twoCommonRev / (double)total,
           (double)self / (double)total,
           (double)other / (double)total);
  }
}

static void evalKernelKer4Point(const panel *target, const panel *source,
                                const double *xPt, const double *yPt,
                                double *out) {
  double rvec[3];
  int k;

  nrmX = (double *)target->normal;
  nrmY = (double *)source->normal;
  for (k = 0; k < 3; k++) {
    rvec[k] = xPt[k] - yPt[k];
  }
  kernelKER4(rvec, out);
}

static void panelProductQuadratureReference(panel *target, panel *source,
                                            int qOrder, double *ref) {
  double *tLeg = tLegA[qOrder];
  double *wLeg = wLegA[qOrder];
  int ix, jx, iy, jy, k, c;

  for (c = 0; c < 4; c++) {
    ref[c] = 0.0;
  }

  for (ix = 0; ix < qOrder; ix++) {
    for (jx = 0; jx < qOrder; jx++) {
      double xPt[3];
      double wx = tLeg[ix] * wLeg[ix] * wLeg[jx];
      for (k = 0; k < 3; k++) {
        xPt[k] = target->vtx[0][k] +
          tLeg[ix] * (target->a[2][k] + tLeg[jx] * target->a[0][k]);
      }
      for (iy = 0; iy < qOrder; iy++) {
        for (jy = 0; jy < qOrder; jy++) {
          double yPt[3], val[4];
          double wy = tLeg[iy] * wLeg[iy] * wLeg[jy];
          for (k = 0; k < 3; k++) {
            yPt[k] = source->vtx[0][k] +
              tLeg[iy] * (source->a[2][k] + tLeg[jy] * source->a[0][k]);
          }
          evalKernelKer4Point(target, source, xPt, yPt, val);
          for (c = 0; c < 4; c++) {
            ref[c] += val[c] * wx * wy;
          }
        }
      }
    }
  }

  for (c = 0; c < 4; c++) {
    ref[c] *= 4.0 * target->area * source->area;
  }
}

static int panelCaseSlot(int nVtx) {
  if (nVtx == 0) return 0;
  if (nVtx == 1) return 1;
  if (nVtx == 2) return 2;
  if (nVtx == -2) return 3;
  if (nVtx == 3) return 4;
  return -1;
}

static const char *panelCaseName(int slot) {
  static const char *names[5] = {
    "disjoint",
    "one-common",
    "two-common",
    "two-common-rev",
    "self"
  };
  return (slot >= 0 && slot < 5) ? names[slot] : "other";
}

static void reportPanelAllTypesDiagnostics(ssystem *sys) {
  int pairIdx;
  int reported[5] = {0, 0, 0, 0, 0};
  int maxReports = 8;
  int refOrder = (maxQuadOrder >= 10) ? 10 : maxQuadOrder;
  double maxRel[5][4];
  double maxAbs[5][4];
  int slot, c;

  for (slot = 0; slot < 5; slot++) {
    for (c = 0; c < 4; c++) {
      maxRel[slot][c] = 0.0;
      maxAbs[slot][c] = 0.0;
    }
  }

  kernel = kernelKER4;
  buildPanelIndexDirect(sys);
  printf("Panel all-types diagnostic: comparing panelIA0 qOrder=%d against product quadrature qOrder=%d\n",
         sys->maxQuadOrder, refOrder);
  printf("Panel all-types diagnostic: self panels are reported without product reference because tensor product quadrature samples the singularity.\n");

  for (pairIdx = 0; pairIdx < sys->nNearPairsFlat; pairIdx++) {
    int srcLeaf = sys->nearPairSrc[pairIdx];
    int dstLeaf = sys->nearPairDst[pairIdx];
    int srcStart = sys->leafPanelStart[srcLeaf];
    int srcCount = sys->leafPanelCount[srcLeaf];
    int dstStart = sys->leafPanelStart[dstLeaf];
    int dstCount = sys->leafPanelCount[dstLeaf];
    int i, j;
    int done = 1;

    for (slot = 0; slot < 5; slot++) {
      if (reported[slot] < maxReports) {
        done = 0;
        break;
      }
    }
    if (done) break;

    for (i = 0; i < dstCount; i++) {
      panel *pnlX = sys->panelByIdx[dstStart + i];
      for (j = 0; j < srcCount; j++) {
        panel *pnlY = sys->panelByIdx[srcStart + j];
        int idxX[3], idxY[3];
        int nVtx = nrCommonVtx(pnlX, pnlY, idxX, idxY);
        slot = panelCaseSlot(nVtx);
        if (slot >= 0 && reported[slot] < maxReports) {
          double kerCopy[4];
          double *ker;

          ker = panelIA0(pnlX, pnlY);
          for (c = 0; c < 4; c++) {
            kerCopy[c] = ker[c];
          }
          printf("Panel %s pair %d: dst=%d src=%d areaX=%e areaY=%e\n",
                 panelCaseName(slot), reported[slot], dstStart + i, srcStart + j,
                 pnlX->area, pnlY->area);
          if (slot == 4) {
            int vv;
            for (c = 0; c < 4; c++) {
              printf("  k%d panelIA0=% .12e\n", c, kerCopy[c]);
            }
            for (vv = 0; vv < 3; vv++) {
              printf("  vtx[%d]=% .6f % .6f % .6f\n", vv,
                     pnlX->vtx[vv][0], pnlX->vtx[vv][1], pnlX->vtx[vv][2]);
            }
            reported[slot]++;
            continue;
          }

          {
          double ref[4];
          panelProductQuadratureReference(pnlX, pnlY, refOrder, ref);
          for (c = 0; c < 4; c++) {
            double absDiff = fabs(kerCopy[c] - ref[c]);
            double relDiff = absDiff / ((fabs(ref[c]) > 1e-300) ? fabs(ref[c]) : 1.0);
            if (absDiff > maxAbs[slot][c]) maxAbs[slot][c] = absDiff;
            if (relDiff > maxRel[slot][c]) maxRel[slot][c] = relDiff;
            printf("  k%d panelIA0=% .12e ref=% .12e abs=% .3e rel=% .3e\n",
                   c, kerCopy[c], ref[c], absDiff, relDiff);
          }
          if ((slot == 2 || slot == 3) && reported[slot] == 0) {
            int vv;
            for (vv = 0; vv < 3; vv++) {
              printf("  Xvtx[%d]=% .6f % .6f % .6f\n", vv,
                     pnlX->vtx[vv][0], pnlX->vtx[vv][1], pnlX->vtx[vv][2]);
            }
            for (vv = 0; vv < 3; vv++) {
              printf("  Yvtx[%d]=% .6f % .6f % .6f\n", vv,
                     pnlY->vtx[vv][0], pnlY->vtx[vv][1], pnlY->vtx[vv][2]);
            }
            printf("  Xnormal=% .6f % .6f % .6f\n",
                   pnlX->normal[0], pnlX->normal[1], pnlX->normal[2]);
            printf("  Ynormal=% .6f % .6f % .6f\n",
                   pnlY->normal[0], pnlY->normal[1], pnlY->normal[2]);
          }
          }
          reported[slot]++;
        }
      }
    }
  }

  for (slot = 0; slot < 5; slot++) {
    if (slot == 4) {
      printf("Panel %s diagnostic summary: samples=%d product-reference=skipped\n",
             panelCaseName(slot), reported[slot]);
    } else {
      printf("Panel %s diagnostic summary: samples=%d max_rel k0=%e k1=%e k2=%e k3=%e max_abs k0=%e k1=%e k2=%e k3=%e\n",
             panelCaseName(slot), reported[slot],
             maxRel[slot][0], maxRel[slot][1], maxRel[slot][2], maxRel[slot][3],
             maxAbs[slot][0], maxAbs[slot][1], maxAbs[slot][2], maxAbs[slot][3]);
    }
  }
}

static int debugDumpDenseSystem(void) {
  const char *env = getenv("FABIPB_DEBUG_DUMP_DENSE_SYSTEM");
  return env != NULL && strcmp(env, "1") == 0;
}

static int denseDumpMaxPanels(void) {
  const char *env = getenv("FABIPB_DEBUG_DUMP_DENSE_SYSTEM_MAX");
  int val;

  if (env == NULL) {
    return 6000;
  }
  val = atoi(env);
  return (val > 0) ? val : 6000;
}

/*
 * Dumps the dense panel-Galerkin system matrix A (built via the same
 * panelIA0() calls used everywhere else in the code, with no FMM
 * far-field approximation) plus each panel's leaf-cube assignment at
 * the preconditioner's tree level. This lets an external tool build
 * both A and the local block-diagonal preconditioner M (M is exactly
 * A restricted to same-leaf-cube panel pairs, per precond_fmm.c's own
 * construction) and directly compare their conditioning, instead of
 * inferring it from GMRES's convergence behavior alone.
 */
static void dumpDenseSystemForAnalysis(ssystem *sys) {
  int nPnls = sys->nPnls;
  int Msize = 2 * nPnls;
  int maxPanels = denseDumpMaxPanels();
  int nlevel = sys->depth - 1;
  double scale1, scale2;
  double *A;
  int *cubeId;
  int idx, i, j;
  cube *cb;
  const char *pathPrefix = getenv("FABIPB_DENSE_DUMP_PATH");
  char pathA[512], pathCube[512], pathMeta[512];
  FILE *fp;

  if (pathPrefix == NULL) {
    pathPrefix = "dense_system";
  }

  if (nPnls > maxPanels) {
    printf("Dense system dump skipped: nPnls=%d exceeds FABIPB_DEBUG_DUMP_DENSE_SYSTEM_MAX=%d\n",
           nPnls, maxPanels);
    return;
  }

  buildPanelIndexDirect(sys);
  kernel = kernelKER4;
  scale1 = (1.0 + epsilon) / 2.0;
  scale2 = (1.0 + 1.0 / epsilon) / 2.0;

  CALLOC(A, (size_t)Msize * (size_t)Msize, double);
  CALLOC(cubeId, nPnls, int);
  for (i = 0; i < nPnls; i++) {
    cubeId[i] = -1;
  }

  for (idx = 0, cb = sys->cubeList[nlevel]; cb != NULL; cb = cb->next, idx++) {
    panel *pnlC = cb->pnls;
    int k;
    for (k = 0; k < cb->nPnls; k++, pnlC = pnlC->nextC) {
      cubeId[pnlC->idx] = idx;
    }
  }

  for (i = 0; i < nPnls; i++) {
    panel *pnlX = sys->panelByIdx[i];
    for (j = 0; j < nPnls; j++) {
      panel *pnlY = sys->panelByIdx[j];
      double *KER = panelIA0(pnlX, pnlY);
      A[i * Msize + j]                   = -KER[1];
      A[i * Msize + j + nPnls]           = -KER[0];
      A[(i + nPnls) * Msize + j]         = -KER[3];
      A[(i + nPnls) * Msize + j + nPnls] = -KER[2];
    }
    A[i * Msize + i]                     += scale1 * pnlX->area;
    A[(i + nPnls) * Msize + i + nPnls]   += scale2 * pnlX->area;
  }

  snprintf(pathA, sizeof(pathA), "%s_A.bin", pathPrefix);
  snprintf(pathCube, sizeof(pathCube), "%s_cubeid.bin", pathPrefix);
  snprintf(pathMeta, sizeof(pathMeta), "%s_meta.txt", pathPrefix);

  fp = fopen(pathA, "wb");
  ASSERT(fp != NULL);
  fwrite(A, sizeof(double), (size_t)Msize * (size_t)Msize, fp);
  fclose(fp);

  fp = fopen(pathCube, "wb");
  ASSERT(fp != NULL);
  fwrite(cubeId, sizeof(int), (size_t)nPnls, fp);
  fclose(fp);

  fp = fopen(pathMeta, "w");
  ASSERT(fp != NULL);
  fprintf(fp, "nPnls=%d\nMsize=%d\nnlevel=%d\nscale1=%.17g\nscale2=%.17g\n",
          nPnls, Msize, nlevel, scale1, scale2);
  fclose(fp);

  printf("Dense system dump: nPnls=%d Msize=%d nlevel=%d -> %s / %s / %s\n",
         nPnls, Msize, nlevel, pathA, pathCube, pathMeta);

  free(A);
  free(cubeId);
}

typedef struct {
  double x[3];
  double n[3];
  double area;
} TabiNodepatchNode;

typedef struct {
  double x[3];
  double n[3];
  double area;
} TabiNodepatchEntry;

typedef struct {
  int nNodes;
  TabiNodepatchNode *nodes;
} TabiNodepatchMesh;

typedef struct TabiNodepatchTreeNode {
  double center[3];
  double lo[3];
  double hi[3];
  double radius;
  int *idx;
  int nIdx;
  int nKids;
  struct TabiNodepatchTreeNode *kids[8];
  double src0;
  double src1;
  double src0n[3];
} TabiNodepatchTreeNode;

typedef struct {
  const TabiNodepatchMesh *mesh;
  TabiNodepatchTreeNode *root;
  int leafMax;
  int maxDepth;
  double theta;
} TabiNodepatchTree;

static TabiNodepatchMesh *gTabiNodepatchMesh = NULL;
static TabiNodepatchTree *gTabiNodepatchTree = NULL;

static int tabiNodepatchMaxNodes(void) {
  const char *env = getenv("FABIPB_TABI_NODEPATCH_MAX_VERTS");
  char *endptr = NULL;
  long value;

  if (env == NULL || env[0] == '\0') {
    return 2000;
  }
  value = strtol(env, &endptr, 10);
  if (endptr == env || *endptr != '\0' || value <= 0 || value > 100000000L) {
    fprintf(stderr, "Warning: ignoring invalid FABIPB_TABI_NODEPATCH_MAX_VERTS='%s'\n", env);
    return 2000;
  }
  return (int)value;
}

static int tabiNodepatchMaxIter(void) {
  const char *env = getenv("FABIPB_TABI_NODEPATCH_MAX_ITER");
  char *endptr = NULL;
  long value;

  if (env == NULL || env[0] == '\0') {
    return 1000;
  }
  value = strtol(env, &endptr, 10);
  if (endptr == env || *endptr != '\0' || value <= 0 || value > 100000000L) {
    fprintf(stderr, "Warning: ignoring invalid FABIPB_TABI_NODEPATCH_MAX_ITER='%s'\n", env);
    return 1000;
  }
  return (int)value;
}

static int tabiNodepatchSampleNodes(void) {
  const char *env = getenv("FABIPB_TABI_NODEPATCH_SAMPLE_VERTS");
  char *endptr = NULL;
  long value;

  if (env == NULL || env[0] == '\0') {
    return 0;
  }
  value = strtol(env, &endptr, 10);
  if (endptr == env || *endptr != '\0' || value < 0 || value > 100000000L) {
    fprintf(stderr, "Warning: ignoring invalid FABIPB_TABI_NODEPATCH_SAMPLE_VERTS='%s'\n", env);
    return 0;
  }
  return (int)value;
}

static int tabiNodepatchTreeEnabled(void) {
  const char *env = getenv("FABIPB_TABI_NODEPATCH_TREE");
  return env != NULL && strcmp(env, "1") == 0;
}

static int tabiNodepatchTreeLeafMax(void) {
  const char *env = getenv("FABIPB_TABI_NODEPATCH_TREE_LEAF_MAX");
  char *endptr = NULL;
  long value;

  if (env == NULL || env[0] == '\0') {
    return 64;
  }
  value = strtol(env, &endptr, 10);
  if (endptr == env || *endptr != '\0' || value <= 0 || value > 1000000L) {
    fprintf(stderr, "Warning: ignoring invalid FABIPB_TABI_NODEPATCH_TREE_LEAF_MAX='%s'\n", env);
    return 64;
  }
  return (int)value;
}

static double tabiNodepatchTreeTheta(void) {
  const char *env = getenv("FABIPB_TABI_NODEPATCH_TREE_THETA");
  char *endptr = NULL;
  double value;

  if (env == NULL || env[0] == '\0') {
    return 0.35;
  }
  value = strtod(env, &endptr);
  if (endptr == env || *endptr != '\0' || value <= 0.0 || value > 2.0) {
    fprintf(stderr, "Warning: ignoring invalid FABIPB_TABI_NODEPATCH_TREE_THETA='%s'\n", env);
    return 0.35;
  }
  return value;
}

static double tabiNodepatchChargeTreeTheta(void) {
  const char *env = getenv("FABIPB_TABI_NODEPATCH_CHARGE_TREE_THETA");
  char *endptr = NULL;
  double value;

  if (env == NULL || env[0] == '\0') {
    return 0.2;
  }
  value = strtod(env, &endptr);
  if (endptr == env || *endptr != '\0' || value <= 0.0 || value > 2.0) {
    fprintf(stderr, "Warning: ignoring invalid FABIPB_TABI_NODEPATCH_CHARGE_TREE_THETA='%s'\n", env);
    return 0.2;
  }
  return value;
}

static int compareTabiNodepatchEntry(const void *left, const void *right) {
  const TabiNodepatchEntry *a = (const TabiNodepatchEntry *)left;
  const TabiNodepatchEntry *b = (const TabiNodepatchEntry *)right;
  int k;

  for (k = 0; k < 3; k++) {
    if (a->x[k] < b->x[k]) {
      return -1;
    }
    if (a->x[k] > b->x[k]) {
      return 1;
    }
  }
  return 0;
}

static int sameTabiNodepatchCoord(const TabiNodepatchEntry *a, const TabiNodepatchEntry *b) {
  return a->x[0] == b->x[0] && a->x[1] == b->x[1] && a->x[2] == b->x[2];
}

static void freeTabiNodepatchMesh(TabiNodepatchMesh *mesh) {
  if (mesh == NULL) {
    return;
  }
  free(mesh->nodes);
  free(mesh);
}

static TabiNodepatchMesh *sampleTabiNodepatchMesh(const TabiNodepatchMesh *fullMesh, int sampleCount) {
  TabiNodepatchMesh *sample;
  int i, srcIdx;

  if (sampleCount <= 0 || sampleCount >= fullMesh->nNodes) {
    return NULL;
  }

  CALLOC(sample, 1, TabiNodepatchMesh);
  sample->nNodes = sampleCount;
  CALLOC(sample->nodes, sampleCount, TabiNodepatchNode);
  for (i = 0; i < sampleCount; i++) {
    srcIdx = (sampleCount == 1) ? 0 :
      (int)(((long long)i * (long long)(fullMesh->nNodes - 1)) / (long long)(sampleCount - 1));
    sample->nodes[i] = fullMesh->nodes[srcIdx];
  }
  return sample;
}

static void computeTabiNodepatchIndexBounds(const TabiNodepatchMesh *mesh, const int *idx, int nIdx,
                                            double *lo, double *hi, double *center, double *radius) {
  int i, k;

  for (k = 0; k < 3; k++) {
    lo[k] = hi[k] = mesh->nodes[idx[0]].x[k];
  }
  for (i = 1; i < nIdx; i++) {
    const TabiNodepatchNode *node = &mesh->nodes[idx[i]];
    for (k = 0; k < 3; k++) {
      if (node->x[k] < lo[k]) lo[k] = node->x[k];
      if (node->x[k] > hi[k]) hi[k] = node->x[k];
    }
  }
  for (k = 0; k < 3; k++) {
    center[k] = 0.5 * (lo[k] + hi[k]);
  }
  *radius = 0.5 * sqrt(SQR(hi[0] - lo[0]) + SQR(hi[1] - lo[1]) + SQR(hi[2] - lo[2]));
}

static TabiNodepatchTreeNode *buildTabiNodepatchTreeNode(const TabiNodepatchMesh *mesh,
                                                         int *idx, int nIdx,
                                                         int depth, int maxDepth,
                                                         int leafMax) {
  TabiNodepatchTreeNode *node;
  int i, k, oct, nonemptyKids = 0;
  int *childCounts = NULL;
  int *childOffsets = NULL;
  int *childCursor = NULL;
  int *childIdx = NULL;

  CALLOC(node, 1, TabiNodepatchTreeNode);
  computeTabiNodepatchIndexBounds(mesh, idx, nIdx, node->lo, node->hi,
                                  node->center, &node->radius);

  if (nIdx <= leafMax || depth >= maxDepth || node->radius <= 0.0) {
    CALLOC(node->idx, nIdx, int);
    memcpy(node->idx, idx, (size_t)nIdx * sizeof(int));
    node->nIdx = nIdx;
    return node;
  }

  CALLOC(childCounts, 8, int);
  for (i = 0; i < nIdx; i++) {
    const TabiNodepatchNode *src = &mesh->nodes[idx[i]];
    oct = 0;
    if (src->x[0] >= node->center[0]) oct |= 1;
    if (src->x[1] >= node->center[1]) oct |= 2;
    if (src->x[2] >= node->center[2]) oct |= 4;
    childCounts[oct]++;
  }
  for (k = 0; k < 8; k++) {
    if (childCounts[k] > 0) nonemptyKids++;
  }
  if (nonemptyKids <= 1) {
    CALLOC(node->idx, nIdx, int);
    memcpy(node->idx, idx, (size_t)nIdx * sizeof(int));
    node->nIdx = nIdx;
    free(childCounts);
    return node;
  }

  CALLOC(childOffsets, 9, int);
  CALLOC(childCursor, 8, int);
  for (k = 0; k < 8; k++) {
    childOffsets[k + 1] = childOffsets[k] + childCounts[k];
    childCursor[k] = childOffsets[k];
  }
  CALLOC(childIdx, nIdx, int);
  for (i = 0; i < nIdx; i++) {
    const TabiNodepatchNode *src = &mesh->nodes[idx[i]];
    oct = 0;
    if (src->x[0] >= node->center[0]) oct |= 1;
    if (src->x[1] >= node->center[1]) oct |= 2;
    if (src->x[2] >= node->center[2]) oct |= 4;
    childIdx[childCursor[oct]++] = idx[i];
  }

  for (k = 0; k < 8; k++) {
    if (childCounts[k] > 0) {
      node->kids[node->nKids++] =
        buildTabiNodepatchTreeNode(mesh, &childIdx[childOffsets[k]], childCounts[k],
                                   depth + 1, maxDepth, leafMax);
    }
  }

  free(childCounts);
  free(childOffsets);
  free(childCursor);
  free(childIdx);
  return node;
}

static TabiNodepatchTree *buildTabiNodepatchTree(const TabiNodepatchMesh *mesh) {
  TabiNodepatchTree *tree;
  int *idx;
  int i;

  CALLOC(tree, 1, TabiNodepatchTree);
  tree->mesh = mesh;
  tree->leafMax = tabiNodepatchTreeLeafMax();
  tree->maxDepth = 24;
  tree->theta = tabiNodepatchTreeTheta();
  CALLOC(idx, mesh->nNodes, int);
  for (i = 0; i < mesh->nNodes; i++) {
    idx[i] = i;
  }
  tree->root = buildTabiNodepatchTreeNode(mesh, idx, mesh->nNodes, 0,
                                          tree->maxDepth, tree->leafMax);
  free(idx);
  return tree;
}

static void freeTabiNodepatchTreeNode(TabiNodepatchTreeNode *node) {
  int k;

  if (node == NULL) {
    return;
  }
  for (k = 0; k < node->nKids; k++) {
    freeTabiNodepatchTreeNode(node->kids[k]);
  }
  free(node->idx);
  free(node);
}

static void freeTabiNodepatchTree(TabiNodepatchTree *tree) {
  if (tree == NULL) {
    return;
  }
  freeTabiNodepatchTreeNode(tree->root);
  free(tree);
}

static void clearTabiNodepatchTreeSources(TabiNodepatchTreeNode *node) {
  int k;

  node->src0 = 0.0;
  node->src1 = 0.0;
  node->src0n[0] = node->src0n[1] = node->src0n[2] = 0.0;
  for (k = 0; k < node->nKids; k++) {
    clearTabiNodepatchTreeSources(node->kids[k]);
  }
}

static void accumulateTabiNodepatchTreeSources(const TabiNodepatchMesh *mesh,
                                               TabiNodepatchTreeNode *node,
                                               const double *x) {
  int i, k, n = mesh->nNodes;

  if (node->nKids == 0) {
    for (i = 0; i < node->nIdx; i++) {
      int idx = node->idx[i];
      const TabiNodepatchNode *src = &mesh->nodes[idx];
      double q0 = src->area * x[idx];
      double q1 = src->area * x[n + idx];
      node->src0 += q0;
      node->src1 += q1;
      for (k = 0; k < 3; k++) {
        node->src0n[k] += q0 * src->n[k];
      }
    }
    return;
  }

  for (k = 0; k < node->nKids; k++) {
    accumulateTabiNodepatchTreeSources(mesh, node->kids[k], x);
    node->src0 += node->kids[k]->src0;
    node->src1 += node->kids[k]->src1;
    for (i = 0; i < 3; i++) {
      node->src0n[i] += node->kids[k]->src0n[i];
    }
  }
}

static TabiNodepatchMesh *buildTabiNodepatchMesh(ssystem *sys) {
  int nEntries = 3 * sys->nPnls;
  int i, j, k, count, current;
  panel *pnl;
  TabiNodepatchEntry *entries;
  TabiNodepatchMesh *mesh;

  CALLOC(entries, nEntries, TabiNodepatchEntry);
  for (i = 0, pnl = sys->pnlLst; pnl != NULL; pnl = pnl->nextC, i++) {
    for (j = 0; j < 3; j++) {
      int idx = 3 * i + j;
      for (k = 0; k < 3; k++) {
        entries[idx].x[k] = pnl->vtx[j][k];
        entries[idx].n[k] = pnl->nrm[j][k];
      }
      entries[idx].area = pnl->area / 3.0;
    }
  }

  qsort(entries, nEntries, sizeof(TabiNodepatchEntry), compareTabiNodepatchEntry);
  count = 0;
  for (i = 0; i < nEntries; i++) {
    if (i == 0 || !sameTabiNodepatchCoord(&entries[i], &entries[i - 1])) {
      count++;
    }
  }

  CALLOC(mesh, 1, TabiNodepatchMesh);
  mesh->nNodes = count;
  CALLOC(mesh->nodes, count, TabiNodepatchNode);

  current = -1;
  for (i = 0; i < nEntries; i++) {
    if (i == 0 || !sameTabiNodepatchCoord(&entries[i], &entries[i - 1])) {
      current++;
      for (k = 0; k < 3; k++) {
        mesh->nodes[current].x[k] = entries[i].x[k];
      }
    }
    mesh->nodes[current].area += entries[i].area;
    for (k = 0; k < 3; k++) {
      mesh->nodes[current].n[k] += entries[i].n[k] * entries[i].area;
    }
  }

  for (i = 0; i < mesh->nNodes; i++) {
    double nrm = sqrt(SQR(mesh->nodes[i].n[0]) +
                      SQR(mesh->nodes[i].n[1]) +
                      SQR(mesh->nodes[i].n[2]));
    if (nrm > 0.0) {
      for (k = 0; k < 3; k++) {
        mesh->nodes[i].n[k] /= nrm;
      }
    }
  }

  free(entries);
  return mesh;
}

static void applyTabiNodepatchOperator(const TabiNodepatchMesh *mesh,
                                       const double *x, double *y,
                                       double alpha, double beta) {
  int n = mesh->nNodes;
  int i, j;
  double coeff0 = 0.5 * (1.0 + epsilon);
  double coeff1 = 0.5 * (1.0 + 1.0 / epsilon);

  for (i = 0; i < n; i++) {
    const TabiNodepatchNode *target = &mesh->nodes[i];
    double sum0 = 0.0;
    double sum1 = 0.0;

    for (j = 0; j < n; j++) {
      const TabiNodepatchNode *source;
      double dx, dy, dz, r2, r, oneOverR, G0, Gk, kappaR, expKappaR;
      double sourceCos, targetCos, tp1, tp2, dot, G3, G4, L1, L2, L3, L4;

      if (i == j) {
        continue;
      }
      source = &mesh->nodes[j];
      dx = source->x[0] - target->x[0];
      dy = source->x[1] - target->x[1];
      dz = source->x[2] - target->x[2];
      r2 = SQR(dx) + SQR(dy) + SQR(dz);
      if (r2 <= 0.0) {
        continue;
      }
      r = sqrt(r2);
      oneOverR = 1.0 / r;
      G0 = fourPiI * oneOverR;
      kappaR = kappa * r;
      expKappaR = exp(-kappaR);
      Gk = expKappaR * G0;
      sourceCos = (source->n[0] * dx + source->n[1] * dy + source->n[2] * dz) * oneOverR;
      targetCos = (target->n[0] * dx + target->n[1] * dy + target->n[2] * dz) * oneOverR;
      tp1 = G0 * oneOverR;
      tp2 = (1.0 + kappaR) * expKappaR;
      dot = source->n[0] * target->n[0] +
            source->n[1] * target->n[1] +
            source->n[2] * target->n[2];
      G3 = (dot - 3.0 * targetCos * sourceCos) * oneOverR * tp1;
      G4 = tp2 * G3 - kappa * kappa * targetCos * sourceCos * Gk;
      L1 = sourceCos * tp1 * (1.0 - tp2 * epsilon);
      L2 = G0 - Gk;
      L3 = G4 - G3;
      L4 = targetCos * tp1 * (1.0 - tp2 / epsilon);

      sum0 += (L1 * x[j] + L2 * x[n + j]) * source->area;
      sum1 += (L3 * x[j] + L4 * x[n + j]) * source->area;
    }

    y[i] = beta * y[i] + alpha * (coeff0 * x[i] - sum0);
    y[n + i] = beta * y[n + i] + alpha * (coeff1 * x[n + i] - sum1);
  }
}

static void addTabiNodepatchKernelContribution(const TabiNodepatchNode *target,
                                               const double *sourceX,
                                               const double *sourceN,
                                               double q0, double q1,
                                               double *sum0, double *sum1) {
  double dx, dy, dz, r2, r, oneOverR, G0, Gk, kappaR, expKappaR;
  double sourceCos, targetCos, tp1, tp2, dot, G3, G4, L1, L2, L3, L4;

  dx = sourceX[0] - target->x[0];
  dy = sourceX[1] - target->x[1];
  dz = sourceX[2] - target->x[2];
  r2 = SQR(dx) + SQR(dy) + SQR(dz);
  if (r2 <= 0.0) {
    return;
  }

  r = sqrt(r2);
  oneOverR = 1.0 / r;
  G0 = fourPiI * oneOverR;
  kappaR = kappa * r;
  expKappaR = exp(-kappaR);
  Gk = expKappaR * G0;
  sourceCos = (sourceN[0] * dx + sourceN[1] * dy + sourceN[2] * dz) * oneOverR;
  targetCos = (target->n[0] * dx + target->n[1] * dy + target->n[2] * dz) * oneOverR;
  tp1 = G0 * oneOverR;
  tp2 = (1.0 + kappaR) * expKappaR;
  dot = sourceN[0] * target->n[0] +
        sourceN[1] * target->n[1] +
        sourceN[2] * target->n[2];
  G3 = (dot - 3.0 * targetCos * sourceCos) * oneOverR * tp1;
  G4 = tp2 * G3 - kappa * kappa * targetCos * sourceCos * Gk;
  L1 = sourceCos * tp1 * (1.0 - tp2 * epsilon);
  L2 = G0 - Gk;
  L3 = G4 - G3;
  L4 = targetCos * tp1 * (1.0 - tp2 / epsilon);

  *sum0 += L1 * q0 + L2 * q1;
  *sum1 += L3 * q0 + L4 * q1;
}

static void addTabiNodepatchClusterContribution(const TabiNodepatchNode *target,
                                                const TabiNodepatchTreeNode *node,
                                                double *sum0, double *sum1) {
  double dx, dy, dz, r2, r, oneOverR, G0, Gk, kappaR, expKappaR;
  double targetCos, tp1, tp2, dot, sourceCos, G3, G4, L2, L4;
  double l1Factor, l3Factor;

  dx = node->center[0] - target->x[0];
  dy = node->center[1] - target->x[1];
  dz = node->center[2] - target->x[2];
  r2 = SQR(dx) + SQR(dy) + SQR(dz);
  if (r2 <= 0.0) {
    return;
  }

  r = sqrt(r2);
  oneOverR = 1.0 / r;
  G0 = fourPiI * oneOverR;
  kappaR = kappa * r;
  expKappaR = exp(-kappaR);
  Gk = expKappaR * G0;
  targetCos = (target->n[0] * dx + target->n[1] * dy + target->n[2] * dz) * oneOverR;
  tp1 = G0 * oneOverR;
  tp2 = (1.0 + kappaR) * expKappaR;

  sourceCos = (node->src0n[0] * dx + node->src0n[1] * dy + node->src0n[2] * dz) * oneOverR;
  dot = node->src0n[0] * target->n[0] +
        node->src0n[1] * target->n[1] +
        node->src0n[2] * target->n[2];
  G3 = (dot - 3.0 * targetCos * sourceCos) * oneOverR * tp1;
  G4 = tp2 * G3 - kappa * kappa * targetCos * sourceCos * Gk;

  l1Factor = tp1 * (1.0 - tp2 * epsilon);
  l3Factor = G4 - G3;
  L2 = G0 - Gk;
  L4 = targetCos * tp1 * (1.0 - tp2 / epsilon);

  *sum0 += l1Factor * sourceCos + L2 * node->src1;
  *sum1 += l3Factor + L4 * node->src1;
}

static void walkTabiNodepatchTree(const TabiNodepatchTree *tree,
                                  const TabiNodepatchTreeNode *node,
                                  int targetIdx, double *sum0, double *sum1) {
  const TabiNodepatchMesh *mesh = tree->mesh;
  const TabiNodepatchNode *target = &mesh->nodes[targetIdx];
  double dx = target->x[0] - node->center[0];
  double dy = target->x[1] - node->center[1];
  double dz = target->x[2] - node->center[2];
  double dist = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
  int i, k;

  if (node->nKids > 0 && node->radius < tree->theta * dist) {
    addTabiNodepatchClusterContribution(target, node, sum0, sum1);
    return;
  }

  if (node->nKids == 0) {
    int n = mesh->nNodes;
    const double *x = NULL;
    (void)n;
    /*
     * The source values are not stored per leaf; exact near-field is handled
     * in applyTabiNodepatchTreeOperator() using the current vector.
     */
    (void)x;
    return;
  }

  for (k = 0; k < node->nKids; k++) {
    walkTabiNodepatchTree(tree, node->kids[k], targetIdx, sum0, sum1);
  }
}

static void walkTabiNodepatchTreeWithNear(const TabiNodepatchTree *tree,
                                          const TabiNodepatchTreeNode *node,
                                          const double *x,
                                          int targetIdx,
                                          double *sum0, double *sum1) {
  const TabiNodepatchMesh *mesh = tree->mesh;
  const TabiNodepatchNode *target = &mesh->nodes[targetIdx];
  double dx = target->x[0] - node->center[0];
  double dy = target->x[1] - node->center[1];
  double dz = target->x[2] - node->center[2];
  double dist = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
  int i, k, n = mesh->nNodes;

  if (node->nKids > 0 && node->radius < tree->theta * dist) {
    addTabiNodepatchClusterContribution(target, node, sum0, sum1);
    return;
  }

  if (node->nKids == 0) {
    for (i = 0; i < node->nIdx; i++) {
      int srcIdx = node->idx[i];
      const TabiNodepatchNode *source;
      double q0, q1;
      if (srcIdx == targetIdx) {
        continue;
      }
      source = &mesh->nodes[srcIdx];
      q0 = source->area * x[srcIdx];
      q1 = source->area * x[n + srcIdx];
      addTabiNodepatchKernelContribution(target, source->x, source->n, q0, q1, sum0, sum1);
    }
    return;
  }

  for (k = 0; k < node->nKids; k++) {
    walkTabiNodepatchTreeWithNear(tree, node->kids[k], x, targetIdx, sum0, sum1);
  }
}

static void applyTabiNodepatchTreeOperator(TabiNodepatchTree *tree,
                                           const double *x, double *y,
                                           double alpha, double beta) {
  const TabiNodepatchMesh *mesh = tree->mesh;
  int i, n = mesh->nNodes;
  double coeff0 = 0.5 * (1.0 + epsilon);
  double coeff1 = 0.5 * (1.0 + 1.0 / epsilon);

  clearTabiNodepatchTreeSources(tree->root);
  accumulateTabiNodepatchTreeSources(mesh, tree->root, x);

  for (i = 0; i < n; i++) {
    double sum0 = 0.0, sum1 = 0.0;
    walkTabiNodepatchTreeWithNear(tree, tree->root, x, i, &sum0, &sum1);
    y[i] = beta * y[i] + alpha * (coeff0 * x[i] - sum0);
    y[n + i] = beta * y[n + i] + alpha * (coeff1 * x[n + i] - sum1);
  }
}

static int MtVTabiNodepatch(double *alpha, double *x, double *beta, double *y) {
  if (gTabiNodepatchTree != NULL) {
    applyTabiNodepatchTreeOperator(gTabiNodepatchTree, x, y, *alpha, *beta);
  } else {
    applyTabiNodepatchOperator(gTabiNodepatchMesh, x, y, *alpha, *beta);
  }
  return 0;
}

static int PtVTabiNodepatchDiag(double *x, double *b) {
  int i, n = gTabiNodepatchMesh->nNodes;
  double coeff0 = 0.5 * (1.0 + epsilon);
  double coeff1 = 0.5 * (1.0 + 1.0 / epsilon);

  for (i = 0; i < n; i++) {
    x[i] = b[i] / coeff0;
    x[n + i] = b[n + i] / coeff1;
  }
  return 0;
}

static void setupTabiNodepatchRHS(ssystem *sys, const TabiNodepatchMesh *mesh, double *rhs) {
  int i, j, n = mesh->nNodes;

  for (i = 0; i < n; i++) {
    const TabiNodepatchNode *target = &mesh->nodes[i];
    double rhs0 = 0.0;
    double rhs1 = 0.0;

    for (j = 0; j < sys->nChar; j++) {
      double dx = sys->pos[3*j] - target->x[0];
      double dy = sys->pos[3*j + 1] - target->x[1];
      double dz = sys->pos[3*j + 2] - target->x[2];
      double r2 = SQR(dx) + SQR(dy) + SQR(dz);
      double r = sqrt(r2);
      double oneOverR = 1.0 / r;
      double G0 = fourPiI * oneOverR;
      double sourceCos = (target->n[0] * dx + target->n[1] * dy + target->n[2] * dz) * oneOverR;
      double G1 = sourceCos * G0 * oneOverR;

      rhs0 += sys->chr[j] * G0 / epsilon1;
      rhs1 += sys->chr[j] * G1 / epsilon1;
    }
    rhs[i] = rhs0;
    rhs[n + i] = rhs1;
  }
}

static void nodepatchChargeTreeRhsWalk(ssystem *sys, cube *chgCb,
                                       const TabiNodepatchNode *target,
                                       double theta,
                                       double *rhs0, double *rhs1) {
  double dx = chgCb->x[0] - target->x[0];
  double dy = chgCb->x[1] - target->x[1];
  double dz = chgCb->x[2] - target->x[2];
  double dist = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
  int i;

  if (chgCb->level >= sys->height && chgCb->eRad < theta * dist) {
    double q = (chgCb->mom_chr != NULL) ? chgCb->mom_chr[0] : 0.0;
    double r2 = SQR(dx) + SQR(dy) + SQR(dz);
    double r = sqrt(r2);
    double oneOverR = 1.0 / r;
    double G0 = fourPiI * oneOverR;
    double sourceCos = (target->n[0] * dx + target->n[1] * dy + target->n[2] * dz) * oneOverR;
    double G1 = sourceCos * G0 * oneOverR;
    *rhs0 += q * G0 / epsilon1;
    *rhs1 += q * G1 / epsilon1;
    return;
  }

  if (chgCb->level == sys->chgDepth) {
    for (i = 0; i < chgCb->nChgs; i++) {
      int j = chgCb->chgIdx[i];
      double rx = sys->pos[3*j] - target->x[0];
      double ry = sys->pos[3*j + 1] - target->x[1];
      double rz = sys->pos[3*j + 2] - target->x[2];
      double r2 = SQR(rx) + SQR(ry) + SQR(rz);
      double r = sqrt(r2);
      double oneOverR = 1.0 / r;
      double G0 = fourPiI * oneOverR;
      double sourceCos = (target->n[0] * rx + target->n[1] * ry + target->n[2] * rz) * oneOverR;
      double G1 = sourceCos * G0 * oneOverR;
      *rhs0 += sys->chr[j] * G0 / epsilon1;
      *rhs1 += sys->chr[j] * G1 / epsilon1;
    }
    return;
  }

  for (i = 0; i < chgCb->nKids; i++) {
    nodepatchChargeTreeRhsWalk(sys, chgCb->kids[i], target, theta, rhs0, rhs1);
  }
}

static void setupTabiNodepatchRHSTree(ssystem *sys, const TabiNodepatchMesh *mesh, double *rhs) {
  int i, n = mesh->nNodes;
  double theta = tabiNodepatchChargeTreeTheta();

  ensureChargeTreeBuilt(sys);
  for (i = 0; i < n; i++) {
    double rhs0 = 0.0, rhs1 = 0.0;
    nodepatchChargeTreeRhsWalk(sys, sys->chgCubeList[0], &mesh->nodes[i], theta, &rhs0, &rhs1);
    rhs[i] = rhs0;
    rhs[n + i] = rhs1;
  }
}

static double computeTabiNodepatchEnergy(ssystem *sys, const TabiNodepatchMesh *mesh, const double *solution) {
  int i, j, n = mesh->nNodes;
  double raw = 0.0;

  for (i = 0; i < n; i++) {
    const TabiNodepatchNode *target = &mesh->nodes[i];
    double potDD = 0.0;
    double potDX = 0.0;
    double potDY = 0.0;
    double potDZ = 0.0;

    for (j = 0; j < sys->nChar; j++) {
      double dx = target->x[0] - sys->pos[3*j];
      double dy = target->x[1] - sys->pos[3*j + 1];
      double dz = target->x[2] - sys->pos[3*j + 2];
      double r2 = SQR(dx) + SQR(dy) + SQR(dz);
      double r = sqrt(r2);
      double oneOverR = 1.0 / r;
      double G0 = fourPiI * oneOverR;
      double expKappaR = exp(-kappa * r);
      double L2 = G0 * (1.0 - expKappaR);
      double L1 = G0 * oneOverR * oneOverR *
                  (1.0 - epsilon * expKappaR * (1.0 + kappa * r));

      potDD += L2 * sys->chr[j];
      potDX += L1 * sys->chr[j] * dx;
      potDY += L1 * sys->chr[j] * dy;
      potDZ += L1 * sys->chr[j] * dz;
    }

    raw += solution[n + i] * target->area * potDD;
    raw += solution[i] * target->area *
           (target->n[0] * potDX + target->n[1] * potDY + target->n[2] * potDZ);
  }

  return raw * 8729.779593448 / 4.184;
}

static void nodepatchChargeTreeEnergyWalk(ssystem *sys, cube *chgCb,
                                          const TabiNodepatchNode *target,
                                          double theta,
                                          double *potDD,
                                          double *potDX,
                                          double *potDY,
                                          double *potDZ) {
  double dx = target->x[0] - chgCb->x[0];
  double dy = target->x[1] - chgCb->x[1];
  double dz = target->x[2] - chgCb->x[2];
  double dist = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
  int i;

  if (chgCb->level >= sys->height && chgCb->eRad < theta * dist) {
    double q = (chgCb->mom_chr != NULL) ? chgCb->mom_chr[0] : 0.0;
    double r2 = SQR(dx) + SQR(dy) + SQR(dz);
    double r = sqrt(r2);
    double oneOverR = 1.0 / r;
    double G0 = fourPiI * oneOverR;
    double expKappaR = exp(-kappa * r);
    double L2 = G0 * (1.0 - expKappaR);
    double L1 = G0 * oneOverR * oneOverR *
                (1.0 - epsilon * expKappaR * (1.0 + kappa * r));
    *potDD += L2 * q;
    *potDX += L1 * q * dx;
    *potDY += L1 * q * dy;
    *potDZ += L1 * q * dz;
    return;
  }

  if (chgCb->level == sys->chgDepth) {
    for (i = 0; i < chgCb->nChgs; i++) {
      int j = chgCb->chgIdx[i];
      double rx = target->x[0] - sys->pos[3*j];
      double ry = target->x[1] - sys->pos[3*j + 1];
      double rz = target->x[2] - sys->pos[3*j + 2];
      double r2 = SQR(rx) + SQR(ry) + SQR(rz);
      double r = sqrt(r2);
      double oneOverR = 1.0 / r;
      double G0 = fourPiI * oneOverR;
      double expKappaR = exp(-kappa * r);
      double L2 = G0 * (1.0 - expKappaR);
      double L1 = G0 * oneOverR * oneOverR *
                  (1.0 - epsilon * expKappaR * (1.0 + kappa * r));
      *potDD += L2 * sys->chr[j];
      *potDX += L1 * sys->chr[j] * rx;
      *potDY += L1 * sys->chr[j] * ry;
      *potDZ += L1 * sys->chr[j] * rz;
    }
    return;
  }

  for (i = 0; i < chgCb->nKids; i++) {
    nodepatchChargeTreeEnergyWalk(sys, chgCb->kids[i], target, theta,
                                  potDD, potDX, potDY, potDZ);
  }
}

static double computeTabiNodepatchEnergyTree(ssystem *sys, const TabiNodepatchMesh *mesh,
                                             const double *solution) {
  int i, n = mesh->nNodes;
  double raw = 0.0;
  double theta = tabiNodepatchChargeTreeTheta();

  ensureChargeTreeBuilt(sys);
  for (i = 0; i < n; i++) {
    const TabiNodepatchNode *target = &mesh->nodes[i];
    double potDD = 0.0, potDX = 0.0, potDY = 0.0, potDZ = 0.0;
    nodepatchChargeTreeEnergyWalk(sys, sys->chgCubeList[0], target, theta,
                                  &potDD, &potDX, &potDY, &potDZ);
    raw += solution[n + i] * target->area * potDD;
    raw += solution[i] * target->area *
           (target->n[0] * potDX + target->n[1] * potDY + target->n[2] * potDZ);
  }
  return raw * 8729.779593448 / 4.184;
}

static void runTabiNodepatchDiagnostic(ssystem *sys) {
  TabiNodepatchMesh *mesh, *sampleMesh;
  TabiNodepatchTree *tree = NULL;
  double *ones, *aOnes, *rhs, *solution, *work, *h;
  double area = 0.0, tol = 1.0e-4, energy, t0, t1;
  int i, n, maxNodes, sampleNodes, useTree, ldw, ldh, restart, maxIter, info;

  t0 = wall_seconds();
  mesh = buildTabiNodepatchMesh(sys);
  t1 = wall_seconds();
  n = mesh->nNodes;
  for (i = 0; i < n; i++) {
    area += mesh->nodes[i].area;
  }
  printf("TABI nodepatch diagnostic: vertices=%d panels=%d area=%f build=%f s\n",
         n, sys->nPnls, area, t1 - t0);
  fflush(stdout);

  sampleMesh = NULL;
  sampleNodes = tabiNodepatchSampleNodes();
  if (sampleNodes > 0 && sampleNodes < n) {
    sampleMesh = sampleTabiNodepatchMesh(mesh, sampleNodes);
    printf("TABI nodepatch diagnostic: using evenly spaced sample vertices=%d of %d for dense direct checks\n",
           sampleMesh->nNodes, n);
    freeTabiNodepatchMesh(mesh);
    mesh = sampleMesh;
    n = mesh->nNodes;
    fflush(stdout);
  }

  useTree = tabiNodepatchTreeEnabled();
  if (useTree) {
    t0 = wall_seconds();
    tree = buildTabiNodepatchTree(mesh);
    t1 = wall_seconds();
    printf("TABI nodepatch tree: leaf_max=%d theta=%f build=%f s\n",
           tree->leafMax, tree->theta, t1 - t0);
    fflush(stdout);
  }

  maxNodes = tabiNodepatchMaxNodes();
  if (!useTree && n > maxNodes) {
    printf("TABI nodepatch diagnostic skipped solve: vertices=%d exceeds "
           "FABIPB_TABI_NODEPATCH_MAX_VERTS=%d\n", n, maxNodes);
    freeTabiNodepatchMesh(mesh);
    return;
  }

  CALLOC(ones, 2*n, double);
  CALLOC(aOnes, 2*n, double);
  CALLOC(rhs, 2*n, double);
  CALLOC(solution, 2*n, double);
  for (i = 0; i < 2*n; i++) {
    ones[i] = 1.0;
  }
  if (tree != NULL) {
    applyTabiNodepatchTreeOperator(tree, ones, aOnes, 1.0, 0.0);
  } else {
    applyTabiNodepatchOperator(mesh, ones, aOnes, 1.0, 0.0);
  }
  printRhsStats("TABI nodepatch A*ones component-0", aOnes, n);
  printRhsStats("TABI nodepatch A*ones component-1", aOnes + n, n);

  t0 = wall_seconds();
  if (tree != NULL) {
    setupTabiNodepatchRHSTree(sys, mesh, rhs);
  } else {
    setupTabiNodepatchRHS(sys, mesh, rhs);
  }
  t1 = wall_seconds();
  printRhsStats("TABI nodepatch RHS component-0", rhs, n);
  printRhsStats("TABI nodepatch RHS component-1", rhs + n, n);
  printf("TABI nodepatch RHS %s time: %f s (vertices=%d charges=%d)\n",
         (tree != NULL) ? "tree" : "direct", t1 - t0, n, sys->nChar);

  gTabiNodepatchMesh = mesh;
  gTabiNodepatchTree = tree;
  ldw = 2*n;
  restart = (ldw < 50) ? ldw : 50;
  ldh = restart + 1;
  maxIter = tabiNodepatchMaxIter();
  CALLOC(work, (size_t)ldw * (size_t)(restart + 4), double);
  CALLOC(h, ldh * (restart + 2), double);
  t0 = wall_seconds();
  gmres(ldw, rhs, solution, restart, work, ldw, h, ldh,
        &maxIter, &tol, MtVTabiNodepatch, PtVTabiNodepatchDiag, &info);
  t1 = wall_seconds();
  printf("TABI nodepatch GMRES status: info=%d iterations=%d final-residual=%e time=%f s\n",
         info, maxIter, tol, t1 - t0);
  printRhsStats("TABI nodepatch solution component-0", solution, n);
  printRhsStats("TABI nodepatch solution component-1", solution + n, n);

  t0 = wall_seconds();
  if (tree != NULL) {
    energy = computeTabiNodepatchEnergyTree(sys, mesh, solution);
  } else {
    energy = computeTabiNodepatchEnergy(sys, mesh, solution);
  }
  t1 = wall_seconds();
  printf("TABI nodepatch solvation energy: %f kcal/mol (%s energy time=%f s)\n",
         energy, (tree != NULL) ? "tree" : "direct", t1 - t0);

  gTabiNodepatchMesh = NULL;
  gTabiNodepatchTree = NULL;
  freeTabiNodepatchTree(tree);
  free(work);
  free(h);
  free(ones);
  free(aOnes);
  free(rhs);
  free(solution);
  freeTabiNodepatchMesh(mesh);
}

typedef struct {
  ssystem *sys;
  panel **panels;
  int begin;
  int end;
  int qOrder;
  double fac;
  double *sgm;
} RhsTreeSetupTask;

static int rhsTreeThreadCount(int nTasks) {
  const char *env = getenv("FABIPB_RHS_THREADS");
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

static void *rhsTreeSetupWorker(void *arg) {
  RhsTreeSetupTask *task = (RhsTreeSetupTask *)arg;
  RhsTreeWorkspace ws;
  double intgr[2];
  int i;

  initRhsTreeWorkspace(task->sys, &ws);
  for (i = task->begin; i < task->end; i++) {
    panel *pnl = task->panels[i];
    panelRHSTreeWorkspace(task->sys, task->qOrder, pnl,
                          task->sys->chgCubeList[0], &ws, intgr);
    task->sgm[pnl->idx] = task->fac * intgr[0];
    task->sgm[task->sys->nPnls + pnl->idx] = task->fac * intgr[1];
  }
  freeRhsTreeWorkspace(&ws);
  return NULL;
}

static panel **getPanelArrayForRhs(ssystem *sys, int *owned) {
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
  if (i != sys->nPnls) {
    fprintf(stderr, "Error: panel list length mismatch while preparing RHS tree array\n");
    exit(1);
  }
  *owned = 1;
  return panels;
}

static void setupRHSTreeParallel(ssystem *sys, int qOrder, double fac, double *sgm) {
  int nPnls = sys->nPnls;
  int nThreads = rhsTreeThreadCount(nPnls);
  int ownPanels = 0;
  int t, created = 0, failed = 0;
  panel **panels = getPanelArrayForRhs(sys, &ownPanels);
  RhsTreeSetupTask *tasks;
  pthread_t *threads;
  double theta = rhsTreeTheta();

  if (sys->benchmarkMode > 0) {
    printf("setupRHS tree evaluator: threads=%d panels=%d theta=%g\n",
           nThreads, nPnls, theta);
  }

  CALLOC(tasks, nThreads, RhsTreeSetupTask);
  CALLOC(threads, nThreads, pthread_t);

  for (t = 0; t < nThreads; t++) {
    int begin = (int)(((long long)nPnls * t) / nThreads);
    int end = (int)(((long long)nPnls * (t + 1)) / nThreads);
    tasks[t].sys = sys;
    tasks[t].panels = panels;
    tasks[t].begin = begin;
    tasks[t].end = end;
    tasks[t].qOrder = qOrder;
    tasks[t].fac = fac;
    tasks[t].sgm = sgm;
  }

  if (nThreads == 1) {
    rhsTreeSetupWorker(&tasks[0]);
  } else {
    for (t = 0; t < nThreads; t++) {
      if (pthread_create(&threads[t], NULL, rhsTreeSetupWorker, &tasks[t]) != 0) {
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
              "Warning: pthread_create failed for setupRHS tree; recomputing serially\n");
      memset(sgm, 0, (size_t)(2 * nPnls) * sizeof(double));
      tasks[0].begin = 0;
      tasks[0].end = nPnls;
      rhsTreeSetupWorker(&tasks[0]);
    }
  }

  free(threads);
  free(tasks);
  if (ownPanels) {
    free(panels);
  }
}

static void setupRHSTree(ssystem *sys, int qOrder, double fac, double *sgm) {
  int i, j;
  int nPnls = sys->nPnls, nChar = sys->nChar;
  double *intgr;
  panel *pnl;

  ensureChargeTreeBuilt(sys);
  setupRHSTreeParallel(sys, qOrder, fac, sgm);
  if (debugCompareRhs()) {
    double *sgmDirect;
    CALLOC(sgmDirect, 2*nPnls, double);
    for ( i=0, pnl=sys->pnlLst; pnl!=NULL; pnl=pnl->nextC, i++ ) {
      sgmDirect[i] = 0.0; sgmDirect[nPnls+i] = 0.0;
      for ( j=0; j<nChar; j++ ) {
        intgr=panelRHS(qOrder, pnl, &sys->pos[3*j]);
        sgmDirect[i] += sys->chr[j]*intgr[0];
        sgmDirect[i+nPnls] += sys->chr[j]*intgr[1];
      }
      sgmDirect[i] *= fac;
      sgmDirect[nPnls+i] *= fac;
    }
    compareSetupRhsOnce(sys, sgmDirect, sgm);
    free(sgmDirect);
  }
  if (debugRhsNorms()) {
    compareRhsToCentroidSource(sys, sgm);
  }
}

void setupRHS(ssystem *sys, double *sgm) {
  int i, j;
  int nPnls = sys->nPnls, nChar = sys->nChar, qOrder=sys->maxQuadOrder;
  unsigned long long rhsPairs = countDirectRhsPairs(nPnls, nChar);
  unsigned long long maxDirectRhsPairs = getActiveDirectRhsPairLimit(sys->gpuMode);
  int useTree = shouldUseTreeRhs(nPnls, nChar, sys->gpuMode);
  double *intgr, fac;
  panel *pnl;
  static int warnedGpuRHS = 0;

  fac = fourPiI/epsilon1;

  if (sys->benchmarkMode > 0) {
    printf("setupRHS direct pairs: panels=%d charges=%d pairs=%llu limit=%llu mode=%s",
           nPnls, nChar, rhsPairs, maxDirectRhsPairs, useTree ? "tree" : "direct");
    if (useTree) {
      printf(" theta=%g", rhsTreeTheta());
    }
    printf("\n");
  }

  if (useTree) {
    setupRHSTree(sys, qOrder, fac, sgm);
    writeRhsSummaryIfRequested(sys, sgm);
    return;
  }

  if (sys->gpuMode > 0 && gpuSetupRHS(sys, qOrder, fac, sgm)) {
    writeRhsSummaryIfRequested(sys, sgm);
    return;
  }
  if (sys->gpuMode > 0 && gpuBackendAvailable() && !warnedGpuRHS) {
    printf("GPU RHS path unavailable; using CPU setupRHS.\n");
    warnedGpuRHS = 1;
  }
  if (rhsPairs > getMaxDirectRhsPairs() && !allowLargeDirectRhs()) {
    printf("GPU RHS path unavailable above CPU direct cap; using tree-accelerated RHS.\n");
    setupRHSTree(sys, qOrder, fac, sgm);
    writeRhsSummaryIfRequested(sys, sgm);
    return;
  }

  /* triangles order for Direct */
  for ( i=0, pnl=sys->pnlLst; pnl!=NULL; pnl=pnl->nextC, i++ ) {
    sgm[i] = 0.0; sgm[nPnls+i] = 0.0;
    for ( j=0; j<nChar; j++ ) {
      intgr=panelRHS(qOrder, pnl, &sys->pos[3*j]);
      sgm[i] += sys->chr[j]*intgr[0];
      sgm[i+nPnls] += sys->chr[j]*intgr[1];
    }
    sgm[i] *= fac;
    sgm[nPnls+i] *= fac;
  }
  if (debugRhsNorms()) {
    compareRhsToCentroidSource(sys, sgm);
  }
  writeRhsSummaryIfRequested(sys, sgm);
} /* setupRHS */



int main(int nargs, char *argv[]){
  char panelfile[80], meshParam[80];
  int order=-1, image=0, refineLev=0, numSurfOne=1;
  int i, j, k, n, nPnls, nChar;
  int numItr=100, arnoldiSz=30, ldw, ldh;
  panel *inputLst, *pnl;
  cube *cb;
  double tolpar=1.0e-4, para=332.0716;
  double *sgm, *pot, *GMRES_work, *GMRES_h, ptl;
  static int info;
  double meshResolution = 1.0;
  double meshOverrideValue = 0.0;
  double meshBackendValue = 0.0;
  double meshControlValue = 0.0;
  int meshOverrideSet = 0;
  int meshOnlyMode = 0;
  const char *meshControlName = NULL;
  const char *backendParamName = NULL;

  double start_t, end_t;
  double stage_t0, loadPanel_t, gkInit_t, setupFMM_t_local;
  double setupPC_t, setupRHS_t, gmres_t, treecode_t;

  ensure_startup_thread_env(nargs, argv);
  CALLOC(sys, 1, ssystem);
  set_benchmark_thread_defaults();
  sys->height = 2;
  sys->maxSepRatio = 0.8;
  sys->maxQuadOrder = 1;
  sys->nKerl = 4;
  sys->depth = 5;
  sys->mesh_flag = 1;
  sys->benchmarkMode = 0;
  sys->gpuMode = -1;
  sys->debugCompareApply = 0;
  sys->debugComparePrecond = 0;
  sys->matvecMode = 0;
  sys->gpuQ2MMode = 0;
  sys->gpuNearfieldMode = 1;
  sys->precondCacheMode = 2;
  double bulk_strength = 0.15;
  //kappa = sqrt(8.430325455*bulk_strength/epsilon2);
  kappa = 0.1257;

  /* parse the command line */
  panelfile[0] = 0;
  for ( i=1; i<nargs; i++ )
    if ( argv[i][0] == '-' )
      switch ( argv[i][1] ) {
        case 'h':
          print_usage(argv[0]);
          return 0;
        case 'S': sys->maxSepRatio = atof( argv[i]+3 );
          break;
        case 'o': tolpar = atof( argv[i]+3 );
          break;
        case 'a':
          arnoldiSz = atoi(argv[i]+3);
          if (arnoldiSz <= 0) {
            printf("Bad GMRES Arnoldi dimension: %s\n", argv[i]+3);
            exit(1);
          }
          break;
        case 'i':
          numItr = atoi(argv[i]+3);
          if (numItr <= 0) {
            printf("Bad GMRES maximum iterations: %s\n", argv[i]+3);
            exit(1);
          }
          break;
        case 'p':
          if ( argv[i][2] == '=' ) order = atoi( argv[i]+3 );
          if ( argv[i][2] == 'm' ) orderMom = atoi( argv[i]+4 );
          break;
        case 'q':
          sys->maxQuadOrder = atoi( argv[i]+3 );
          break;
        case 't': sys->depth = atoi( argv[i]+3 );
          break;
        case 'H': sys->height = atoi( argv[i]+3 );
          break;
        case 'R':
          if (!parse_double_arg(argv[i] + 3, &meshResolution)) {
            printf("Bad mesh resolution: %s\n", argv[i] + 3);
            exit(0);
          }
          break;
        case 'e':
          if ( argv[i][4] == '1' ) epsilon1 = atof( argv[i]+6 );
          if ( argv[i][4] == '2' ) epsilon2 = atof( argv[i]+6 );
          break;
        case 'k': kappa = atof( argv[i]+3 );
          break;
        case 'm': sys->mesh_flag = atoi( argv[i]+3 );
          break;
        case 'M': meshOnlyMode = atoi( argv[i]+3 );
          break;
        case 'B': sys->benchmarkMode = atoi( argv[i]+3 );
          break;
        case 'g': sys->gpuMode = atoi( argv[i]+3 );
          break;
        case 'c': sys->debugCompareApply = atoi( argv[i]+3 );
          break;
        case 'C': sys->debugComparePrecond = atoi( argv[i]+3 );
          break;
        case 'r': sys->matvecMode = atoi( argv[i]+3 );
          break;
        case 'Q': sys->gpuQ2MMode = atoi( argv[i]+3 );
          break;
        case 'G': sys->gpuNearfieldMode = atoi( argv[i]+3 );
          break;
        case 'P': sys->precondCacheMode = atoi( argv[i]+3 );
          break;
        case 'd':
          if (!parse_double_arg(argv[i] + 3, &meshOverrideValue)) {
            printf("Bad mesh override: %s\n", argv[i] + 3);
            exit(0);
          }
          meshOverrideSet = 1;
          break;
      }
    else {
      strcpy(panelfile,argv[i]);
    }

  epsilon = epsilon2/epsilon1;
  if ( panelfile[0] == 0 ) {
    printf("\n Name of the panel file > ");
    if ( scanf("%s",panelfile) < 1 ) {
      printf("PDB name input failed\n");
      exit(0);
    }
  }
  if ( sys->depth < 0 ) {
    printf("Select tree depth > ");
    if ( scanf("%d", &sys->depth) < 1 ) {
      printf("PDB density input failed\n");
      exit(0);
    }
    if( sys->depth < 1 ) {
      printf("Bad tree depth: %d\n", sys->depth );
      exit(0);
    }
  }
  if ( sys->height < 0 ) {
    printf("Bad FMM height: %d\n", sys->height );
    exit(0);
  }
  if ( sys->height > sys->depth ) {
    printf("Bad FMM level range: height=%d depth=%d\n", sys->height, sys->depth );
    exit(0);
  }
  if (sys->mesh_flag != 1 && sys->mesh_flag != 2) {
    printf("Bad mesh mode: %d (use -m=1 for MSMS or -m=2 for NanoShaper)\n", sys->mesh_flag);
    exit(0);
  }
  if (meshResolution <= 0.0) {
    printf("Bad mesh resolution: %g (must be > 0)\n", meshResolution);
    exit(0);
  }
  if (meshOverrideSet && meshOverrideValue <= 0.0) {
    printf("Bad mesh override: %g (must be > 0)\n", meshOverrideValue);
    exit(0);
  }
  if (!resolve_mesh_parameter(sys->mesh_flag, meshOverrideSet, meshOverrideValue,
                              meshResolution, &meshBackendValue,
                              &meshControlName, &meshControlValue,
                              &backendParamName)) {
    printf("Failed to resolve mesh control parameters for mesh mode %d\n", sys->mesh_flag);
    exit(0);
  }
  snprintf(meshParam, sizeof(meshParam), "%.12g", meshBackendValue);
  //printf("PDB id: %s, MSMS density: %s\n", panelfile, density);
  if (sys->gpuMode < 0) {
    sys->gpuMode = gpuBackendAvailable() ? 1 : 0;
  }

  if (sys->benchmarkMode > 0) {
    if (missing_external_thread_env()) {
      fprintf(stderr,
              "Warning: BLAS/OpenMP thread env vars were not exported before startup. "
              "For reproducible benchmarking, prefer scripts/with_benchmark_env.sh because "
              "late in-process defaults may be too late for some libraries.\n");
    }
    printf("----------------------------\n");
    if (sys->matvecMode == 0) {
      printf("FMM variables: nLev=%d height=%d ord=%d SepRat=%lg qOrd=%d\n",
        sys->depth, sys->height, order, sys->maxSepRatio, sys->maxQuadOrder );
    } else {
      printf("Direct baseline mode enabled (no FMM matvec)\n");
    }
    printf("GMRES variables: tol=%1.e arnoldiSz=%d maxIt=%d\n",
      tolpar, arnoldiSz, numItr);
    printf("kappa=%f, eps1=%f, eps2=%f\n", kappa, epsilon1, epsilon2);
    printf("GPU mode=%d (0=CPU, 1=GPU)\n", sys->gpuMode);
    printf("Mesh control=%s (%g) -> %s=%g\n",
           meshControlName, meshControlValue, backendParamName, meshBackendValue);
    printf("Matvec mode=%d (0=FMM, 1=direct GPU baseline, 2=direct CPU baseline)\n",
           sys->matvecMode);
    if (sys->matvecMode == 0) {
      printf("GPU Q2M mode=%d (0=CPU default, 1=GPU debug)\n", sys->gpuQ2MMode);
      printf("GPU nearfield mode=%d (0=interaction, 1=destination-leaf)\n", sys->gpuNearfieldMode);
    }
    printf("Preconditioner mode=%d (-1=disabled, 0=original, 1=cached-blocks, 2=cached-LU, 3=diagonal)\n", sys->precondCacheMode);
    if (sys->debugCompareApply > 0 || sys->debugComparePrecond > 0) {
      printf("Debug compare flags: apply=%d precond=%d\n",
             sys->debugCompareApply, sys->debugComparePrecond);
    }
  }
  //printf("----------------------------\n");


  /*
   * get panels by msms from pqr
   * or use the panel on sphere test example
   */
  start_t = wall_seconds();
  stage_t0 = start_t;
  inputLst = loadPanel(panelfile, meshParam, &nPnls, sys,
                       meshControlName, meshControlValue,
                       backendParamName, meshBackendValue);
  if (meshOnlyMode > 0) {
    printf("Mesh-only mode: stopping after mesh generation.\n");
    return 0;
  }
  reportDirectRhsLimit(nPnls, sys->nChar, sys->gpuMode);
  loadPanel_t = wall_seconds() - stage_t0;
  sys->pnlOLst = inputLst;

  stage_t0 = wall_seconds();
  gkInit(sys, inputLst, order, orderMom);
  gkInit_t = wall_seconds() - stage_t0;

  CALLOC(sgm, 2*nPnls, double);
  CALLOC(pot, 2*nPnls, double);

  stage_t0 = wall_seconds();
  setupFMM(sys);
  setupFMM_t_local = wall_seconds() - stage_t0;
  if (sys->matvecMode != 0 || debugOperatorNorms()) {
    /* Direct matvec modes still use treecode/FMM-side data for postprocessing. */
    buildPanelIndexDirect(sys);
  }

  if (debugPanelCases()) {
    reportPanelCaseDistribution(sys);
    if (getenv("FABIPB_STOP_AFTER_PANEL_CASES") != NULL) {
      printf("FABIPB_STOP_AFTER_PANEL_CASES set: stopping after panel case diagnostic.\n");
      printf("Stage times (s): loadPanel=%f gkInit=%f setupFMM=%f\n",
             loadPanel_t, gkInit_t, setupFMM_t_local);
      return 0;
    }
  }

  if (debugDumpDenseSystem()) {
    dumpDenseSystemForAnalysis(sys);
    if (getenv("FABIPB_STOP_AFTER_DENSE_DUMP") != NULL) {
      printf("FABIPB_STOP_AFTER_DENSE_DUMP set: stopping after dense system dump.\n");
      printf("Stage times (s): loadPanel=%f gkInit=%f setupFMM=%f\n",
             loadPanel_t, gkInit_t, setupFMM_t_local);
      return 0;
    }
  }

  if (debugPanelAllTypes()) {
    reportPanelAllTypesDiagnostics(sys);
    if (getenv("FABIPB_STOP_AFTER_PANEL_ALL_TYPES") != NULL ||
        getenv("FABIPB_STOP_AFTER_PANEL_ONE_COMMON") != NULL) {
      printf("FABIPB_STOP_AFTER_PANEL_ALL_TYPES set: stopping after all-types panel diagnostic.\n");
      printf("Stage times (s): loadPanel=%f gkInit=%f setupFMM=%f\n",
             loadPanel_t, gkInit_t, setupFMM_t_local);
      return 0;
    }
  }

  if (debugTabiNodepatch()) {
    runTabiNodepatchDiagnostic(sys);
    if (getenv("FABIPB_STOP_AFTER_TABI_NODEPATCH") != NULL) {
      printf("FABIPB_STOP_AFTER_TABI_NODEPATCH set: stopping after nodepatch diagnostic.\n");
      printf("Stage times (s): loadPanel=%f gkInit=%f setupFMM=%f\n",
             loadPanel_t, gkInit_t, setupFMM_t_local);
      return 0;
    }
  }

  if (debugOperatorNorms()) {
    compareOperatorToTabiCollocation(sys);
    if (getenv("FABIPB_STOP_AFTER_OPERATOR") != NULL) {
      printf("FABIPB_STOP_AFTER_OPERATOR set: stopping after operator diagnostic.\n");
      printf("Stage times (s): loadPanel=%f gkInit=%f setupFMM=%f\n",
             loadPanel_t, gkInit_t, setupFMM_t_local);
      return 0;
    }
  }

  if (getenv("FABIPB_STOP_AFTER_RHS") != NULL) {
    /* RHS-only smoke test: skip the (untested-at-this-scale) preconditioner
     * build and GMRES entirely, so a large mesh's tree-accelerated setupRHS
     * can be timed on its own before committing to a full solve attempt. */
    stage_t0 = wall_seconds();
    setupRHS(sys, sgm);
    setupRHS_t = wall_seconds() - stage_t0;
    printf("FABIPB_STOP_AFTER_RHS set: stopping after setupRHS.\n");
    printf("Stage times (s): loadPanel=%f gkInit=%f setupFMM=%f setupRHS=%f\n",
           loadPanel_t, gkInit_t, setupFMM_t_local, setupRHS_t);
    return 0;
  }

  stage_t0 = wall_seconds();
  setupPreconditioning(sys);
  setupPC_t = wall_seconds() - stage_t0;

  stage_t0 = wall_seconds();
  setupRHS(sys, sgm);
  setupRHS_t = wall_seconds() - stage_t0;
  if (sys->debugCompareApply > 0) {
    compareApplyFMMOnce(sys, sgm);
  }
  if (sys->debugComparePrecond > 0) {
    comparePrecondOnce(sys, sgm);
  }
  for ( i=0; i<2*sys->nPnls; i++ ) pot[i] = sgm[i];
  {
    const char *initialMode = gmresInitialMode();
    if (strcmp(initialMode, "zero") == 0) {
      for ( i=0; i<2*sys->nPnls; i++ ) sgm[i] = 0.0;
    }
    if (sys->benchmarkMode > 0) {
      printf("GMRES initial guess mode: %s\n", initialMode);
    }
  }

  MtV = MtVmain;
  PtV = PtVmain;
  ldw = 2*nPnls;
  ldh = arnoldiSz+1;

  {
    size_t gmresWorkCount = (size_t)ldw * (size_t)(arnoldiSz + 4);
    if (sys->benchmarkMode > 0) {
      printf("GMRES workspace: dimension=%d restart=%d storage=%.3f GiB\n",
             ldw, arnoldiSz,
             (double)(gmresWorkCount * sizeof(double)) /
                 (1024.0 * 1024.0 * 1024.0));
    }
    CALLOC(GMRES_work, gmresWorkCount, double);
  }
  CALLOC(GMRES_h, ldh*(arnoldiSz+2), double);

  resetFmmMatvecStats();
  resetGmresStats();
  stage_t0 = wall_seconds();
  gmres(ldw, pot, sgm, arnoldiSz, GMRES_work, ldw, GMRES_h, ldh,
        &numItr, &tolpar, MtV, PtV, &info);
  gmres_t = wall_seconds() - stage_t0;
  printf("GMRES status: info=%d iterations=%d final-residual=%e\n",
         info, numItr, tolpar);
  if (debugSolutionNorms()) {
    printRhsStats("solution component-0", sgm, nPnls);
    printRhsStats("solution component-1", sgm + nPnls, nPnls);
  }
  writeSolutionIfRequested(sys, sgm);

  if (getenv("FABIPB_STOP_AFTER_GMRES") != NULL) {
    treecode_t = 0.0;
    end_t = wall_seconds() - start_t;
    printf("FABIPB_STOP_AFTER_GMRES set: skipping post-GMRES treecode energy.\n");
    printf("ttl time: %f, gmres-its=%d\n", end_t, numItr);
    printf("solvation energy: skipped\n");
  } else {
    stage_t0 = wall_seconds();
    if (debugEnergyComponents()) {
      double p0, p1;
      applyTreecodeComponents(sys, sgm, &p0, &p1);
      ptl = p0 + p1;
      printf("Energy component debug: raw_component0=%e scaled_component0=%e "
             "raw_component1=%e scaled_component1=%e raw_total=%e scale=%e\n",
             p0, p0*twoPi*para, p1, p1*twoPi*para, ptl, twoPi*para);
    } else {
      applyTreecode( sys, sgm, &ptl );
    }
    treecode_t = wall_seconds() - stage_t0;
    ptl *= twoPi*para;
    end_t = wall_seconds() - start_t;
    printf("ttl time: %f, gmres-its=%d\n", end_t, numItr);
    printf("solvation energy: %f\n", ptl);
  }
  if (sys->benchmarkMode > 0) {
    printf("Top-level stage times (s): loadPanel=%.6f gkInit=%.6f setupFMM=%.6f setupPC=%.6f setupRHS=%.6f gmres=%.6f treecode=%.6f\n",
           loadPanel_t, gkInit_t, setupFMM_t_local, setupPC_t, setupRHS_t, gmres_t, treecode_t);
  }
  printSetupFmmStats();
  printPrecondSetupStats();
  printPrecondApplyStats();
  printGmresStats(gmres_t);
  if (sys->benchmarkMode > 0 && sys->matvecMode == 0) {
    printFmmMatvecStats();
  } else if (sys->benchmarkMode > 0) {
    printf("Direct baseline run: FMM stage stats omitted.\n");
    if (sys->gpuMode > 0 && sys->matvecMode == 1) {
      printDirectMatvecStats();
    }
  }

}



/*
 * Matrix times Vector, subroutine of the iterative solver
 * the vector sgm and the result pot are ordered contiguously within cubes
 */
int MtVmain(double *alpha, double *sgm, double *beta, double *pot) {
  int i, lev, inc=1, nPnls = sys->nPnls;
  cube *cb;
  panel *pnl;
  double scale1, scale2, inv_beta;
  double callStart, callEnd, applyStart, applyEnd;
  static int warnedDirectGpu = 0;
  static int warnedDirectCpu = 0;

  scale1 = (1.0+epsilon)/2.0*(*alpha);
  scale2 = (1.0+1.0/epsilon)/2.0*(*alpha);

  callStart = wall_seconds();
  inv_beta = -(*beta);
  applyStart = wall_seconds();
  if (sys->matvecMode == 1) {
    if (!gpuDirectApply(sys, *alpha, inv_beta, sgm, pot)) {
      if (!warnedDirectGpu) {
        printf("Direct GPU matvec unavailable; using FMM path.\n");
        warnedDirectGpu = 1;
      }
      applyFMM(sys, alpha, sgm, &inv_beta, pot);
    }
  } else if (sys->matvecMode == 2) {
    if (!cpuDirectApply(sys, *alpha, inv_beta, sgm, pot)) {
      if (!warnedDirectCpu) {
        printf("Direct CPU matvec unavailable; using FMM path.\n");
        warnedDirectCpu = 1;
      }
      applyFMM(sys, alpha, sgm, &inv_beta, pot);
    }
  } else {
    applyFMM(sys, alpha, sgm, &inv_beta, pot);
  }
  applyEnd = wall_seconds();
  for (  i=0, pnl=sys->pnlLst; pnl!=NULL; pnl=pnl->nextC, i++ ) {
    pot[i] = (scale1*pnl->area*sgm[i]-pot[i]);
    pot[i+nPnls] = scale2*pnl->area*sgm[i+nPnls]-pot[i+nPnls];
  }
  callEnd = wall_seconds();

  mtvCalls++;
  mtvApplyFMMTime += (applyEnd - applyStart);
  mtvTotalTime += (callEnd - callStart);

  return 0;
} /* MtVmain */

int PtVmain(double *pot, double *sgm) {
  int i;
  int n = 2 * sys->nPnls;

  if (sys->precondCacheMode < 0) {
    for (i = 0; i < n; i++) {
      pot[i] = sgm[i];
    }
    return 0;
  }
  if (sys->precondCacheMode == 3) {
    return PtVfmmDiagonal(pot, sgm);
  }
  if (sys->precondCacheMode > 1) {
    return PtVfmmCachedLU(pot, sgm);
  }
  if (sys->precondCacheMode > 0) {
    return PtVfmmCached(pot, sgm);
  }
  return PtVfmm(pot, sgm);
}
