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
void applyPanelChargeTreeEnergy( ssystem *sys, double *sgm, double *pot );

/*
 * Pin the *external* math-library thread counts to 1.
 *
 * These control BLAS/LAPACK/OpenMP internal parallelism, where the reduction
 * order is not fixed, so leaving them free makes results vary run to run.
 * Pinning them keeps a run reproducible.
 *
 * FABIPB's own worker-thread counts are deliberately NOT in this list. They
 * partition their loops into disjoint output slots, so the result is identical
 * at any thread count (verified: the nearfield build gives the same energy to
 * every printed digit at 1, 4, 16 and 32 threads). Pinning them here cost real
 * time for no reproducibility benefit -- the nearfield host build alone ran
 * 9.52 s single-threaded versus 2.73 s at 32 threads on a 420k-panel mesh, and
 * the production runner never overrode it. Benchmark runs still get a fully
 * pinned environment from scripts/with_benchmark_env.sh, which sets each of
 * them explicitly.
 */
static void set_benchmark_thread_defaults(void) {
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

/*
 * Only the external math libraries are listed here. They read their thread
 * count once at library init, which can happen before main(), so setting the
 * variable has to be followed by a re-exec to take effect. FABIPB's own worker
 * counts are read lazily through getenv() at each use, need no re-exec, and are
 * left unpinned -- see set_benchmark_thread_defaults() for why.
 */
static void ensure_startup_thread_env(int nargs, char *argv[]) {
  const char *vars[] = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS"
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

/*
 * Option-parsing helpers.
 *
 * These replace the previous "switch on argv[i][1], read the value at a fixed
 * offset" scheme (atoi(argv[i]+3), argv[i][4]=='1'). That scheme silently
 * misparsed any spelling whose length differed from the one it assumed, and
 * bare atoi/atof turned every typo into 0 rather than an error:
 *
 *   -e1=4      the form printed by -h, tested argv[i][4] which only lands on
 *              '1' for the 6-character "-eps1=" prefix, so the dielectric
 *              silently stayed at its default and the answer was ~4x wrong
 *   -kappa=..  matched case 'k', then atof("pa=...") == 0, silently disabling
 *              Debye screening
 *   -q=0       accepted, leaving every quadrature loop with zero iterations,
 *              which reports "solvation energy: 0.000000" as a clean success
 *
 * Matching the full option name and validating the value removes that whole
 * class of failure: a mistyped flag now stops the run instead of quietly
 * changing the physics.
 */
static const char *optValue(const char *arg, const char *name) {
  size_t n = strlen(name);

  if (arg[0] != '-') {
    return NULL;
  }
  if (strncmp(arg + 1, name, n) != 0 || arg[1 + n] != '=') {
    return NULL;
  }
  return arg + 1 + n + 1;
}

static double optDouble(const char *arg, const char *v, double lo, double hi) {
  double out;

  if (!parse_double_arg(v, &out)) {
    fprintf(stderr, "Error: bad option '%s': expected a number after '='\n", arg);
    exit(1);
  }
  if (!(out >= lo && out <= hi)) {
    fprintf(stderr, "Error: bad option '%s': %g is outside the accepted range [%g, %g]\n",
            arg, out, lo, hi);
    exit(1);
  }
  return out;
}

static int optInt(const char *arg, const char *v, int lo, int hi) {
  char *endptr = NULL;
  long out;

  errno = 0;
  out = strtol(v, &endptr, 10);
  if (errno != 0 || endptr == v || *endptr != '\0') {
    fprintf(stderr, "Error: bad option '%s': expected an integer after '='\n", arg);
    exit(1);
  }
  if (out < (long)lo || out > (long)hi) {
    fprintf(stderr, "Error: bad option '%s': %ld is outside the accepted range [%d, %d]\n",
            arg, out, lo, hi);
    exit(1);
  }
  return (int)out;
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
  printf("  -Q=0|1    CPU dgemv loop or GPU Q2M path (default: 1)\n");
  printf("  -G=0|1    interaction or destination-leaf GPU nearfield (default: 1)\n");
  printf("  -P=-1|0|1|2|3  disabled, original, cached-block, cached-LU, or diagonal/Jacobi preconditioner (default: 2)\n");
  printf("  -t=<lev>  tree depth\n");
  printf("  -H=<lev>  coarsest active FMM level (default: 2)\n");
  printf("  -q=<ord>  panel quadrature order\n");
  printf("  -k=<val>  Debye-Huckel kappa\n");
  printf("  -eps1=<val> solute dielectric   (alias: -e1=<val>)\n");
  printf("  -eps2=<val> solvent dielectric  (alias: -e2=<val>)\n");
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
  printf("Post-solve energy evaluation:\n");
  printf("  FABIPB_ENERGY_MODE=charge-tree|panel-tree|compare  energy evaluator (default: charge-tree)\n");
  printf("                            panel-tree is threaded and more accurate; compare runs both and reports the diff\n");
  printf("  FABIPB_ENERGY_TREE_THETA=<x> panel-tree acceptance ratio (default 0.2; independent of the RHS ratio)\n");
  printf("  FABIPB_ENERGY_THREADS=<n> panel-tree worker threads (default: online CPUs, max 128)\n");
  printf("Diagnostics (optional; disabled unless explicitly enabled):\n");
  printf("  FABIPB_RHS_SUMMARY_PATH=<path> write raw and TABI-style RHS summary CSV\n");
  printf("  FABIPB_RHS_SAMPLE_PATH=<path> write sampled RHS rows after setupRHS\n");
  printf("  FABIPB_RHS_SAMPLE_STRIDE=<n> sample every nth RHS row (default: 1000)\n");
  printf("  FABIPB_GMRES_INITIAL=rhs|zero  choose GMRES initial guess (default: rhs)\n");
  printf("  FABIPB_GMRES_STOP_AFTER_ITER=<n>  debug stop after n GMRES iterations\n");
  printf("  FABIPB_WRITE_SOLUTION=<path>  write post-GMRES panel solution CSV\n");
  printf("  FABIPB_SKIP_PANEL_CASES=one,two,self  diagnostic: skip selected singular panel topologies\n");
  printf("  FABIPB_REUSE_MESH=1       preserve and reuse existing .face/.vert files\n");
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

static const char *energyMode(void) {
  const char *env = getenv("FABIPB_ENERGY_MODE");

  if (env == NULL || env[0] == '\0') {
    return "charge-tree";
  }
  if (strcmp(env, "charge-tree") == 0 ||
      strcmp(env, "panel-tree") == 0 ||
      strcmp(env, "compare") == 0) {
    return env;
  }
  fprintf(stderr,
          "Warning: ignoring invalid FABIPB_ENERGY_MODE='%s'; using charge-tree\n",
          env);
  return "charge-tree";
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
  ensureChargeTreeBuilt(sys);
  /* Same walk as the threaded CPU path below, on the device. Falls back
   * whenever the GPU is unavailable or the quadrature order is not the
   * single-point rule the kernel assumes. */
  if (sys->gpuMode > 0 && gpuChargeTreeRHS(sys, qOrder, fac, sgm)) {
    return;
  }
  setupRHSTreeParallel(sys, qOrder, fac, sgm);
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
  writeRhsSummaryIfRequested(sys, sgm);
} /* setupRHS */



int main(int nargs, char *argv[]){
  char panelfile[80], meshParam[80];
  int order=-1;
  int i, nPnls;
  int numItr=100, arnoldiSz=30, ldw, ldh;
  panel *inputLst;
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
  sys->gpuQ2MMode = 1;
  sys->gpuNearfieldMode = 1;
  sys->precondCacheMode = 2;
  //kappa = sqrt(8.430325455*bulk_strength/epsilon2); // bulk_strength = 0.15
  kappa = 0.1257;

  /* parse the command line */
  panelfile[0] = 0;
  for ( i=1; i<nargs; i++ ) {
    const char *arg = argv[i];
    const char *v;

    if ( arg[0] != '-' ) {
      if (strlen(arg) >= sizeof(panelfile)) {
        fprintf(stderr, "Error: input path exceeds %zu characters: %s\n",
                sizeof(panelfile) - 1, arg);
        exit(1);
      }
      strcpy(panelfile, arg);
      continue;
    }

    if ( strcmp(arg, "-h") == 0 ) {
      print_usage(argv[0]);
      return 0;
    }

    /* order/orderMom accept negatives: they select adaptive-order policies
     * (see scripts/run_fmm_param_matrix.sh, e.g. "-p=-6 -pm=-1"). */
    if      ((v = optValue(arg, "S"))    != NULL) sys->maxSepRatio        = optDouble(arg, v, 1e-12, 10.0);
    else if ((v = optValue(arg, "o"))    != NULL) tolpar                  = optDouble(arg, v, 1e-300, 1.0);
    else if ((v = optValue(arg, "a"))    != NULL) arnoldiSz               = optInt(arg, v, 1, 100000);
    else if ((v = optValue(arg, "i"))    != NULL) numItr                  = optInt(arg, v, 1, 1000000);
    else if ((v = optValue(arg, "pm"))   != NULL) orderMom                = optInt(arg, v, -32, 32);
    else if ((v = optValue(arg, "p"))    != NULL) order                   = optInt(arg, v, -32, 32);
    /* tLegA/wLegA hold 11 slots filled only at 1..10 (src/numQuad.c): 0 makes
     * every quadrature loop run zero iterations and report a 0.0 energy as
     * success, and >10 reads past the table. */
    else if ((v = optValue(arg, "q"))    != NULL) sys->maxQuadOrder       = optInt(arg, v, 1, 10);
    else if ((v = optValue(arg, "t"))    != NULL) sys->depth              = optInt(arg, v, -1, 64);
    else if ((v = optValue(arg, "H"))    != NULL) sys->height             = optInt(arg, v, 0, 64);
    else if ((v = optValue(arg, "R"))    != NULL) meshResolution          = optDouble(arg, v, 1e-12, 1e6);
    /* epsilon must stay strictly positive: epsilon2/epsilon1 feeds the jump
     * terms in MtVmain, so a zero here yields inf and then a NaN solution.
     * -eps1/-eps2 is the spelling every script and doc uses; -e1/-e2 is what
     * -h used to advertise, so both are accepted. */
    else if ((v = optValue(arg, "eps1")) != NULL) epsilon1                = optDouble(arg, v, 1e-12, 1e6);
    else if ((v = optValue(arg, "eps2")) != NULL) epsilon2                = optDouble(arg, v, 1e-12, 1e6);
    else if ((v = optValue(arg, "e1"))   != NULL) epsilon1                = optDouble(arg, v, 1e-12, 1e6);
    else if ((v = optValue(arg, "e2"))   != NULL) epsilon2                = optDouble(arg, v, 1e-12, 1e6);
    else if ((v = optValue(arg, "k"))    != NULL) kappa                   = optDouble(arg, v, 0.0, 1e6);
    else if ((v = optValue(arg, "m"))    != NULL) sys->mesh_flag          = optInt(arg, v, 1, 2);
    else if ((v = optValue(arg, "M"))    != NULL) meshOnlyMode            = optInt(arg, v, 0, 1);
    else if ((v = optValue(arg, "B"))    != NULL) sys->benchmarkMode      = optInt(arg, v, 0, 1);
    else if ((v = optValue(arg, "g"))    != NULL) sys->gpuMode            = optInt(arg, v, -1, 1);
    else if ((v = optValue(arg, "c"))    != NULL) sys->debugCompareApply  = optInt(arg, v, 0, 1);
    else if ((v = optValue(arg, "C"))    != NULL) sys->debugComparePrecond= optInt(arg, v, 0, 1);
    else if ((v = optValue(arg, "r"))    != NULL) sys->matvecMode         = optInt(arg, v, 0, 2);
    else if ((v = optValue(arg, "Q"))    != NULL) sys->gpuQ2MMode         = optInt(arg, v, 0, 1);
    else if ((v = optValue(arg, "G"))    != NULL) sys->gpuNearfieldMode   = optInt(arg, v, 0, 1);
    else if ((v = optValue(arg, "P"))    != NULL) sys->precondCacheMode   = optInt(arg, v, -1, 3);
    else if ((v = optValue(arg, "d"))    != NULL) {
      meshOverrideValue = optDouble(arg, v, 1e-12, 1e9);
      meshOverrideSet = 1;
    }
    else {
      fprintf(stderr, "Error: unknown option '%s'\n", arg);
      fprintf(stderr, "Run '%s -h' for the list of supported options.\n", argv[0]);
      exit(1);
    }
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
      printf("GPU Q2M mode=%d (1=GPU, 0=CPU dgemv loop)\n", sys->gpuQ2MMode);
      printf("GPU nearfield mode=%d (0=interaction, 1=destination-leaf)\n", sys->gpuNearfieldMode);
    }
    printf("Preconditioner mode=%d (-1=disabled, 0=original, 1=cached-blocks, 2=cached-LU, 3=diagonal)\n", sys->precondCacheMode);
    printf("Energy mode=%s (charge-tree, panel-tree, compare)\n", energyMode());
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
  if (sys->matvecMode != 0) {
    /* Direct matvec modes still use treecode/FMM-side data for postprocessing. */
    buildPanelIndexDirect(sys);
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
  writeSolutionIfRequested(sys, sgm);

  if (getenv("FABIPB_STOP_AFTER_GMRES") != NULL) {
    treecode_t = 0.0;
    end_t = wall_seconds() - start_t;
    printf("FABIPB_STOP_AFTER_GMRES set: skipping post-GMRES treecode energy.\n");
    printf("ttl time: %f, gmres-its=%d\n", end_t, numItr);
    printf("solvation energy: skipped\n");
  } else {
    const char *postEnergyMode = energyMode();
    int treecodeTimeSet = 0;
    stage_t0 = wall_seconds();
    if (strcmp(postEnergyMode, "panel-tree") == 0) {
      ensureChargeTreeBuilt(sys);
      applyPanelChargeTreeEnergy(sys, sgm, &ptl);
    } else if (strcmp(postEnergyMode, "compare") == 0) {
      double chargeTreeRaw, panelTreeRaw;
      double chargeTreeTime, panelTreeTime;

      applyTreecode(sys, sgm, &chargeTreeRaw);
      chargeTreeTime = wall_seconds() - stage_t0;

      stage_t0 = wall_seconds();
      ensureChargeTreeBuilt(sys);
      applyPanelChargeTreeEnergy(sys, sgm, &panelTreeRaw);
      panelTreeTime = wall_seconds() - stage_t0;

      ptl = chargeTreeRaw;
      treecode_t = chargeTreeTime + panelTreeTime;
      treecodeTimeSet = 1;
      printf("Energy compare: charge-tree raw=%.17g panel-tree raw=%.17g abs-diff=%.17g rel-diff=%.17g charge-tree-time=%.6f panel-tree-time=%.6f\n",
             chargeTreeRaw, panelTreeRaw, fabs(chargeTreeRaw - panelTreeRaw),
             (fabs(chargeTreeRaw) > 0.0)
                 ? fabs(chargeTreeRaw - panelTreeRaw) / fabs(chargeTreeRaw)
                 : fabs(chargeTreeRaw - panelTreeRaw),
             chargeTreeTime, panelTreeTime);
    } else {
      applyTreecode( sys, sgm, &ptl );
    }
    if (!treecodeTimeSet) {
      treecode_t = wall_seconds() - stage_t0;
    }
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
  int i, nPnls = sys->nPnls;
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
  /* Indexed over the contiguous area array rather than walking the panel list;
   * see buildPanelIndex() in fmm.c. */
  {
    const double *area = sys->panelArea;
    for ( i = 0; i < nPnls; i++ ) {
      pot[i] = (scale1*area[i]*sgm[i]-pot[i]);
      pot[i+nPnls] = scale2*area[i]*sgm[i+nPnls]-pot[i+nPnls];
    }
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
