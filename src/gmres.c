#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>

#include "gmres.h"
#include "gkGlobal.h"
#include "fabipb_system.h"

/* BLAS prototypes */
extern int dcopy_(const int *n, const double *dx, const int *incx, double *dy, const int *incy);
extern double dnrm2_(const int *n, const double *x, const int *incx);
extern int dscal_(const int *n, const double *da, double *dx, const int *incx);
extern int drot_(const int *n, double *dx, const int *incx, double *dy, const int *incy, const double *c, const double *s);
extern int drotg_(double *da, double *db, double *c, double *s);
extern double ddot_(const int *n, const double *dx, const int *incx, const double *dy, const int *incy);
extern int daxpy_(const int *n, const double *da, const double *dx, const int *incx, double *dy, const int *incy);
extern int dtrsv_(const char *uplo, const char *trans, const char *diag, const int *n,
                  const double *a, const int *lda, double *x, const int *incx);
extern int dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
                  const double *a, const int *lda, const double *x, const int *incx,
                  const double *beta, double *y, const int *incy);

static void gmres_update(int iter, int n, double *x, double *h, int ldh,
                         double *y, const double *s, double *v, int ldv);
static void gmres_basis(int iter, int n, double *h_col, double *v, int ldv, double *w);

static double *gmres_work_col(double *work, int col, int ldw)
{
    return work + (size_t)col * (size_t)ldw;
}

static double wall_seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}


/*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the
*     Solution of Linear Systems: Building Blocks for Iterative
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
*     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
*  Purpose
*  =======
*
*  GMRES solves the linear system Ax = b using the
*  Generalized Minimal Residual iterative method with preconditioning.
*
*  Convergence test: ( norm( b - A*x ) / norm( b ) ) < TOL.
*  For other measures, see the above reference.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER.
*          On entry, the dimension of the matrix.
*          Unchanged on exit.
*
*  B       (input) DOUBLE PRECISION array, dimension N.
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  X       (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess; on exit, the iterated solution.
*
*  RESTRT  (input) INTEGER
*          Restart parameter, <= N. This parameter controls the amount
*          of memory required for matrix H (see WORK and H).
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,RESTRT+4).
*
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW >= max(1,N).
*
*  H       (workspace) DOUBLE PRECISION array, dimension (LDH,RESTRT+2).
*          This workspace is used for constructing and storing the
*          upper Hessenberg matrix. The two extra columns are used to
*          store the Givens rotation matrices.
*
*  LDH    (input) INTEGER
*          The leading dimension of the array H. LDH >= max(1,RESTRT+1).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (input/output) DOUBLE PRECISION
*          On input, the allowable convergence measure for
*          norm( b - A*x ) / norm( b ).
*          On output, the final value of this measure.
*
*  MATVEC  (external subroutine)
*          The user must provide a subroutine to perform the
*          matrix-vector product
*
*               y := alpha*A*x + beta*y,
*
*          where alpha and beta are scalars, x and y are vectors,
*          and A is a matrix. Vector x must remain unchanged.
*          The solution is over-written on vector y.
*
*          The call is:
*
*             CALL MATVEC( ALPHA, X, BETA, Y )
*
*          The matrix is passed into the routine in a common block.
*
*  PSOLVE  (external subroutine)
*          The user must provide a subroutine to perform the
*          preconditioner solve routine for the linear system
*
*               M*x = b,
*
*          where x and b are vectors, and M a matrix. Vector b must
*          remain unchanged.
*          The solution is over-written on vector x.
*
*          The call is:
*
*             CALL PSOLVE( X, B )
*
*          The preconditioner is passed into the routine in a common
*          block.
*
*  INFO    (output) INTEGER
*
*          =  0: Successful exit. Iterated approximate solution returned.
*
*          >  0: Convergence to tolerance not achieved. This will be
*                set to the number of iterations performed.
*
*          <  0: Illegal input parameter.
*
*                   -1: matrix dimension N < 0
*                   -2: LDW < N
*                   -3: Maximum number of iterations ITER <= 0.
*                   -4: LDH < RESTRT
*
*  BLAS CALLS:   DAXPY, DCOPY, DDOT, DNRM2, DROT, DROTG, DSCAL
*  ============================================================
*/

int gmres(int n, double *b, double *x, int restrt, double *work, int ldw,
          double *h, int ldh, int *iter, double *resid,
          GmresMatVecFn matvec, GmresPrecondFn psolve, int *info)
{
    const int inc = 1;
    const double neg_one = -1.0;
    const double one = 1.0;
    const double zero = 0.0;
    int i;
    int k;
    int maxit;
    int r_col;
    int s_col;
    int w_col;
    int y_col;
    int av_col;
    int v_col;
    int cs_col;
    int sn_col;
    double aa;
    double bb;
    double bnrm2;
    double rnorm;
    double tol;
    int logResid = 0;
    int stopAfterIter = 0;
    const char *logResidEnv = getenv("FABIPB_GMRES_LOG_RESID");
    const char *stopAfterIterEnv = getenv("FABIPB_GMRES_STOP_AFTER_ITER");

    if (logResidEnv != NULL && atoi(logResidEnv) > 0) {
        logResid = 1;
    }
    if (stopAfterIterEnv != NULL && atoi(stopAfterIterEnv) > 0) {
        stopAfterIter = atoi(stopAfterIterEnv);
    }

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (ldw < n) {
        *info = -2;
    } else if (*iter <= 0) {
        *info = -3;
    } else if (ldh < restrt + 1) {
        *info = -4;
    }
    if (*info != 0) {
        return 0;
    }

    maxit = *iter;
    tol = *resid;

    r_col = 0;
    s_col = 1;
    w_col = 2;
    y_col = 2;
    av_col = 2;
    v_col = 3;
    cs_col = restrt;
    sn_col = restrt + 1;

    dcopy_(&n, b, &inc, gmres_work_col(work, av_col, ldw), &inc);
    if (dnrm2_(&n, x, &inc) != 0.0) {
        dcopy_(&n, b, &inc, gmres_work_col(work, av_col, ldw), &inc);
        aa = wall_seconds();
        matvec((double *)&neg_one, x, (double *)&one,
               gmres_work_col(work, av_col, ldw));
        gmresMatvecTime += wall_seconds() - aa;
        gmresMatvecCalls++;
    }
    aa = wall_seconds();
    psolve(gmres_work_col(work, r_col, ldw), gmres_work_col(work, av_col, ldw));
    gmresPsolveTime += wall_seconds() - aa;
    gmresPsolveCalls++;
    bnrm2 = dnrm2_(&n, b, &inc);
    if (bnrm2 == 0.0) {
        bnrm2 = 1.0;
    }
    if (dnrm2_(&n, gmres_work_col(work, r_col, ldw), &inc) / bnrm2 < tol) {
        return 0;
    }

    *iter = 0;

    for (;;) {
        i = 0;

        dcopy_(&n, gmres_work_col(work, r_col, ldw), &inc,
               gmres_work_col(work, v_col, ldw), &inc);
        rnorm = dnrm2_(&n, gmres_work_col(work, v_col, ldw), &inc);
        /* A zero (or non-finite) restart residual would make 1/rnorm inf or
         * NaN, fill the Krylov basis with NaN, and leave *resid NaN so that
         * "*resid <= tol" never holds -- the solver would then burn every
         * remaining iteration and return a NaN solution reported as a normal
         * iteration-limit exit. Stop cleanly instead. */
        if (!(rnorm > 0.0) || rnorm != rnorm) {
            *resid = (rnorm != rnorm) ? rnorm : 0.0;
            *info = (rnorm != rnorm) ? 2 : 0;
            return 0;
        }
        aa = 1.0 / rnorm;
        dscal_(&n, &aa, gmres_work_col(work, v_col, ldw), &inc);
        gmres_work_col(work, s_col, ldw)[0] = rnorm;
        for (k = 1; k < n; ++k) {
            gmres_work_col(work, s_col, ldw)[k] = 0.0;
        }

        for (;;) {
            ++i;
            ++(*iter);

            aa = wall_seconds();
            matvec((double *)&one, gmres_work_col(work, v_col + i - 1, ldw),
                   (double *)&zero, gmres_work_col(work, av_col, ldw));
            gmresMatvecTime += wall_seconds() - aa;
            gmresMatvecCalls++;
            aa = wall_seconds();
            psolve(gmres_work_col(work, w_col, ldw), gmres_work_col(work, av_col, ldw));
            gmresPsolveTime += wall_seconds() - aa;
            gmresPsolveCalls++;

            aa = wall_seconds();
            gmres_basis(i, n, &h[(i - 1) * ldh], gmres_work_col(work, v_col, ldw), ldw,
                        gmres_work_col(work, w_col, ldw));
            gmresBasisTime += wall_seconds() - aa;

            for (k = 0; k < i - 1; ++k) {
                drot_(&inc, &h[(i - 1) * ldh + k], &inc, &h[(i - 1) * ldh + k + 1],
                      &inc, &h[cs_col * ldh + k], &h[sn_col * ldh + k]);
            }

            aa = h[(i - 1) * ldh + (i - 1)];
            bb = h[(i - 1) * ldh + i];
            drotg_(&aa, &bb, &h[cs_col * ldh + (i - 1)], &h[sn_col * ldh + (i - 1)]);
            drot_(&inc, &h[(i - 1) * ldh + (i - 1)], &inc, &h[(i - 1) * ldh + i],
                  &inc, &h[cs_col * ldh + (i - 1)], &h[sn_col * ldh + (i - 1)]);

            drot_(&inc, &gmres_work_col(work, s_col, ldw)[i - 1], &inc,
                  &gmres_work_col(work, s_col, ldw)[i],
                  &inc, &h[cs_col * ldh + (i - 1)], &h[sn_col * ldh + (i - 1)]);
            aa = wall_seconds();
            *resid = fabs(gmres_work_col(work, s_col, ldw)[i]) / bnrm2;
            gmresResidualTime += wall_seconds() - aa;
            if (logResid) {
                printf("GMRES residual: iter=%d resid=%e\n", *iter, *resid);
            }
            if (stopAfterIter > 0 && *iter >= stopAfterIter) {
                fprintf(stderr, "GMRES debug stop after iter=%d\n", *iter);
                aa = wall_seconds();
                gmres_update(i, n, x, h, ldh, gmres_work_col(work, y_col, ldw),
                             gmres_work_col(work, s_col, ldw),
                             gmres_work_col(work, v_col, ldw), ldw);
                gmresUpdateTime += wall_seconds() - aa;
                *info = 77;
                return 0;
            }

            if (*resid <= tol) {
                aa = wall_seconds();
                gmres_update(i, n, x, h, ldh, gmres_work_col(work, y_col, ldw),
                             gmres_work_col(work, s_col, ldw),
                             gmres_work_col(work, v_col, ldw), ldw);
                gmresUpdateTime += wall_seconds() - aa;
                return 0;
            }
            if (*iter == maxit || i >= restrt) {
                break;
            }
        }

        aa = wall_seconds();
        /* maxit can stop a partial restart cycle, so only i columns are valid. */
        gmres_update(i, n, x, h, ldh, gmres_work_col(work, y_col, ldw),
                     gmres_work_col(work, s_col, ldw),
                     gmres_work_col(work, v_col, ldw), ldw);
        gmresUpdateTime += wall_seconds() - aa;

        dcopy_(&n, b, &inc, gmres_work_col(work, av_col, ldw), &inc);
        aa = wall_seconds();
        matvec((double *)&neg_one, x, (double *)&one,
               gmres_work_col(work, av_col, ldw));
        gmresMatvecTime += wall_seconds() - aa;
        gmresMatvecCalls++;
        aa = wall_seconds();
        psolve(gmres_work_col(work, r_col, ldw), gmres_work_col(work, av_col, ldw));
        gmresPsolveTime += wall_seconds() - aa;
        gmresPsolveCalls++;
        aa = wall_seconds();
        gmres_work_col(work, s_col, ldw)[i] =
            dnrm2_(&n, gmres_work_col(work, r_col, ldw), &inc);
        *resid = gmres_work_col(work, s_col, ldw)[i] / bnrm2;
        gmresResidualTime += wall_seconds() - aa;
        if (logResid) {
            printf("GMRES residual: iter=%d resid=%e\n", *iter, *resid);
        }
        if (stopAfterIter > 0 && *iter >= stopAfterIter) {
            fprintf(stderr, "GMRES debug stop after iter=%d\n", *iter);
            *info = 77;
            return 0;
        }
        if (*resid <= tol) {
            return 0;
        }
        if (*iter == maxit) {
            *info = 1;
            return 0;
        }
    }
}

static void gmres_update(int iter, int n, double *x, double *h, int ldh,
                         double *y, const double *s, double *v, int ldv)
{
    const int inc = 1;
    const double one = 1.0;

    dcopy_(&iter, s, &inc, y, &inc);
    dtrsv_("U", "N", "N", &iter, h, &ldh, y, &inc);
    dgemv_("N", &n, &iter, &one, v, &ldv, y, &inc, &one, x, &inc);
}

/*
 * Threaded modified Gram-Schmidt.
 *
 * The orthogonalisation is the largest CPU-side stage of a large solve --
 * 58.97 s of the 700.83 s sdens=1 capsid run, 16% of GMRES -- and it was
 * running on one core. Not because of a missing pragma: the ddot/daxpy pair
 * goes through OpenBLAS, and every benchmark and production run starts under
 * scripts/with_benchmark_env.sh, which pins OPENBLAS_NUM_THREADS=1. The runner
 * overrides FABIPB's own worker pools but never that one. These are pure
 * bandwidth-bound level-1 operations on 163 MB vectors, so one core leaves
 * most of the socket's bandwidth unused.
 *
 * Threading a dot product changes the summation order, so it has to be done in
 * a way that does not make results depend on the thread count. The vector is
 * cut into fixed-size chunks independent of how many threads exist; each chunk
 * is reduced by OpenBLAS exactly as before, and the per-chunk results are then
 * summed in ascending chunk order on one thread. The answer therefore depends
 * only on BASIS_CHUNK, not on FABIPB_GMRES_BASIS_THREADS -- the same guarantee
 * the disjoint-output pools elsewhere in the code give. daxpy is elementwise
 * and is bit-identical at any thread count.
 *
 * Below BASIS_MIN_N the original single-call path runs, so small cases stay
 * bit-identical to previous releases; only runs big enough to care change, and
 * there the shift is a reassociated sum well inside the 1e-4 solve tolerance.
 *
 * dnrm2/dcopy/dscal are left alone: one call each against `iter` dot/axpy
 * pairs, so they are a few percent of the traffic, and keeping dnrm2_ avoids
 * changing its overflow-scaling semantics for no measurable gain.
 */
#define BASIS_CHUNK  (1 << 16)
#define BASIS_MIN_N  (1 << 20)

typedef struct {
    int nChunks;
    int n;
    const double *vk;
    double *w;
    double *partial;      /* [nChunks]; chunk c reduced independently */
    double alpha;         /* daxpy scale; unused in the dot phase */
    int doAxpy;           /* 0 => dot phase, 1 => axpy phase */
    int next;             /* shared chunk cursor */
    int phase;            /* incremented by main to release one phase */
    int done;             /* workers finished with the current phase */
    int stop;             /* set by main to retire the pool */
    int nWorkers;         /* pool size, excluding the main thread */
    pthread_mutex_t lock;
    pthread_cond_t cv;
} BasisJob;

/* One chunked phase. Every participant, main thread included, runs this. */
static void gmresBasisPhase(BasisJob *job)
{
    const int inc = 1;

    for (;;) {
        int c, off, len;

        pthread_mutex_lock(&job->lock);
        c = job->next++;
        pthread_mutex_unlock(&job->lock);
        if (c >= job->nChunks) break;

        off = c * BASIS_CHUNK;
        len = job->n - off;
        if (len > BASIS_CHUNK) len = BASIS_CHUNK;

        if (job->doAxpy) {
            daxpy_(&len, &job->alpha, job->vk + off, &inc, job->w + off, &inc);
        } else {
            job->partial[c] = ddot_(&len, job->w + off, &inc, job->vk + off, &inc);
        }
    }
}

/*
 * Workers live for the whole k-loop rather than being respawned per phase.
 * Spawning per phase would cost 2*iter*(nThreads-1) pthread_create calls per
 * gmres_basis -- roughly 380k over an sdens=1 solve, several seconds against a
 * 59 s target. Here it is one create per thread per call.
 *
 * Released by a phase counter rather than a barrier. A barrier has to be sized
 * up front, which makes a partial pthread_create failure awkward to handle: the
 * already-started workers may be parked on it, so it can neither be resized nor
 * destroyed safely. A counter is correct for whatever pool actually starts,
 * including none at all.
 *
 * The phases cannot overlap -- the axpy scale is the dot result -- so main
 * publishes parameters, joins the work itself, then waits for every worker to
 * report done before touching the job again.
 */
static void *gmresBasisWorker(void *arg)
{
    BasisJob *job = (BasisJob *)arg;
    int seen = 0;

    for (;;) {
        pthread_mutex_lock(&job->lock);
        while (job->phase == seen && !job->stop) {
            pthread_cond_wait(&job->cv, &job->lock);
        }
        if (job->stop) { pthread_mutex_unlock(&job->lock); break; }
        seen = job->phase;
        pthread_mutex_unlock(&job->lock);

        gmresBasisPhase(job);

        pthread_mutex_lock(&job->lock);
        job->done++;
        pthread_cond_broadcast(&job->cv);
        pthread_mutex_unlock(&job->lock);
    }
    return NULL;
}

/* Publish one phase, work alongside the pool, and wait for it to drain. */
static void gmresBasisRunPhase(BasisJob *job)
{
    pthread_mutex_lock(&job->lock);
    job->next = 0;
    job->done = 0;
    job->phase++;
    pthread_cond_broadcast(&job->cv);
    pthread_mutex_unlock(&job->lock);

    gmresBasisPhase(job);

    pthread_mutex_lock(&job->lock);
    while (job->done < job->nWorkers) {
        pthread_cond_wait(&job->cv, &job->lock);
    }
    pthread_mutex_unlock(&job->lock);
}

static int gmresBasisThreadCount(void)
{
    static int cached = 0;
    const char *env;
    int threads;

    if (cached != 0) return cached;
    env = getenv("FABIPB_GMRES_BASIS_THREADS");
    if (env != NULL && env[0] != '\0') {
        threads = atoi(env);
    } else {
        threads = fabipb_online_cpu_count();
    }
    if (threads < 1) threads = 1;
    if (threads > 128) threads = 128;
    cached = threads;
    return cached;
}

static void gmres_basis(int iter, int n, double *h_col, double *v, int ldv, double *w)
{
    const int inc = 1;
    int k;
    double scale;
    int nThreads = (n >= BASIS_MIN_N) ? gmresBasisThreadCount() : 1;

    if (nThreads > 1 && iter > 0) {
        int nChunks = (n + BASIS_CHUNK - 1) / BASIS_CHUNK;
        double *partial = (double *)malloc((size_t)nChunks * sizeof(double));
        pthread_t *tids = (pthread_t *)malloc((size_t)nThreads * sizeof(pthread_t));
        int created = 0;

        if (partial != NULL && tids != NULL) {
            BasisJob job;
            int t;

            job.nChunks = nChunks;
            job.n = n;
            job.w = w;
            job.partial = partial;
            job.next = 0;
            job.phase = 0;
            job.done = 0;
            job.stop = 0;
            job.nWorkers = 0;
            pthread_mutex_init(&job.lock, NULL);
            pthread_cond_init(&job.cv, NULL);

            for (t = 0; t < nThreads - 1; ++t) {
                if (pthread_create(&tids[created], NULL, gmresBasisWorker, &job) != 0) break;
                ++created;
            }
            /* Under the lock: workers are already running, and this is the
             * count gmresBasisRunPhase waits on. */
            pthread_mutex_lock(&job.lock);
            job.nWorkers = created;   /* whatever actually started */
            pthread_mutex_unlock(&job.lock);

            for (k = 0; k < iter; ++k) {
                double sum = 0.0;
                int c;

                job.vk = &v[(size_t)k * (size_t)ldv];
                job.doAxpy = 0;
                gmresBasisRunPhase(&job);

                /* Ascending chunk order: the result must not depend on which
                 * thread happened to reduce which chunk. */
                for (c = 0; c < nChunks; ++c) sum += partial[c];
                h_col[k] = sum;

                job.alpha = -sum;
                job.doAxpy = 1;
                gmresBasisRunPhase(&job);
            }

            pthread_mutex_lock(&job.lock);
            job.stop = 1;
            pthread_cond_broadcast(&job.cv);
            pthread_mutex_unlock(&job.lock);
            for (t = 0; t < created; ++t) pthread_join(tids[t], NULL);
            pthread_cond_destroy(&job.cv);
            pthread_mutex_destroy(&job.lock);
            free(tids);
            free(partial);
            h_col[iter] = dnrm2_(&n, w, &inc);
            goto finish;
        }
        free(tids);
        free(partial);
        /* Allocation failed: fall through to the serial path. */
    }

    for (k = 0; k < iter; ++k) {
        h_col[k] = ddot_(&n, w, &inc, &v[(size_t)k * (size_t)ldv], &inc);
        scale = -h_col[k];
        daxpy_(&n, &scale, &v[(size_t)k * (size_t)ldv], &inc, w, &inc);
    }

    h_col[iter] = dnrm2_(&n, w, &inc);

finish:
    dcopy_(&n, w, &inc, &v[(size_t)iter * (size_t)ldv], &inc);
    /* Happy breakdown: h[i+1,i] == 0 means the Krylov space is exhausted and
     * the current subspace already contains the exact solution. Scaling by
     * 1/0 here would fill the basis vector with NaN and poison every later
     * iteration. Leaving it zero is safe: the Givens rotation below sees
     * bb == 0, produces c=1/s=0, drives the residual to 0, and the caller's
     * "*resid <= tol" test exits with the exact solution. */
    if (h_col[iter] > 0.0) {
        scale = 1.0 / h_col[iter];
        dscal_(&n, &scale, &v[(size_t)iter * (size_t)ldv], &inc);
    }
}
