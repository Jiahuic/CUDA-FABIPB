#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "gmres.h"
#include "gkGlobal.h"

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

static void gmres_vector_preview(const char *tag, int iter, const double *x, int n)
{
    int i;
    int limit = (n < 6) ? n : 6;
    double inf = 0.0;

    for (i = 0; i < n; ++i) {
        double ax = fabs(x[i]);
        if (ax > inf) {
            inf = ax;
        }
    }

    printf("GMRES vector %s: iter=%d inf=%e", tag, iter, inf);
    for (i = 0; i < limit; ++i) {
        printf(" x[%d]=%e", i, x[i]);
    }
    printf("\n");
}

static const char *gmres_dump_prefix(void)
{
    const char *env = getenv("FABIPB_GMRES_DUMP_PREFIX");
    return (env != NULL && env[0] != '\0') ? env : NULL;
}

static int gmres_dump_stride(void)
{
    const char *env = getenv("FABIPB_GMRES_DUMP_STRIDE");
    int stride;

    if (env == NULL || env[0] == '\0') {
        return 1;
    }
    stride = atoi(env);
    return (stride > 0) ? stride : 1;
}

static char *gmres_dump_path(const char *prefix, const char *tag, const char *suffix)
{
    size_t len = strlen(prefix) + 1 + strlen(tag) + strlen(suffix) + 1;
    char *path = (char *)malloc(len);
    if (path == NULL) {
        return NULL;
    }
    snprintf(path, len, "%s_%s%s", prefix, tag, suffix);
    return path;
}

static void gmres_dump_metadata(const char *prefix, const char *tag, int iter,
                                int n, int stride, double l1, double l2,
                                double inf, double sum,
                                double comp0Sum, double comp0L1,
                                double comp0L2, double comp0Inf,
                                double comp1Sum, double comp1L1,
                                double comp1L2, double comp1Inf)
{
    char *path = gmres_dump_path(prefix, "metadata", ".txt");
    FILE *fp;
    int half = n / 2;
    int comp1N = n - half;

    if (path == NULL) {
        return;
    }
    fp = fopen(path, "a");
    if (fp == NULL) {
        fprintf(stderr, "Warning: cannot write GMRES metadata dump '%s'\n", path);
        free(path);
        return;
    }
    fprintf(fp,
            "tag=%s iter=%d n=%d stride=%d sum=%.17g l1=%.17g l2=%.17g inf=%.17g mean=%.17g "
            "component0_n=%d component0_sum=%.17g component0_l1=%.17g component0_l2=%.17g component0_inf=%.17g component0_mean=%.17g "
            "component1_n=%d component1_sum=%.17g component1_l1=%.17g component1_l2=%.17g component1_inf=%.17g component1_mean=%.17g\n",
            tag, iter, n, stride, sum, l1, l2, inf, (n > 0) ? sum / (double)n : 0.0,
            half, comp0Sum, comp0L1, comp0L2, comp0Inf,
            (half > 0) ? comp0Sum / (double)half : 0.0,
            comp1N, comp1Sum, comp1L1, comp1L2, comp1Inf,
            (comp1N > 0) ? comp1Sum / (double)comp1N : 0.0);
    fclose(fp);
    free(path);
}

static void gmres_dump_scalar(const char *prefix, const char *tag, int iter,
                              double value)
{
    char *path = gmres_dump_path(prefix, "metadata", ".txt");
    FILE *fp;

    if (path == NULL) {
        return;
    }
    fp = fopen(path, "a");
    if (fp == NULL) {
        fprintf(stderr, "Warning: cannot write GMRES scalar dump '%s'\n", path);
        free(path);
        return;
    }
    fprintf(fp, "tag=%s iter=%d value=%.17g\n", tag, iter, value);
    fclose(fp);
    free(path);
}

static void gmres_dump_vector(const char *prefix, const char *tag, int iter,
                              const double *x, int n)
{
    int stride = gmres_dump_stride();
    int half = n / 2;
    int i;
    double l1 = 0.0;
    double l2 = 0.0;
    double inf = 0.0;
    double sum = 0.0;
    double compSum[2] = {0.0, 0.0};
    double compL1[2] = {0.0, 0.0};
    double compL2[2] = {0.0, 0.0};
    double compInf[2] = {0.0, 0.0};
    char *path;
    FILE *fp;

    if (prefix == NULL) {
        return;
    }
    path = gmres_dump_path(prefix, tag, ".csv");
    if (path == NULL) {
        return;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "Warning: cannot write GMRES vector dump '%s'\n", path);
        free(path);
        return;
    }
    fprintf(fp, "idx,component,value\n");
    for (i = 0; i < n; ++i) {
        double value = x[i];
        double av = fabs(value);
        int comp = (i < half) ? 0 : 1;
        l1 += av;
        l2 += value * value;
        sum += value;
        if (av > inf) {
            inf = av;
        }
        compSum[comp] += value;
        compL1[comp] += av;
        compL2[comp] += value * value;
        if (av > compInf[comp]) {
            compInf[comp] = av;
        }
        if ((i % stride) == 0) {
            fprintf(fp, "%d,%d,%.17g\n", i, comp, value);
        }
    }
    fclose(fp);
    gmres_dump_metadata(prefix, tag, iter, n, stride, l1, sqrt(l2), inf,
                        sum,
                        compSum[0], compL1[0], sqrt(compL2[0]), compInf[0],
                        compSum[1], compL1[1], sqrt(compL2[1]), compInf[1]);
    printf("GMRES vector dump: tag=%s iter=%d path=%s stride=%d\n",
           tag, iter, path, stride);
    free(path);
}

static void gmres_dump_first_update(const char *prefix, int iter, int n,
                                    const double *x, double *h, int ldh,
                                    double *y, const double *s,
                                    double *v, int ldv)
{
    double *x_after;
    int inc = 1;

    if (prefix == NULL || iter != 1) {
        return;
    }
    x_after = (double *)malloc((size_t)n * sizeof(double));
    if (x_after == NULL) {
        fprintf(stderr, "Warning: cannot allocate GMRES first-iteration dump vector\n");
        return;
    }
    dcopy_(&n, x, &inc, x_after, &inc);
    gmres_update(iter, n, x_after, h, ldh, y, s, v, ldv);
    gmres_dump_vector(prefix, "x_after_iter1", iter, x_after, n);
    free(x_after);
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
    int logVectors = 0;
    const char *logResidEnv = getenv("FABIPB_GMRES_LOG_RESID");
    const char *stopAfterIterEnv = getenv("FABIPB_GMRES_STOP_AFTER_ITER");
    const char *logVectorsEnv = getenv("FABIPB_GMRES_LOG_VECTORS");
    const char *dumpPrefix = gmres_dump_prefix();

    if (logResidEnv != NULL && atoi(logResidEnv) > 0) {
        logResid = 1;
    }
    if (stopAfterIterEnv != NULL && atoi(stopAfterIterEnv) > 0) {
        stopAfterIter = atoi(stopAfterIterEnv);
    }
    if (logVectorsEnv != NULL && atoi(logVectorsEnv) > 0) {
        logVectors = 1;
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

    if (dumpPrefix != NULL) {
        gmres_dump_vector(dumpPrefix, "b", 0, b, n);
        gmres_dump_vector(dumpPrefix, "x0", 0, x, n);
    }

    dcopy_(&n, b, &inc, gmres_work_col(work, av_col, ldw), &inc);
    if (dnrm2_(&n, x, &inc) != 0.0) {
        dcopy_(&n, b, &inc, gmres_work_col(work, av_col, ldw), &inc);
        aa = wall_seconds();
        matvec((double *)&neg_one, x, (double *)&one,
               gmres_work_col(work, av_col, ldw));
        gmresMatvecTime += wall_seconds() - aa;
        gmresMatvecCalls++;
    }
    if (dumpPrefix != NULL) {
        gmres_dump_vector(dumpPrefix, "r0", 0, gmres_work_col(work, av_col, ldw), n);
    }

    if (logVectors) {
        gmres_vector_preview("b", 0, gmres_work_col(work, av_col, ldw), n);
    }
    aa = wall_seconds();
    psolve(gmres_work_col(work, r_col, ldw), gmres_work_col(work, av_col, ldw));
    gmresPsolveTime += wall_seconds() - aa;
    gmresPsolveCalls++;
    if (logVectors) {
        gmres_vector_preview("Mb", 0, gmres_work_col(work, r_col, ldw), n);
    }
    if (dumpPrefix != NULL) {
        gmres_dump_vector(dumpPrefix, "Minv_r0", 0, gmres_work_col(work, r_col, ldw), n);
    }
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
        aa = 1.0 / rnorm;
        dscal_(&n, &aa, gmres_work_col(work, v_col, ldw), &inc);
        if (logVectors) {
            gmres_vector_preview("v", 0, gmres_work_col(work, v_col, ldw), n);
        }
        if (dumpPrefix != NULL && *iter == 0) {
            gmres_dump_vector(dumpPrefix, "v1", 0, gmres_work_col(work, v_col, ldw), n);
        }

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
            if (logVectors && *iter == 1) {
                gmres_vector_preview("Av", *iter, gmres_work_col(work, av_col, ldw), n);
            }
            if (dumpPrefix != NULL && *iter == 1) {
                gmres_dump_vector(dumpPrefix, "Av1", *iter, gmres_work_col(work, av_col, ldw), n);
            }
            aa = wall_seconds();
            psolve(gmres_work_col(work, w_col, ldw), gmres_work_col(work, av_col, ldw));
            gmresPsolveTime += wall_seconds() - aa;
            gmresPsolveCalls++;
            if (logVectors && *iter == 1) {
                gmres_vector_preview("MAv", *iter, gmres_work_col(work, w_col, ldw), n);
            }
            if (dumpPrefix != NULL && *iter == 1) {
                gmres_dump_vector(dumpPrefix, "Minv_Av1", *iter, gmres_work_col(work, w_col, ldw), n);
            }

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
            if (dumpPrefix != NULL && *iter == 1) {
                gmres_dump_scalar(dumpPrefix, "first_residual", *iter, *resid);
                gmres_dump_first_update(dumpPrefix, i, n, x, h, ldh,
                                        gmres_work_col(work, y_col, ldw),
                                        gmres_work_col(work, s_col, ldw),
                                        gmres_work_col(work, v_col, ldw), ldw);
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

static void gmres_basis(int iter, int n, double *h_col, double *v, int ldv, double *w)
{
    const int inc = 1;
    int k;
    double scale;

    for (k = 0; k < iter; ++k) {
        h_col[k] = ddot_(&n, w, &inc, &v[(size_t)k * (size_t)ldv], &inc);
        scale = -h_col[k];
        daxpy_(&n, &scale, &v[(size_t)k * (size_t)ldv], &inc, w, &inc);
    }

    h_col[iter] = dnrm2_(&n, w, &inc);
    dcopy_(&n, w, &inc, &v[(size_t)iter * (size_t)ldv], &inc);
    scale = 1.0 / h_col[iter];
    dscal_(&n, &scale, &v[(size_t)iter * (size_t)ldv], &inc);
}
