#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

    dcopy_(&n, b, &inc, &work[av_col * ldw], &inc);
    if (dnrm2_(&n, x, &inc) != 0.0) {
        dcopy_(&n, b, &inc, &work[av_col * ldw], &inc);
        aa = wall_seconds();
        matvec((double *)&neg_one, x, (double *)&one, &work[av_col * ldw]);
        gmresMatvecTime += wall_seconds() - aa;
        gmresMatvecCalls++;
    }

    if (logVectors) {
        gmres_vector_preview("b", 0, &work[av_col * ldw], n);
    }
    aa = wall_seconds();
    psolve(&work[r_col * ldw], &work[av_col * ldw]);
    gmresPsolveTime += wall_seconds() - aa;
    gmresPsolveCalls++;
    if (logVectors) {
        gmres_vector_preview("Mb", 0, &work[r_col * ldw], n);
    }
    bnrm2 = dnrm2_(&n, b, &inc);
    if (bnrm2 == 0.0) {
        bnrm2 = 1.0;
    }
    if (dnrm2_(&n, &work[r_col * ldw], &inc) / bnrm2 < tol) {
        return 0;
    }

    *iter = 0;

    for (;;) {
        i = 0;

        dcopy_(&n, &work[r_col * ldw], &inc, &work[v_col * ldw], &inc);
        rnorm = dnrm2_(&n, &work[v_col * ldw], &inc);
        aa = 1.0 / rnorm;
        dscal_(&n, &aa, &work[v_col * ldw], &inc);
        if (logVectors) {
            gmres_vector_preview("v", 0, &work[v_col * ldw], n);
        }

        work[s_col * ldw] = rnorm;
        for (k = 1; k < n; ++k) {
            work[s_col * ldw + k] = 0.0;
        }

        for (;;) {
            ++i;
            ++(*iter);

            aa = wall_seconds();
            matvec((double *)&one, &work[(v_col + i - 1) * ldw], (double *)&zero,
                   &work[av_col * ldw]);
            gmresMatvecTime += wall_seconds() - aa;
            gmresMatvecCalls++;
            if (logVectors && *iter == 1) {
                gmres_vector_preview("Av", *iter, &work[av_col * ldw], n);
            }
            aa = wall_seconds();
            psolve(&work[w_col * ldw], &work[av_col * ldw]);
            gmresPsolveTime += wall_seconds() - aa;
            gmresPsolveCalls++;
            if (logVectors && *iter == 1) {
                gmres_vector_preview("MAv", *iter, &work[w_col * ldw], n);
            }

            aa = wall_seconds();
            gmres_basis(i, n, &h[(i - 1) * ldh], &work[v_col * ldw], ldw,
                        &work[w_col * ldw]);
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

            drot_(&inc, &work[s_col * ldw + (i - 1)], &inc, &work[s_col * ldw + i],
                  &inc, &h[cs_col * ldh + (i - 1)], &h[sn_col * ldh + (i - 1)]);
            aa = wall_seconds();
            *resid = fabs(work[s_col * ldw + i]) / bnrm2;
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
                aa = wall_seconds();
                gmres_update(i, n, x, h, ldh, &work[y_col * ldw], &work[s_col * ldw],
                             &work[v_col * ldw], ldw);
                gmresUpdateTime += wall_seconds() - aa;
                return 0;
            }
            if (*iter == maxit || i >= restrt) {
                break;
            }
        }

        aa = wall_seconds();
        /* maxit can stop a partial restart cycle, so only i columns are valid. */
        gmres_update(i, n, x, h, ldh, &work[y_col * ldw], &work[s_col * ldw],
                     &work[v_col * ldw], ldw);
        gmresUpdateTime += wall_seconds() - aa;

        dcopy_(&n, b, &inc, &work[av_col * ldw], &inc);
        aa = wall_seconds();
        matvec((double *)&neg_one, x, (double *)&one, &work[av_col * ldw]);
        gmresMatvecTime += wall_seconds() - aa;
        gmresMatvecCalls++;
        aa = wall_seconds();
        psolve(&work[r_col * ldw], &work[av_col * ldw]);
        gmresPsolveTime += wall_seconds() - aa;
        gmresPsolveCalls++;
        aa = wall_seconds();
        work[s_col * ldw + i] = dnrm2_(&n, &work[r_col * ldw], &inc);
        *resid = work[s_col * ldw + i] / bnrm2;
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
        h_col[k] = ddot_(&n, w, &inc, &v[k * ldv], &inc);
        scale = -h_col[k];
        daxpy_(&n, &scale, &v[k * ldv], &inc, w, &inc);
    }

    h_col[iter] = dnrm2_(&n, w, &inc);
    dcopy_(&n, w, &inc, &v[iter * ldv], &inc);
    scale = 1.0 / h_col[iter];
    dscal_(&n, &scale, &v[iter * ldv], &inc);
}
