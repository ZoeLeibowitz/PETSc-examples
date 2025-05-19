static char help[] = "1D modified helmholtz problem with DMDA and SNES. With multigrid. Option prefix -rct_.\n\n";

#include <petsc.h>
#include <petscdmda.h>
#include <petscdmshell.h>
#include <petscsnes.h>

typedef struct {
    PetscReal  alpha, beta;
} AppCtx;


// BROKEN PCMGSetRScale()
// mg as a solver:
// ./explicit_mg_matfree_transfers -ksp_monitor -pc_type mg -pc_mg_type full -da_refine 4 -ksp_type preonly -mg_levels_ksp_max_it 10 -mg_levels_pc_type jacobi -mg_levels_ksp_type richardson -ksp_converged_reason -mg_levels_4_pc_type lu


extern PetscReal f_source(PetscReal);
extern PetscErrorCode InitialAndExact(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);
extern PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *dummy);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);

extern PetscErrorCode InterpMult(Mat A, Vec X, Vec Y);

// shell routines

static PetscErrorCode CreateMatrix(DM, Mat *);
static PetscErrorCode CreateGlobalVector(DM, Vec *);
static PetscErrorCode CreateLocalVector(DM, Vec *);
static PetscErrorCode Refine(DM, MPI_Comm, DM *);
static PetscErrorCode Coarsen(DM, MPI_Comm, DM *);
static PetscErrorCode CreateInterpolation(DM, DM, Mat *, Vec *);
static PetscErrorCode CreateRestriction(DM, DM, Mat *);
static PetscErrorCode Destroy(void *);


// solving -phi.laplace + k^2 phi = f on [0,1] with Dirichlet BCs
// phi(0) = alpha = 0., phi(1) = beta = 1.
// f(x) = 2 + k^2*(1-x^2) - (l^2*pi^2 + k^2)*cos(l*pi*x)
// l=3, k=1 :
// => phi_exact(x) = 1 - x^2 - cos(l*pi*x)


typedef struct {
    PetscInt M1, M2;
  } InterpCtx;


static PetscErrorCode MyDMShellCreate(MPI_Comm comm, DM da, DM *shell)
{
  PetscFunctionBeginUser;
  VecScatter da_gtol;
  PetscCall(DMShellCreate(comm, shell));
  PetscCall(DMShellSetContext(*shell, da));
  PetscCall(DMShellSetCreateMatrix(*shell, CreateMatrix));
  PetscCall(DMShellSetCreateGlobalVector(*shell, CreateGlobalVector));
  PetscCall(DMShellSetCreateLocalVector(*shell, CreateLocalVector));
  PetscCall(DMShellSetRefine(*shell, Refine));
  PetscCall(DMShellSetCoarsen(*shell, Coarsen));
  PetscCall(DMShellSetCreateInterpolation(*shell, CreateInterpolation));


  PetscCall(DMShellSetCreateRestriction(*shell, CreateRestriction));
  PetscCall(DMShellSetDestroyContext(*shell, Destroy));

  PetscCall(DMDAGetScatter(da, &da_gtol, NULL));
  PetscCall(DMShellSetGlobalToLocalVecScatter(*shell, da_gtol));

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc,char **args) {
  DM            da, shell;
  SNES          snes;
  AppCtx        user;
  Mat          J;
  Vec           phi, phiexact;
  PetscReal     errnorm, *aphi, *aphiex;
  DMDALocalInfo info;
  PetscInt levels;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  user.alpha = 0.;
  user.beta  = 1.;

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3,1,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetApplicationContext(da,&user));

  PetscCall(DMCreateGlobalVector(da,&phi));
  PetscCall(VecDuplicate(phi,&phiexact));
  PetscCall(DMDAVecGetArray(da,phi,&aphi));

  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDAVecGetArray(da,phiexact,&aphiex));
  PetscCall(InitialAndExact(&info,aphi,aphiex,&user));
  PetscCall(DMDAVecRestoreArray(da,phi,&aphi));
  PetscCall(DMDAVecRestoreArray(da,phiexact,&aphiex));

  // Shell stuff
  PetscCall(MyDMShellCreate(PETSC_COMM_WORLD, da, &shell));
  PetscCall(DMGetRefineLevel(da, &levels));
  PetscCall(DMSetRefineLevel(shell, levels));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,shell));
  PetscCall(SNESSetFunction(snes,NULL,FormFunction,NULL));

  PetscCall(DMCreateMatrix(shell,&J));
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, NULL));

  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(SNESSolve(snes,NULL,phi));

  PetscCall(VecAXPY(phi,-1.0,phiexact));    // phi <- phi + (-1.0) uexact
  PetscCall(VecNorm(phi,NORM_INFINITY,&errnorm));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "on %d point grid:  |u-u_exact|_inf = %g\n",info.mx,errnorm));

  PetscCall(VecDestroy(&phi));
  PetscCall(VecDestroy(&phiexact));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&shell));
  PetscCall(PetscFinalize());
  return 0;
}


PetscReal f_source(PetscReal x) {
    return 2.0 + (1.0 - x * x) - ((9.0 * PETSC_PI * PETSC_PI) + 1.0) * PetscCosReal(3.0 * PETSC_PI * x);
}

PetscErrorCode InitialAndExact(DMDALocalInfo *info, PetscReal *u0,
                               PetscReal *uex, AppCtx *user) {
    PetscInt   i;
    PetscReal  h = 1.0 / (info->mx-1), x;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = h * i;
        // initial guess is just linear interpolation
        // between grid points
        u0[i] = user->alpha + (user->beta - user->alpha) * x;
        uex[i] = 1.0 - (x * x) - PetscCosReal(3.0*PETSC_PI * x);

    }
    return 0;
}


PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *dummy)
{
    PetscFunctionBeginUser;
    Vec xlocal;
    PetscScalar * f_vec;
    PetscScalar * x_vec;
    DM da, dashell;
    PetscInt   i;
    DMDALocalInfo info;
    AppCtx *user;
    PetscInt xs, xm;

    PetscCall(SNESGetDM(snes, &dashell));
    PetscCall(DMShellGetContext(dashell, &da));

    PetscCall(DMDAGetLocalInfo(da, &info));

    PetscReal  h = 1.0 / (info.mx-1), x, R;
    PetscCall(DMGetLocalVector(da, &xlocal));

    PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, xlocal));
    PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, xlocal));

    PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));
    PetscCall(DMGetApplicationContext(da, &user));

    PetscCall(DMDAVecGetArrayRead(da, xlocal, &x_vec));

    PetscCall(DMDAVecGetArray(da, F, &f_vec));

    for (i=info.xs; i<info.xs+info.xm; i++) {
        if (i == 0) {
            f_vec[i] = (x_vec[i] - user->alpha);
        } else if (i == info.mx-1) {
            f_vec[i] = (x_vec[i] - user->beta);
        } else {  // interior location
            if (i == 1) {
                f_vec[i] = - x_vec[i+1] + 2.0 * x_vec[i] - user->alpha;
            } else if (i == info.mx-2) {
                f_vec[i] = - user->beta + 2.0 * x_vec[i] - x_vec[i-1];
            } else {
                f_vec[i] = - x_vec[i+1] + 2.0 * x_vec[i] - x_vec[i-1];
            }
            R = -x_vec[i];
            x = i * h;
            f_vec[i] -= h*h * (R + f_source(x));
        }
    }

    PetscCall(DMDAVecRestoreArrayRead(da, xlocal, &x_vec));
    PetscCall(DMDAVecRestoreArray(da, F, &f_vec));
    PetscCall(DMRestoreLocalVector(da, &xlocal));

    PetscFunctionReturn(0);
}


PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat P, void *dummy)
{

    PetscFunctionBeginUser;

    DM dashell;
    PetscCall(SNESGetDM(snes, &dashell));
    DM da;
    PetscCall(DMShellGetContext(dashell, &da));
    DMDALocalInfo info;
    PetscCall(DMDAGetLocalInfo(da, &info));
    PetscInt   i, col[3];
    PetscReal  h = 1.0 / (info.mx-1), dRdphi, v[3];

    for (i=info.xs; i<info.xs+info.xm; i++) {
        if ((i == 0) | (i == info.mx-1)) {
            v[0] = 1.0;
            PetscCall(MatSetValues(P,1,&i,1,&i,v,INSERT_VALUES));
        } else {
            col[0] = i;
            v[0] = 2.0;
            dRdphi = - 1.0;
            v[0] -= h*h * dRdphi;

            col[1] = i-1;   v[1] = (i > 1) ? - 1.0 : 0.0;
            col[2] = i+1;   v[2] = (i < info.mx-2) ? - 1.0 : 0.0;
            PetscCall(MatSetValues(P,1,&i,3,col,v,INSERT_VALUES));
        }
    }

    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
  PetscFunctionReturn(PETSC_SUCCESS);
}


// shell routines

static PetscErrorCode Destroy(void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(DMDestroy((DM *)&ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMatrix(DM shell, Mat *A)
{
  DM da;

  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMCreateMatrix(da, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}
  
  
static PetscErrorCode CreateInterpolation(DM dm1, DM dm2, Mat *mat, Vec *vec)
{
    DM        da1, da2;
    InterpCtx *ctx;
    PetscInt  M1, M2;
  
    Vec scale;

    PetscFunctionBeginUser;
  
    PetscCall(DMShellGetContext(dm1, &da1));
    PetscCall(DMShellGetContext(dm2, &da2));
  
    PetscCall(DMDAGetInfo(da1, NULL, &M1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    PetscCall(DMDAGetInfo(da2, NULL, &M2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  
    PetscCall(PetscNew(&ctx));
    ctx->M1 = M1;
    ctx->M2 = M2;
  
    PetscCall(MatCreateShell(PETSC_COMM_SELF, M2, M1, M2, M1, ctx, mat));
    PetscCall(MatShellSetOperation(*mat, MATOP_MULT, (void (*)(void))InterpMult));

    // rethink how to do this
    // PetscCall(VecSet(*vec, 1.0));

    // PetscCall(DMCreateInterpolationScale(da1, da2, *mat, vec));

    // PetscCall(VecCreate(PETSC_COMM_SELF, &scale));
    // PetscCall(VecSet(scale, 1.0));


    // PetscCall(MatRestrict(*mat, *vec, scale));
    // set first and last to 0.66667
    // PetscCall(VecSetValue(*vec, 0, 0.66667, INSERT_VALUES));
    // PetscCall(VecSetValue(*vec, M2-1, 0.66667, INSERT_VALUES));

    PetscFunctionReturn(PETSC_SUCCESS);
}
  
PetscErrorCode InterpMult(Mat A, Vec X, Vec Y)
  {
    InterpCtx     *ctx;
    const PetscScalar *x_array;
    PetscScalar       *y_array;
    PetscInt           i;
    Vec xlocal;
  
    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(A, &ctx));
    
    // need to do a global to local for X but need the correct dm ?

    PetscCall(VecGetArrayRead(X, &x_array));
    PetscCall(VecGetArray(Y, &y_array));
  
    for (i = 0; i < ctx->M2; ++i) {
      if (i % 2 == 0) {
          PetscInt col = i / 2;
          y_array[i] = (col < ctx->M1) ? x_array[col] : 0.0;
          } else {
              PetscInt col = (i - 1) / 2;
          if (col + 1 < ctx->M1) {
              y_array[i] = 0.5 * (x_array[col] + x_array[col + 1]);
          } else {
              y_array[i] = 0.0;
          }
          }
      }
    PetscCall(VecRestoreArrayRead(X, &x_array));
    PetscCall(VecRestoreArray(Y, &y_array));
    PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode CreateRestriction(DM dm1, DM dm2, Mat *mat)
{
  DM             da1, da2;
  PetscInt       i, M1, M2;
  PetscInt       row, rows[2];
  PetscScalar    vals[2];

  PetscFunctionBeginUser;

  PetscCall(DMShellGetContext(dm1, &da1));
  PetscCall(DMShellGetContext(dm2, &da2));

  PetscCall(DMDAGetInfo(da1, NULL, &M1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetInfo(da2, NULL, &M2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));

  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, M1, M2, 3, NULL, mat));

  for (i = 0; i < M2; ++i) {
    if (i % 2 == 0) {
      row = i / 2;
      if (row < M1) {
        PetscCall(MatSetValue(*mat, row, i, 1.0, INSERT_VALUES));
      }
    } else {
      row = (i - 1) / 2;
      if (row + 1 < M1) {
        rows[0] = row;
        rows[1] = row + 1;
        vals[0] = 0.5;
        vals[1] = 0.5;
        PetscCall(MatSetValues(*mat, 2, rows, 1, &i, vals, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));

  //view the matview
  // PetscCall(MatView(*mat, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateGlobalVector(DM shell, Vec *x)
{
  DM da;

  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMCreateGlobalVector(da, x));
  PetscCall(VecSetDM(*x, shell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateLocalVector(DM shell, Vec *x)
{
  DM da;

  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMCreateLocalVector(da, x));
  PetscCall(VecSetDM(*x, shell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Refine(DM shell, MPI_Comm comm, DM *dmnew)
{
  DM da, dafine;

  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMRefine(da, comm, &dafine));
  PetscCall(MyDMShellCreate(PetscObjectComm((PetscObject)shell), dafine, dmnew));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Coarsen(DM shell, MPI_Comm comm, DM *dmnew)
{
  DM da, dacoarse;

  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMCoarsen(da, comm, &dacoarse));
  PetscCall(MyDMShellCreate(PetscObjectComm((PetscObject)shell), dacoarse, dmnew));
  PetscFunctionReturn(PETSC_SUCCESS);
}
