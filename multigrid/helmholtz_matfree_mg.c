static char help[] = "1D helmholtz problem with DMDA and SNES matfree. Multigrid.  Option prefix -rct_.\n\n";

// solving -phi.laplace + k^2 phi = f on [0,1] with Dirichlet BCs
// phi(0) = alpha = 0., phi(1) = beta = 1.
// f(x) = 2 + k^2*(1-x^2) - (l^2*pi^2 + k^2)*cos(l*pi*x)
// l=3, k=1 :
// => phi_exact(x) = 1 - x^2 - cos(l*pi*x)


// note: broken ./helmholtz_matfree_mg -ksp_monitor -pc_type mg -mg_levels_1_pc_type none -mg_levels_1_ksp_type gmres -mg_levels_0_pc_type none -mg_levels_0_ksp_type gmres -da_refine 1

#include <petsc.h>
#include <petscsnes.h>
#include <petscdmda.h>

typedef struct {
    PetscReal  alpha, beta;
} AppCtx;

extern PetscReal f_source(PetscReal);
extern PetscErrorCode InitialAndExact(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);
extern PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *dummy);
extern PetscErrorCode FormJacobian(Mat J, Vec X, Vec Y);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);


// shell routines

static PetscErrorCode CreateMatrix(DM, Mat *);
static PetscErrorCode CreateGlobalVector(DM, Vec *);
static PetscErrorCode CreateLocalVector(DM, Vec *);
static PetscErrorCode Refine(DM, MPI_Comm, DM *);
static PetscErrorCode Coarsen(DM, MPI_Comm, DM *);
static PetscErrorCode CreateInterpolation(DM, DM, Mat *, Vec *);
static PetscErrorCode CreateRestriction(DM, DM, Mat *);
static PetscErrorCode Destroy(void *);



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
  Vec           phi, phiexact;
  PetscReal     errnorm, *aphi, *aphiex;
  DMDALocalInfo info;
  Mat J;
  PetscInt levels;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  user.alpha = 0.;
  user.beta  = 1.;

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,9,1,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetApplicationContext(da,&user));

  PetscCall(DMCreateGlobalVector(da,&phi));
  PetscCall(DMSetMatType(da, MATSHELL));

  // Shell stuff
  PetscCall(MyDMShellCreate(PETSC_COMM_WORLD, da, &shell));
  PetscCall(DMGetRefineLevel(da, &levels));
  PetscCall(DMSetRefineLevel(shell, levels));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,shell));
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(DMCreateMatrix(shell,&J));

  PetscCall(SNESSetJacobian(snes, J, J, MatMFFDComputeJacobian, NULL));
  PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))FormJacobian));
  PetscCall(SNESSetFunction(snes,NULL,FormFunction,NULL));

  PetscCall(VecDuplicate(phi,&phiexact));
  PetscCall(DMDAVecGetArray(da,phi,&aphi));

  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDAVecGetArray(da,phiexact,&aphiex));
  PetscCall(InitialAndExact(&info,aphi,aphiex,&user));
  PetscCall(DMDAVecRestoreArray(da,phi,&aphi));
  PetscCall(DMDAVecRestoreArray(da,phiexact,&aphiex));
  PetscCall(MatSetDM(J,shell));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(SNESSolve(snes,NULL,phi));

  PetscCall(VecAXPY(phi,-1.0,phiexact));    // phi <- phi + (-1.0) phiexact
  PetscCall(VecNorm(phi,NORM_INFINITY,&errnorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "on %d point grid:  |phi-phi_exact|_inf = %g\n",info.mx,errnorm));

  PetscCall(VecDestroy(&phi));
  PetscCall(VecDestroy(&phiexact));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}


PetscReal f_source(PetscReal x) {
    return 2.0 + (1.0 - x * x) - ((9.0 * PETSC_PI * PETSC_PI) + 1.0) * PetscCosReal(3.0 * PETSC_PI * x);
}


PetscErrorCode InitialAndExact(DMDALocalInfo *info, PetscReal *phi0,
    PetscReal *phiex, AppCtx *user) {
    PetscInt   i;
    PetscReal  h = 1.0 / (info->mx-1), x;
    for (i=info->xs; i<info->xs+info->xm; i++) {
    x = h * i;
    // initial guess is just linear interpolation
    // between grid points
    phi0[i] = user->alpha + (user->beta - user->alpha) * x;
    phiex[i] = 1.0 - (x * x) - PetscCosReal(3.0*PETSC_PI * x);
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

    for (i=xs; i<xs+xm; i++) {
        if (i==0){
            f_vec[i] = x_vec[i] - user->alpha;
        }
        else if (i==info.mx-1){
            f_vec[i] = x_vec[i] - user->beta;
        }
        else{
            if (i==1){
                f_vec[i] = - x_vec[i+1] + 2.0 * x_vec[i] - user->alpha;
            }
            else if (i==info.mx-2){
                f_vec[i] = - user->beta + 2.0 * x_vec[i] - x_vec[i-1];
            }
            else{
                f_vec[i] = - x_vec[i+1] + 2.0 * x_vec[i] - x_vec[i-1];
            }
            R = -x_vec[i];
            x = (i) * h;
            f_vec[i] -= h*h * (R + f_source(x));
        }
    }

    PetscCall(DMDAVecRestoreArrayRead(da, xlocal, &x_vec));
    PetscCall(DMDAVecRestoreArray(da, F, &f_vec));
    PetscCall(DMRestoreLocalVector(da, &xlocal));
    PetscFunctionReturn(0);
}


PetscErrorCode FormJacobian(Mat J, Vec X, Vec Y)
{
    PetscFunctionBeginUser;

    DM dm0, dashell;
    DMDALocalInfo info;
    Vec xloc;
    Vec yloc;
    AppCtx *user;
    PetscReal dRdphi, h;
    PetscInt    xs, xm;

    PetscScalar * x_u_vec;
    PetscScalar * y_u_vec;
  
    PetscCall(MatGetDM(J,&dashell));
    PetscCall(DMShellGetContext(dashell, &dm0));

    PetscCall(DMGetLocalVector(dm0,&(xloc)));

    PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
    PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));

    PetscCall(DMGetLocalVector(dm0,&(yloc)));
    PetscCall(DMDAVecGetArray(dm0,yloc,&y_u_vec));
    PetscCall(DMDAVecGetArray(dm0,xloc,&x_u_vec));
    PetscCall(DMDAGetLocalInfo(dm0,&(info)));

    h = 1.0 / (info.mx-1);
    PetscCall(DMGetApplicationContext(dm0,&(user)));

    PetscCall(DMDAGetCorners(dm0, &xs, NULL, NULL, &xm, NULL, NULL));

    for (int ix = xs; ix < xs+xm; ix++)
    {
        dRdphi = (-1.0)*h*h;

        if (ix == 0) {
            y_u_vec[ix] = x_u_vec[ix];
        }
        else if (ix == info.mx-1){
            y_u_vec[ix] =  x_u_vec[ix];
        }
        else if (ix == 1){
            y_u_vec[ix] = - x_u_vec[ix + 1] + (2.0 - dRdphi) * x_u_vec[ix];
    
        }
        else if (ix == info.mx-2){
            y_u_vec[ix] = (2.0 - dRdphi) * x_u_vec[ix] - x_u_vec[ix-1];
        }
        else {
            y_u_vec[ix] = - x_u_vec[ix + 1] + (2.0 - dRdphi) * x_u_vec[ix] - x_u_vec[ix-1];
        }
    }

    PetscCall(DMDAVecRestoreArray(dm0,yloc,&y_u_vec));

    PetscCall(DMDAVecRestoreArray(dm0,xloc,&x_u_vec));
    PetscCall(DMLocalToGlobalBegin(dm0,yloc,INSERT_VALUES,Y));
    PetscCall(DMLocalToGlobalEnd(dm0,yloc,INSERT_VALUES,Y));
    PetscCall(DMRestoreLocalVector(dm0,&(xloc)));
    PetscCall(DMRestoreLocalVector(dm0,&(yloc)));

    PetscFunctionReturn(0);

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
  DM             da1, da2;
  PetscInt       i, M1, M2;
  PetscInt       col, cols[2];
  PetscScalar    vals[2];

  PetscFunctionBeginUser;

  PetscCall(DMShellGetContext(dm1, &da1)); 
  PetscCall(DMShellGetContext(dm2, &da2));

  PetscCall(DMDAGetInfo(da1, NULL, &M1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetInfo(da2, NULL, &M2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, M2, M1, 2, NULL, mat));

  for (i = 0; i < M2; ++i) {
    if (i % 2 == 0) {
      col = i / 2;
      if (col < M1) {
        PetscCall(MatSetValue(*mat, i, col, 1.0, INSERT_VALUES));
      }
    } else {
      col = (i - 1) / 2;
      if (col + 1 < M1) {
        cols[0] = col;
        cols[1] = col + 1;
        vals[0] = 0.5;
        vals[1] = 0.5;
        PetscCall(MatSetValues(*mat, 1, &i, 2, cols, vals, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));

  PetscCall(DMCreateInterpolationScale(da1, da2, *mat, vec));
//   PetscCall(MatView(*mat, PETSC_VIEWER_STDOUT_WORLD));
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

  //view the mat
  PetscCall(MatView(*mat, PETSC_VIEWER_STDOUT_WORLD));
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