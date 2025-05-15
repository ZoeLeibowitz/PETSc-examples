static char help[] = "1D helmholtz problem with DMDA and SNES matfree.  Option prefix -rct_.\n\n";

// solving -phi.laplace + k^2 phi = f on [0,1] with Dirichlet BCs
// phi(0) = alpha = 0., phi(1) = beta = 1.
// f(x) = 2 + k^2*(1-x^2) - (l^2*pi^2 + k^2)*cos(l*pi*x)
// l=3, k=1 :
// => phi_exact(x) = 1 - x^2 - cos(l*pi*x)

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


int main(int argc,char **args) {
  DM            da;
  SNES          snes;
  AppCtx        user;
  Vec           phi, phiexact;
  PetscReal     errnorm, *aphi, *aphiex;
  DMDALocalInfo info;
  Mat J;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  user.alpha = 0.;
  user.beta  = 1.;

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,257,1,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateGlobalVector(da,&phi));

  PetscCall(DMSetMatType(da, MATSHELL));
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,da));
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(DMCreateMatrix(da,&J));

  PetscCall(DMSetApplicationContext(da,&user));
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
  PetscCall(MatSetDM(J,da));
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
    DM da;
    PetscInt   i;
    DMDALocalInfo info;
    AppCtx *user;
    PetscInt xs, xm, Mx;

    PetscCall(SNESGetDM(snes, &da));
    PetscCall(DMDAGetLocalInfo(da, &info));

    PetscReal  h = 1.0 / (info.mx-1), x, R;
    PetscCall(DMGetLocalVector(da, &xlocal));

    PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, xlocal));
    PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, xlocal));

    PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));
    PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

    PetscCall(DMGetApplicationContext(da, &user));

    PetscCall(DMDAVecGetArrayRead(da, xlocal, &x_vec));
    PetscCall(DMDAVecGetArray(da, F, &f_vec));

    for (i=xs; i<xs+xm; i++) {
        if (i==0){
            f_vec[i] = x_vec[i] - user->alpha;
        }

        else if (i==Mx-1){
            f_vec[i] = x_vec[i] - user->beta;
        }
        else{
            if (i==1){
                f_vec[i] = - x_vec[i+1] + 2.0 * x_vec[i] - user->alpha;
            }
            else if (i==Mx-2){
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

    DM dm0;
    DMDALocalInfo info;
    Vec xloc;
    Vec yloc;
    AppCtx *user;
    PetscReal dRdphi, h;
    PetscInt    xs, xm, Mx;

    PetscScalar * x_u_vec;
    PetscScalar * y_u_vec;
  
    PetscCall(MatGetDM(J,&(dm0)));
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

    PetscCall(DMDAGetInfo(dm0, PETSC_IGNORE, &Mx, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

    for (int ix = xs; ix < xs+xm; ix++)
    {
        dRdphi = (-1.0)*h*h;

        if (ix == 0) {
            y_u_vec[ix] = x_u_vec[ix];
        }
        else if (ix == Mx-1){
            y_u_vec[ix] =  x_u_vec[ix];
        }
        else if (ix == 1){
            y_u_vec[ix] = - x_u_vec[ix + 1] + (2.0 - dRdphi) * x_u_vec[ix];
    
        }
        else if (ix == Mx-2){
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
