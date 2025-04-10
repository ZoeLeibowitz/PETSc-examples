static char help[] = "1D reaction-diffusion problem with DMDA and SNES.  Option prefix -rct_.\n\n";
// Matrix free version of https://github.com/bueler/p4pdes/blob/master/c/ch4/reaction.c
// with a hand-written jacobian

#include <petsc.h>
#include <petscsnes.h>
#include <petscdmda.h>

typedef struct {
    PetscReal  rho, M, alpha, beta;
    PetscBool  noRinJ;
    Vec sol;
} AppCtx;

extern PetscReal f_source(PetscReal);
extern PetscErrorCode InitialAndExact(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);
extern PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *dummy);
extern PetscErrorCode FormJacobian(Mat J, Vec X, Vec Y);

//STARTMAIN
int main(int argc,char **args) {
  DM            da;
  SNES          snes;
  AppCtx        user;
  Vec           u, uexact, r;
  PetscReal     errnorm, *au, *auex;
  DMDALocalInfo info;
  Mat J;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  user.rho   = 10.0;
  user.M     = PetscSqr(user.rho / 12.0);
  user.alpha = user.M;
  user.beta  = 16.0 * user.M;
  user.noRinJ = PETSC_FALSE;
  user.sol = u;

  PetscOptionsBegin(PETSC_COMM_WORLD,"rct_","options for reaction",""); 
  PetscCall(PetscOptionsBool("-noRinJ","do not include R(u) term in Jacobian",
      "reaction.c",user.noRinJ,&(user.noRinJ),NULL));
  PetscOptionsEnd();

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,9,1,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));


  PetscCall(DMCreateGlobalVector(da,&u));

  PetscCall(DMSetMatType(da, MATSHELL));
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,da));
  PetscCall(DMCreateMatrix(da,&J));

  PetscCall(DMSetApplicationContext(da,&user));
  PetscCall(SNESSetJacobian(snes, J, J, MatMFFDComputeJacobian, NULL));
  PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))FormJacobian));
  PetscCall(SNESSetFunction(snes,NULL,FormFunction,NULL));

  PetscCall(VecDuplicate(u,&uexact));
  PetscCall(DMDAVecGetArray(da,u,&au));

  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDAVecGetArray(da,uexact,&auex));
  PetscCall(InitialAndExact(&info,au,auex,&user));
  PetscCall(DMDAVecRestoreArray(da,u,&au));
  PetscCall(DMDAVecRestoreArray(da,uexact,&auex));
  PetscCall(MatSetDM(J,da));

  PetscCall(SNESSetFromOptions(snes));

  PetscCall(SNESSolve(snes,NULL,u));
//   PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uexact
  PetscCall(VecNorm(u,NORM_INFINITY,&errnorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "on %d point grid:  |u-u_exact|_inf = %g\n",info.mx,errnorm));

  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&uexact));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}


PetscReal f_source(PetscReal x) {
    return 0.0;
}

PetscErrorCode InitialAndExact(DMDALocalInfo *info, PetscReal *u0,
                               PetscReal *uex, AppCtx *user) {
    PetscInt   i;
    PetscReal  h = 1.0 / (info->mx-1), x;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = h * i;
        u0[i]  = user->alpha * (1.0 - x) + user->beta * x;
        uex[i] = user->M * PetscPowReal(x + 1.0,4.0);
    }
    return 0;
}


PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *dummy)
{
    PetscFunctionBeginUser;
    Vec flocal, xlocal;
    PetscScalar * f_vec;
    PetscScalar * x_vec;
    DM da;
    PetscInt   i;
    DMDALocalInfo info;
    AppCtx *user;

    PetscCall(SNESGetDM(snes, &da));
    PetscCall(DMDAGetLocalInfo(da, &info));

    PetscReal  h = 1.0 / (info.mx-1), x, R;
    PetscCall(DMGetLocalVector(da, &flocal));
    PetscCall(DMGetLocalVector(da, &xlocal));

    PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, xlocal));
    PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, xlocal));

    PetscCall(VecGetArray(xlocal, &x_vec));
    PetscCall(VecGetArray(flocal, &f_vec));

    PetscCall(DMGetApplicationContext(da, &user));


    for (i=info.xs; i<= info.xm-1; i++) {
        if (i == 0) {
            f_vec[i+1] = x_vec[i+1] - user->alpha;
        }
        else if (i == info.mx-1) {
            f_vec[i+1] = x_vec[i+1] - user->beta;
        }
        else {  // interior location
            if (i == 1) {
                f_vec[i+1] = - x_vec[i+2] + 2.0 * x_vec[i+1] - user->alpha;
            } else if (i == info.mx-2) {
                f_vec[i+1] = - user->beta + 2.0 * x_vec[i+1] - x_vec[i];
            } else {
                f_vec[i+1] = - x_vec[i+2] + 2.0 * x_vec[i+1] - x_vec[i];
            }
            R = - user->rho * PetscSqrtReal(x_vec[i+1]);
            x = (i+1) * h;
            f_vec[i+1] -= h*h * (R + f_source(x));
        }
    }

    PetscCall(VecRestoreArray(xlocal, &x_vec));
    PetscCall(VecRestoreArray(flocal, &f_vec));
    PetscCall(DMLocalToGlobalBegin(da, flocal, INSERT_VALUES, F));
    PetscCall(DMLocalToGlobalEnd(da, flocal, INSERT_VALUES, F));
    PetscCall(DMRestoreLocalVector(da, &flocal));
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
    SNES snes;
    PetscReal dRdu;

    Vec sol_local;

    PetscScalar * x_u_vec;
    PetscScalar * y_u_vec;
    PetscScalar * sol_vec;
  
    PetscCall(MatGetDM(J,&(dm0)));
    PetscCall(DMGetLocalVector(dm0,&(xloc)));
    PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
    PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
    PetscCall(DMGetLocalVector(dm0,&(yloc)));
    PetscCall(VecGetArray(yloc,&y_u_vec));
    PetscCall(VecGetArray(xloc,&x_u_vec));
    PetscCall(DMDAGetLocalInfo(dm0,&(info)));

    PetscReal  h = 1.0 / (info.mx-1), x, R;
    PetscCall(DMGetApplicationContext(dm0,&(user)));

    PetscCall(DMGetLocalVector(dm0,&sol_local));
    PetscCall(DMGlobalToLocal(dm0,user->sol,INSERT_VALUES,sol_local));
    PetscCall(VecGetArray(sol_local,&sol_vec));


    for (int ix = 0; ix <= 8; ix += 1)
    {
        dRdu = (- (user->rho / 2.0) / PetscSqrtReal(sol_vec[ix+1]))*h*h;

        if (ix == 0) {
            y_u_vec[ix + 1] = x_u_vec[ix + 1];
        }
        else if (ix == 8){
            y_u_vec[ix + 1] =  x_u_vec[ix + 1];
        }
        else if (ix == 1){
            y_u_vec[ix + 1] = - x_u_vec[ix + 2] + (2.0 - dRdu) * x_u_vec[ix + 1];
        }
        else if (ix == 7){
            y_u_vec[ix + 1] =  (2.0 - dRdu) * x_u_vec[ix + 1] - x_u_vec[ix];
        }
        else {
            y_u_vec[ix + 1] = - x_u_vec[ix + 2] + (2.0 - dRdu) * x_u_vec[ix + 1] - x_u_vec[ix];

        }

    }
    PetscCall(VecRestoreArray(sol_local,&sol_vec));
    PetscCall(VecRestoreArray(yloc,&y_u_vec));
    PetscCall(VecRestoreArray(xloc,&x_u_vec));
    PetscCall(DMLocalToGlobalBegin(dm0,yloc,INSERT_VALUES,Y));
    PetscCall(DMLocalToGlobalEnd(dm0,yloc,INSERT_VALUES,Y));
    PetscCall(DMRestoreLocalVector(dm0,&(xloc)));
    PetscCall(DMRestoreLocalVector(dm0,&(yloc)));
    PetscCall(DMRestoreLocalVector(dm0,&sol_local));

    PetscFunctionReturn(0);

}