static char help[] = "1D modified helmholtz problem with DMDA and SNES. No multigrid. Option prefix -rct_.\n\n";

#include <petsc.h>

typedef struct {
    PetscReal  alpha, beta;
} AppCtx;

extern PetscReal f_source(PetscReal);
extern PetscErrorCode InitialAndExact(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, PetscReal*, Mat, Mat, AppCtx*);


// solving -phi.laplace + k^2 phi = f on [0,1] with Dirichlet BCs
// phi(0) = alpha = 0., phi(1) = beta = 1.
// f(x) = 2 + k^2*(1-x^2) - (l^2*pi^2 + k^2)*cos(l*pi*x)
// l=3, k=1 :
// => phi_exact(x) = 1 - x^2 - cos(l*pi*x)


int main(int argc,char **args) {
  DM            da;
  SNES          snes;
  AppCtx        user;
  Vec           phi, phiexact;
  PetscReal     errnorm, *aphi, *aphiex;
  DMDALocalInfo info;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  user.alpha = 0.;
  user.beta  = 1.;

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,33,1,1,NULL,&da));
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

  PetscCall(VecView(phiexact,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(phi,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,da));
  PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunctionFn *)FormFunctionLocal,&user));
  PetscCall(DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobianFn *)FormJacobianLocal,&user));
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(SNESSolve(snes,NULL,phi));
//   PetscCall(VecView(phi, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecAXPY(phi,-1.0,phiexact));    // phi <- phi + (-1.0) uexact
  PetscCall(VecNorm(phi,NORM_INFINITY,&errnorm));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "on %d point grid:  |u-u_exact|_inf = %g\n",info.mx,errnorm));

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

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal *phi,
                                 PetscReal *FF, AppCtx *user) {
    PetscInt   i;
    PetscReal  h = 1.0 / (info->mx-1), x, R;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0) {
            FF[i] = (phi[i] - user->alpha);
        } else if (i == info->mx-1) {
            FF[i] = (phi[i] - user->beta);
        } else {  // interior location
            if (i == 1) {
                FF[i] = - phi[i+1] + 2.0 * phi[i] - user->alpha;
            } else if (i == info->mx-2) {
                FF[i] = - user->beta + 2.0 * phi[i] - phi[i-1];
            } else {
                FF[i] = - phi[i+1] + 2.0 * phi[i] - phi[i-1];
            }
            R = -phi[i];
            x = i * h;
            FF[i] -= h*h * (R + f_source(x));
        }
    }
    return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscReal *phi,
                                 Mat J, Mat P, AppCtx *user) {
    PetscInt   i, col[3];
    PetscReal  h = 1.0 / (info->mx-1), dRdphi, v[3];
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if ((i == 0) | (i == info->mx-1)) {
            v[0] = 1.0;
            PetscCall(MatSetValues(P,1,&i,1,&i,v,INSERT_VALUES));
        } else {
            col[0] = i;
            v[0] = 2.0;
            dRdphi = - 1.0;
            v[0] -= h*h * dRdphi;

            col[1] = i-1;   v[1] = (i > 1) ? - 1.0 : 0.0;
            col[2] = i+1;   v[2] = (i < info->mx-2) ? - 1.0 : 0.0;
            PetscCall(MatSetValues(P,1,&i,3,col,v,INSERT_VALUES));
        }
    }
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}
