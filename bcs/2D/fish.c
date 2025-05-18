// ED BUELER - Petsc4pdes

static char help[] =
"Solves structured-grid Poisson problem in 2D.  Option prefix fsh_.\n"
"Equation is\n"
"    - cx u_xx - cy u_yy = f,\n"
"subject to Dirichlet boundary conditions.  Solves three different problems\n"
"where exact solution is known.  Uses DMDA and SNES.  Equation is put in form\n"
"F(u) = - grad^2 u - f.  Call-backs fully-rediscretize for the supplied grid.\n"
"Defaults to 2D, a SNESType of KSPONLY, and a KSPType of CG.\n\n";

#include <petsc.h>
#include "poissonfunctions.h"

// exact solutions  u(x,y),  for boundary condition and error calculation


static PetscReal u_exact_2Dmanupoly(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    return x*x * (1.0 - x*x) * y*y *(y*y - 1.0);
}

static PetscReal u_exact_2Dmanuexp(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    return - x * PetscExpReal(y);
}

static PetscReal zero(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    return 0.0;
}

// right-hand-side functions  f(x,y) = - laplacian u

static PetscReal f_rhs_2Dmanupoly(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    PoissonCtx* user = (PoissonCtx*)ctx;
    PetscReal   aa, bb, ddaa, ddbb;
    aa = x*x * (1.0 - x*x);
    bb = y*y * (y*y - 1.0);
    ddaa = 2.0 * (1.0 - 6.0 * x*x);
    ddbb = 2.0 * (6.0 * y*y - 1.0);
    return - (user->cx * ddaa * bb + user->cy * aa * ddbb);
}

static PetscReal f_rhs_2Dmanuexp(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    return x * PetscExpReal(y);  // note  f = - (u_xx + u_yy) = - u
}

// functions simply to put u_exact()=g_bdry() into a grid
// these are irritatingly-dimension-dependent inside ...
extern PetscErrorCode Form2DUExact(DMDALocalInfo*, Vec, PoissonCtx*);

//STARTPTRARRAYS
// arrays of pointers to functions
// static DMDASNESFunctionFn* residual_ptr[1]
//     = {(DMDASNESFunctionFn*)&Poisson2DFunctionLocal};

// static DMDASNESJacobianFn* jacobian_ptr[1]
//     = {(DMDASNESJacobianFn*)&Poisson2DJacobianLocal};

static DMDASNESFunctionFn *residual_ptr = (DMDASNESFunctionFn *)&Poisson2DFunctionLocal;
    
static DMDASNESJacobianFn* jacobian_ptr = (DMDASNESJacobianFn *)&Poisson2DJacobianLocal;

typedef PetscErrorCode (*ExactFcnVec)(DMDALocalInfo*,Vec,PoissonCtx*);

// static ExactFcnVec getuexact_ptr[1]
//     = {&Form2DUExact};

static ExactFcnVec getuexact_ptr = &Form2DUExact;

//ENDPTRARRAYS

typedef enum {MANUPOLY, MANUEXP, ZERO} ProblemType;
static const char* ProblemTypes[] = {"manupoly","manuexp","zero",
                                     "ProblemType", "", NULL};

// more arrays of pointers to functions:   ..._ptr[DIMS][PROBLEMS]
typedef PetscReal (*PointwiseFcn)(PetscReal,PetscReal,PetscReal,void*);

static PointwiseFcn g_bdry_ptr[1][3]
    = {{&u_exact_2Dmanupoly, &u_exact_2Dmanuexp, &zero}};

static PointwiseFcn f_rhs_ptr[1][3]
    = {{&f_rhs_2Dmanupoly, &f_rhs_2Dmanuexp, &zero}};

static const char* InitialTypes[] = {"zeros","random",
                                     "InitialType", "", NULL};

int main(int argc,char **argv) {
    DM             da, da_after;
    SNES           snes;
    KSP            ksp;
    Vec            u_initial, u, u_exact;
    PoissonCtx     user;
    DMDALocalInfo  info;
    PetscReal      errinf, normconst2h, err2h;
    char           gridstr[99];
    ExactFcnVec    getuexact;

    // fish defaults:
    PetscInt       dim = 2;                  // 2D
    ProblemType    problem = MANUEXP;        // manufactured problem using exp()
    InitialType    initial = ZEROS;          // set u=0 for initial iterate
    PetscBool      gonboundary = PETSC_TRUE; // initial iterate has u=g on boundary

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    // get options and configure context
    user.Lx = 1.0;
    user.Ly = 1.0;
    user.Lz = 1.0;
    user.cx = 1.0;
    user.cy = 1.0;
    user.cz = 1.0;
    PetscOptionsBegin(PETSC_COMM_WORLD,"fsh_", "options for fish.c", "");
    PetscCall(PetscOptionsReal("-cx",
         "set coefficient of x term u_xx in equation",
         "fish.c",user.cx,&user.cx,NULL));
    PetscCall(PetscOptionsReal("-cy",
         "set coefficient of y term u_yy in equation",
         "fish.c",user.cy,&user.cy,NULL));
    PetscCall(PetscOptionsReal("-cz",
         "set coefficient of z term u_zz in equation",
         "fish.c",user.cz,&user.cz,NULL));
    PetscCall(PetscOptionsInt("-dim",
         "dimension of problem (=1,2,3 only)",
         "fish.c",dim,&dim,NULL));
    PetscCall(PetscOptionsBool("-initial_gonboundary",
         "set initial iterate to have correct boundary values",
         "fish.c",gonboundary,&gonboundary,NULL));
    PetscCall(PetscOptionsEnum("-initial_type",
         "type of initial iterate",
         "fish.c",InitialTypes,(PetscEnum)initial,(PetscEnum*)&initial,NULL));
    PetscCall(PetscOptionsReal("-Lx",
         "set Lx in domain ([0,Lx] x [0,Ly] x [0,Lz], etc.)",
         "fish.c",user.Lx,&user.Lx,NULL));
    PetscCall(PetscOptionsReal("-Ly",
         "set Ly in domain ([0,Lx] x [0,Ly] x [0,Lz], etc.)",
         "fish.c",user.Ly,&user.Ly,NULL));
    PetscCall(PetscOptionsReal("-Lz",
         "set Ly in domain ([0,Lx] x [0,Ly] x [0,Lz], etc.)",
         "fish.c",user.Lz,&user.Lz,NULL));
    PetscCall(PetscOptionsEnum("-problem",
         "problem type; determines exact solution and RHS",
         "fish.c",ProblemTypes,(PetscEnum)problem,(PetscEnum*)&problem,NULL));
    PetscOptionsEnd();
    user.g_bdry = g_bdry_ptr[0][problem];
    user.f_rhs = f_rhs_ptr[0][problem];

//STARTCREATE

    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR, 3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
    
    PetscCall(DMSetApplicationContext(da,&user));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));  // call BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(da,0.0,user.Lx,0.0,user.Ly,0.0,user.Lz));

    // set SNES call-backs
    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetDM(snes,da));

    PetscCall(DMDASNESSetFunctionLocal(da, INSERT_VALUES, residual_ptr, &user));
    PetscCall(DMDASNESSetJacobianLocal(da, jacobian_ptr, &user));
             

    // default to KSPONLY+CG because problem is linear and SPD
    PetscCall(SNESSetType(snes,SNESKSPONLY));
    PetscCall(SNESGetKSP(snes,&ksp));
    PetscCall(KSPSetType(ksp,KSPCG));
    PetscCall(SNESSetFromOptions(snes));

    // set initial iterate and then solve
    PetscCall(DMGetGlobalVector(da,&u_initial));
    PetscCall(InitialState(da, initial, gonboundary, u_initial, &user));
    //view u_initial
    PetscCall(VecView(u_initial,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(SNESSolve(snes,NULL,u_initial));
    PetscCall(VecView(u_initial,PETSC_VIEWER_STDOUT_WORLD));
//ENDCREATE

//STARTGETSOLUTION
    // -snes_grid_sequence could change grid resolution
    PetscCall(DMRestoreGlobalVector(da,&u_initial));
    PetscCall(DMDestroy(&da));

    // evaluate error and report
    PetscCall(SNESGetSolution(snes,&u));  // SNES owns u; do not destroy it
    PetscCall(SNESGetDM(snes,&da_after)); // SNES owns da_after; do not destroy it
    PetscCall(DMDAGetLocalInfo(da_after,&info));
    PetscCall(DMCreateGlobalVector(da_after,&u_exact));

    getuexact = getuexact_ptr;

    PetscCall((*getuexact)(&info,u_exact,&user));
    PetscCall(VecAXPY(u,-1.0,u_exact));   // u <- u + (-1.0) uexact
    PetscCall(VecDestroy(&u_exact));      // no longer needed
    PetscCall(VecNorm(u,NORM_INFINITY,&errinf));
    PetscCall(VecNorm(u,NORM_2,&err2h));
//ENDGETSOLUTION


    normconst2h = PetscSqrtReal((PetscReal)(info.mx-1)*(info.my-1));
    snprintf(gridstr,99,"%d x %d point 2D",info.mx,info.my);

    err2h /= normconst2h; // like continuous L2
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "problem %s on %s grid:\n"
                "  error |u-uexact|_inf = %.3e, |u-uexact|_h = %.3e\n",
                ProblemTypes[problem],gridstr,errinf,err2h));

    // destroy what we explicitly Created
    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}


PetscErrorCode Form2DUExact(DMDALocalInfo *info, Vec u, PoissonCtx* user) {
    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, x, y, **au;
    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    PetscCall(DMDAVecGetArray(info->da, u, &au));
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = xymin[1] + j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = xymin[0] + i * hx;
            au[j][i] = user->g_bdry(x,y,0.0,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da, u, &au));
    return 0;
}
