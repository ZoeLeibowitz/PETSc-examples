// ED BUELER - Petsc4pdes

#ifndef POISSONFUNCTIONS_H_
#define POISSONFUNCTIONS_H_

/*
In 2D these functions approximate the residual of, and the Jacobian of, the
(slightly-generalized) Poisson equation
    - cx u_xx - cy u_yy = f(x,y)
with Dirichlet boundary conditions  u = g(x,y)  on a domain
Omega = (0,Lx) x (0,Ly) discretized using a DMDA structured grid.

(The domain is an interval, rectangle, or rectangular
solid.)  All of these function work with equally-spaced structured grids.  The
dimensions hx, hy, hz of the rectangular cells can have any positive values.


The matrices A are normalized so that if cells are square (h = hx = hy)
then A / h^d approximates the Laplacian in d dimensions.  This is the way
the rows would be scaled in a Galerkin FEM scheme.  (The entries are O(1)
only if d=2.)  The Dirichlet boundary conditions are approximated using
diagonal Jacobian entries with the same values as the diagonal entries for
points in the interior.  Thus these Jacobian matrices have constant diagonal.
*/

// warning: the user is in charge of setting up ALL of this content!
//STARTDECLARE
typedef struct {
    // domain dimensions
    PetscReal Lx, Ly;
    // coefficients in  - cx u_xx - cy u_yy = f
    PetscReal cx, cy;
    // right-hand-side f(x,y)
    PetscReal (*f_rhs)(PetscReal x, PetscReal y, PetscReal z, void *ctx);
    // Dirichlet boundary condition g(x,y,z)
    PetscReal (*g_bdry)(PetscReal x, PetscReal y, PetscReal z, void *ctx);
    // // additional context; see example usage in ch7/minimal.c
    // void   *addctx;
} PoissonCtx;

PetscErrorCode Poisson2DFunctionLocal(DMDALocalInfo *info,
    PetscReal **au, PetscReal **aF, PoissonCtx *user);

//ENDDECLARE


/* If h = hx = hy and h = L/(m-1) then this generates a 2m-1 bandwidth
sparse matrix.  If cx=cy=1 then it has 4 on the diagonal and -1 or zero in
off-diagonal positions.  For example,
    ./fish -fsh_dim 2 -mat_view :foo.m:ascii_matlab -da_refine N
produces a matrix which can be read into Matlab/Octave.                   */
PetscErrorCode Poisson2DJacobianLocal(DMDALocalInfo *info, PetscReal **au,
                                      Mat J, Mat Jpre, PoissonCtx *user);

PetscErrorCode InitialState(DM da, PetscBool gbdry,
                            Vec u, PoissonCtx *user);

#endif

