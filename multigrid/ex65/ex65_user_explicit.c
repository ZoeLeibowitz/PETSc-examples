/*
 Partial differential equation

   d   d u = 1, 0 < x < 1,
   --   --
   dx   dx
with boundary conditions

   u = 0 for x = 0, x = 1

   This uses multigrid to solve the linear system

   Demonstrates how to build a DMSHELL for managing multigrid. The DMSHELL simply creates a
   DMDA1d to construct all the needed PETSc objects.

*/

static char help[] = "Solves 1D constant coefficient Laplacian using DMSHELL and multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmshell.h>
#include <petscksp.h>

static PetscErrorCode ComputeMatrix(KSP, Mat, Mat, void *);
static PetscErrorCode ComputeRHS(KSP, Vec, void *);
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  KSP      ksp;
  DM       da, shell;
  PetscInt levels;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 5, 1, 1, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(MyDMShellCreate(PETSC_COMM_WORLD, da, &shell));
  /* these two lines are not needed but allow PCMG to automatically know how many multigrid levels the user wants */
  PetscCall(DMGetRefineLevel(da, &levels));
  PetscCall(DMSetRefineLevel(shell, levels));


  PetscCall(KSPSetDM(ksp, shell));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, NULL));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrix, NULL));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, NULL, NULL));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&shell));
  PetscCall(PetscFinalize());
  return 0;
}

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

// static PetscErrorCode CreateInterpolation(DM dm1, DM dm2, Mat *mat, Vec *vec)
// {
//   DM da1, da2;

//   PetscFunctionBeginUser;
//   PetscCall(DMShellGetContext(dm1, &da1));
//   PetscCall(DMShellGetContext(dm2, &da2));
//   PetscCall(DMCreateInterpolation(da1, da2, mat, vec));
//   PetscCall(MatView(*mat, PETSC_VIEWER_STDOUT_WORLD));

//   // can use this instead of using dmcreateinterpolation to generate it
//   PetscCall(DMCreateInterpolationScale(da1, da2, *mat, vec));

//   PetscFunctionReturn(PETSC_SUCCESS);
// }

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
  // PetscCall(MatView(*mat, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}



static PetscErrorCode CreateRestriction(DM dm1, DM dm2, Mat *mat)
{
  DM  da1, da2;
  Mat tmat;
  PetscFunctionBeginUser;
  PetscCall(DMShellGetContext(dm1, &da1));
  PetscCall(DMShellGetContext(dm2, &da2));
  PetscCall(DMCreateInterpolation(da1, da2, &tmat, NULL));
  PetscCall(MatTranspose(tmat, MAT_INITIAL_MATRIX, mat));
//   PetscCall(MatView(*mat, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&tmat));
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

static PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  PetscInt    mx, idx[2];
  PetscScalar h, v[2];
  DM          da, shell;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &shell));
  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  h = 1.0 / (mx - 1);
  PetscCall(VecSet(b, h));
  idx[0] = 0;
  idx[1] = mx - 1;
  v[0] = v[1] = 0.0;
  PetscCall(VecSetValues(b, 2, idx, v, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx)
{
  PetscInt    i, mx, xm, xs;
  PetscScalar v[3], h;
  MatStencil  row, col[3];
  DM          da, shell;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &shell));
  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, 0, 0, &xm, 0, 0));
  h = 1.0 / (mx - 1);

  for (i = xs; i < xs + xm; i++) {
    row.i = i;
    if (i == 0 || i == mx - 1) {
      v[0] = 2.0 / h;
      PetscCall(MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES));
    } else {
      v[0]     = (-1.0) / h;
      col[0].i = i - 1;
      v[1]     = (2.0) / h;
      col[1].i = row.i;
      v[2]     = (-1.0) / h;
      col[2].i = i + 1;
      PetscCall(MatSetValuesStencil(jac, 1, &row, 3, col, v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:
      nsize: 4
      args: -ksp_monitor -pc_type mg -da_refine 3

TEST*/
