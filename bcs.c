#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "petscsnes.h"
#include "petscdmda.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct UserCtx0
{
  PetscScalar h_x;
  PetscInt x_M;
  PetscInt x_ltkn0;
  PetscInt x_ltkn1;
  PetscInt x_m;
  PetscInt x_rtkn0;
  PetscInt x_rtkn2;
  struct dataobj * f2_vec;
} ;

struct dataobj
{
  void * data;
  unsigned long * size;
  unsigned long * npsize;
  unsigned long * dsize;
  PetscInt * hsize;
  PetscInt * hofs;
  PetscInt * oofs;
  void * dmap;
} ;

struct profiler
{
  PetscScalar section0;
} ;

PetscErrorCode MatMult0(Mat J, Vec X, Vec Y);
PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy);
PetscErrorCode FormRHS0(DM dm0, Vec B);
PetscErrorCode FormInitialGuess0(DM dm0, Vec xloc);
PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * f2_vec, const PetscScalar h_x, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2);

int Kernel(struct dataobj * f1_vec, struct dataobj * f2_vec, const PetscScalar h_x, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2, struct profiler * timers)
{
  Mat J0;
  Vec bglobal0;
  DM da0;
  KSP ksp0;
  PetscInt localsize0;
  PC pc0;
  PetscMPIInt size;
  SNES snes0;
  Vec xglobal0;
  Vec xlocal0;

  struct UserCtx0 ctx0;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&(size)));

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,6,1,2,NULL,&(da0)));
  PetscCall(DMSetUp(da0));
  PetscCall(DMSetMatType(da0,MATSHELL));
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&(snes0)));
  PetscCall(SNESSetDM(snes0,da0));
  PetscCall(DMCreateMatrix(da0,&(J0)));
  PetscCall(SNESSetJacobian(snes0,J0,J0,MatMFFDComputeJacobian,NULL));
  PetscCall(SNESSetType(snes0,SNESKSPONLY));
  PetscCall(DMCreateGlobalVector(da0,&(xglobal0)));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_WORLD,1,f1_vec->size[0],f1_vec->data,&(xlocal0)));
  PetscCall(VecGetSize(xlocal0,&(localsize0)));
  PetscCall(DMCreateGlobalVector(da0,&(bglobal0)));
  PetscCall(SNESGetKSP(snes0,&(ksp0)));
  PetscCall(KSPSetTolerances(ksp0,1e-05,1e-50,100000.0,10000.0));
  PetscCall(KSPSetType(ksp0,KSPGMRES));
  PetscCall(KSPGetPC(ksp0,&(pc0)));
  PetscCall(PCSetType(pc0,PCNONE));
  PetscCall(KSPSetFromOptions(ksp0));
  PetscCall(MatShellSetOperation(J0,MATOP_MULT,(void (*)(void))MatMult0));
  PetscCall(SNESSetFunction(snes0,NULL,FormFunction0,(void*)(da0)));
  PetscCall(PopulateUserContext0(&(ctx0),f2_vec,h_x,x_M,x_ltkn0,x_ltkn1,x_m,x_rtkn0,x_rtkn2));
  PetscCall(MatSetDM(J0,da0));
  PetscCall(DMSetApplicationContext(da0,&(ctx0)));


  START(section0)
  PetscCall(FormRHS0(da0,bglobal0));
  PetscCall(FormInitialGuess0(da0,xlocal0));
  PetscCall(DMLocalToGlobal(da0,xlocal0,INSERT_VALUES,xglobal0));
  PetscCall(SNESSolve(snes0,bglobal0,xglobal0));
  PetscCall(DMGlobalToLocal(da0,xglobal0,INSERT_VALUES,xlocal0));

  STOP(section0,timers)

  PetscCall(VecDestroy(&(bglobal0)));
  PetscCall(VecDestroy(&(xglobal0)));
  PetscCall(VecDestroy(&(xlocal0)));
  PetscCall(MatDestroy(&(J0)));
  PetscCall(SNESDestroy(&(snes0)));
  PetscCall(DMDestroy(&(da0)));

  return 0;
}

PetscErrorCode MatMult0(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  DM dm0;
  Vec xloc;
  Vec yloc;

  struct UserCtx0 * ctx0;
  PetscScalar * x_f1_vec;
  PetscScalar * y_f1_vec;

  PetscCall(MatGetDM(J,&(dm0)));
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  PetscCall(DMGetLocalVector(dm0,&(xloc)));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&(yloc)));
  PetscCall(VecGetArray(yloc,&y_f1_vec));
  PetscCall(VecGetArray(xloc,&x_f1_vec));

  PetscScalar (* x_f1) = (PetscScalar (*)) x_f1_vec;
  PetscScalar (* y_f1) = (PetscScalar (*)) y_f1_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    y_f1[ix + 2] = -(-x_f1[ix + 2] + x_f1[ix + 3])/ctx0->h_x + x_f1[ix + 2];
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    y_f1[ix + 2] = x_f1[ix + 2];
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    y_f1[ix + 2] = x_f1[ix + 2];
  }
  PetscCall(VecRestoreArray(yloc,&y_f1_vec));
  PetscCall(VecRestoreArray(xloc,&x_f1_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,yloc,INSERT_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm0,yloc,INSERT_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm0,&(xloc)));
  PetscCall(DMRestoreLocalVector(dm0,&(yloc)));

  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy)
{
  PetscFunctionBeginUser;

  Vec floc;
  Vec xloc;

  struct UserCtx0 * ctx0;
  PetscScalar * f_f1_vec;
  PetscScalar * x_f1_vec;

  DM dm0 = (DM)(dummy);
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  PetscCall(DMGetLocalVector(dm0,&(xloc)));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&(floc)));
  PetscCall(VecGetArray(floc,&f_f1_vec));
  PetscCall(VecGetArray(xloc,&x_f1_vec));

  PetscScalar (* f_f1) = (PetscScalar (*)) f_f1_vec;
  PetscScalar (* x_f1) = (PetscScalar (*)) x_f1_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    f_f1[ix + 2] = -(-x_f1[ix + 2] + x_f1[ix + 3])/ctx0->h_x + x_f1[ix + 2];
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    f_f1[ix + 2] = 0.0;
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    f_f1[ix + 2] = 0.0;
  }
  PetscCall(VecRestoreArray(floc,&f_f1_vec));
  PetscCall(VecRestoreArray(xloc,&x_f1_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,floc,INSERT_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(dm0,floc,INSERT_VALUES,F));
  PetscCall(DMRestoreLocalVector(dm0,&(xloc)));
  PetscCall(DMRestoreLocalVector(dm0,&(floc)));

  PetscFunctionReturn(0);
}

PetscErrorCode FormRHS0(DM dm0, Vec B)
{
  PetscFunctionBeginUser;

  Vec blocal0;

  PetscScalar * b_f1_vec;
  struct UserCtx0 * ctx0;

  PetscCall(DMGetLocalVector(dm0,&(blocal0)));
  PetscCall(DMGlobalToLocalBegin(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(DMGlobalToLocalEnd(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(VecGetArray(blocal0,&b_f1_vec));
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  struct dataobj * f2_vec = ctx0->f2_vec;

  PetscScalar (* b_f1) = (PetscScalar (*)) b_f1_vec;
  PetscScalar (* f2) __attribute__ ((aligned (64))) = (PetscScalar (*)) f2_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    b_f1[ix + 2] = f2[ix + 2] + 2.0;
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    b_f1[ix + 2] = 0.0;
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    b_f1[ix + 2] = 0.0;
  }
  PetscCall(DMLocalToGlobalBegin(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(DMLocalToGlobalEnd(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(VecRestoreArray(blocal0,&b_f1_vec));
  PetscCall(DMRestoreLocalVector(dm0,&(blocal0)));

  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialGuess0(DM dm0, Vec xloc)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscScalar * x_f1_vec;

  PetscCall(VecGetArray(xloc,&x_f1_vec));
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));

  PetscScalar (* x_f1) = (PetscScalar (*)) x_f1_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    x_f1[ix + 2] = 4.0;
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    x_f1[ix + 2] = 8.0;
  }
  PetscCall(VecRestoreArray(xloc,&x_f1_vec));

  PetscFunctionReturn(0);
}

PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * f2_vec, const PetscScalar h_x, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2)
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  ctx0->h_x = h_x;
  ctx0->x_M = x_M;
  ctx0->x_ltkn0 = x_ltkn0;
  ctx0->x_ltkn1 = x_ltkn1;
  ctx0->x_m = x_m;
  ctx0->x_rtkn0 = x_rtkn0;
  ctx0->x_rtkn2 = x_rtkn2;
  ctx0->f2_vec = f2_vec;

  PetscFunctionReturn(0);
}
