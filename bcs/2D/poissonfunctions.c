// ED BUELER - Petsc4pdes

#include <petsc.h>
#include "poissonfunctions.h"


//STARTFORM2DFUNCTION
PetscErrorCode Poisson2DFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                      PetscReal **aF, PoissonCtx *user) {
    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, darea, scx, scy, scdiag, x, y,
               ue, uw, un, us;
    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    darea = hx * hy;
    scx = user->cx * hy / hx;
    scy = user->cy * hx / hy;
    scdiag = 2.0 * (scx + scy);    // diagonal scaling
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = xymin[1] + j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = xymin[0] + i * hx;
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                aF[j][i] = au[j][i] - user->g_bdry(x,y,0.0,user);
                aF[j][i] *= scdiag;
            } else {
                ue = (i+1 == info->mx-1) ? user->g_bdry(x+hx,y,0.0,user)
                                         : au[j][i+1];
                uw = (i-1 == 0)          ? user->g_bdry(x-hx,y,0.0,user)
                                         : au[j][i-1];
                un = (j+1 == info->my-1) ? user->g_bdry(x,y+hy,0.0,user)
                                         : au[j+1][i];
                us = (j-1 == 0)          ? user->g_bdry(x,y-hy,0.0,user)
                                         : au[j-1][i];
                aF[j][i] = scdiag * au[j][i]
                           - scx * (uw + ue) - scy * (us + un)
                           - darea * user->f_rhs(x,y,0.0,user);
            }
        }
    }
    PetscCall(PetscLogFlops(11.0*info->xm*info->ym));
    return 0;
}
//ENDFORM2DFUNCTION

PetscErrorCode Poisson2DJacobianLocal(DMDALocalInfo *info, PetscScalar **au,
                                      Mat J, Mat Jpre, PoissonCtx *user) {
    PetscReal   xymin[2], xymax[2], hx, hy, scx, scy, scdiag, v[5];
    PetscInt    i,j,ncols;
    MatStencil  col[5],row;

    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    scx = user->cx * hy / hx;
    scy = user->cy * hx / hy;
    scdiag = 2.0 * (scx + scy);
    for (j = info->ys; j < info->ys+info->ym; j++) {
        row.j = j;
        col[0].j = j;
        for (i = info->xs; i < info->xs+info->xm; i++) {
            row.i = i;
            col[0].i = i;
            ncols = 1;
            v[0] = scdiag;
            if (i>0 && i<info->mx-1 && j>0 && j<info->my-1) {
                if (i-1 > 0) {
                    col[ncols].j = j;    col[ncols].i = i-1;  v[ncols++] = - scx;  }
                if (i+1 < info->mx-1) {
                    col[ncols].j = j;    col[ncols].i = i+1;  v[ncols++] = - scx;  }
                if (j-1 > 0) {
                    col[ncols].j = j-1;  col[ncols].i = i;    v[ncols++] = - scy;  }
                if (j+1 < info->my-1) {
                    col[ncols].j = j+1;  col[ncols].i = i;    v[ncols++] = - scy;  }
            }
            PetscCall(MatSetValuesStencil(Jpre,1,&row,ncols,col,v,INSERT_VALUES));
        }
    }

    PetscCall(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
    if (J != Jpre) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}

PetscErrorCode InitialState(DM da, PetscBool gbdry,
                            Vec u, PoissonCtx *user) {
    DMDALocalInfo  info;
    
    // Initial zeros
    PetscCall(VecSet(u,0.0));

    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, x, y, **au;
    PetscCall(DMDAVecGetArray(da, u, &au));
    PetscCall(DMGetBoundingBox(da,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (info.mx - 1);
    hy = (xymax[1] - xymin[1]) / (info.my - 1);
    for (j = info.ys; j < info.ys + info.ym; j++) {
        y = xymin[1] + j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++) {
            if (i==0 || i==info.mx-1 || j==0 || j==info.my-1) {
                x = xymin[0] + i * hx;
                au[j][i] = user->g_bdry(x,y,0.0,user);
            }
        }
    }
    PetscCall(DMDAVecRestoreArray(da, u, &au));
    return 0;
}
