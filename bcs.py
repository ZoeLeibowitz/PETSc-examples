import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


PetscInitialize()

# Subdomains to implement BCs
class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, = dimensions
        return {x: ('left', 1)}

class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, = dimensions
        return {x: ('right', 1)}


sub1 = SubLeft()
sub2 = SubRight()

grid = Grid(
    shape=(6,), extent=(1.,), subdomains=(sub1,sub2,), dtype=np.float64
)

f1 = Function(name='f1', grid=grid, space_order=2)
f2 = Function(name='f2', grid=grid, space_order=2)

eqn = Eq(f1, f1.dx+f2+2., subdomain=grid.interior)

bcs = [EssentialBC(f1, np.float64(4.), subdomain=sub1)]
bcs += [EssentialBC(f1, np.float64(8.), subdomain=sub2)]

petsc = PETScSolve([eqn]+bcs, f1)

with switchconfig(language='petsc'):
    op = Operator(petsc)
    op.apply()

print(op.ccode)
