include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

fish: fish.o poissonfunctions.o
	-${CLINKER} -o fish fish.o poissonfunctions.o ${PETSC_LIB}
	${RM} fish.o poissonfunctions.o
