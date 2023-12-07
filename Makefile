CFLAGS = -Wall -Wextra -O2
LDLIBS = -lm
CC = gcc
MPICC = mpicc

.PHONY: all

all: mpi-omp-parallel mpi-parallel parallel sequential

mpi-omp-parallel: mpi-omp-parallel.c
	$(MPICC) $(CFLAGS) -fopenmp -o $@ mpi-omp-parallel.c $(LDLIBS)

mpi-parallel: mpi-parallel.c
	$(MPICC) $(CFLAGS) -o $@ mpi-parallel.c $(LDLIBS)

parallel: parallel.c
	$(CC) $(CFLAGS) -fopenmp -o $@ parallel.c $(LDLIBS)

sequential: sequential.c
	$(CC) $(CFLAGS) -o $@ sequential.c $(LDLIBS)

mpirun: mpi-parallel
	mpiexec -n 4 ./mpi-parallel

clean:
	rm -f mpi-omp-parallel mpi-parallel parallel sequential *.o *.a deps.mk
