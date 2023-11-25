CFLAGS = -Wall -Wextra -O2
LDLIBS = -lm
CC = gcc
MPICC = mpicc

.PHONY: all

all: mpi-parallel parallel sequential

mpi-parallel: mpi-parallel.c
	$(MPICC) $(CFLAGS) -o $@ mpi-parallel.c $(LDLIBS)

parallel: parallel.c
	$(CC) $(CFLAGS) -fopenmp -o $@ parallel.c $(LDLIBS)

sequential: sequential.c
	$(CC) $(CFLAGS) -o $@ sequential.c $(LDLIBS)

mpirun: mpi-parallel
	mpiexec -n 2 ./mpi-parallel

clean:
	rm -f mpi-parallel parallel sequential *.o *.a *.bin deps.mk
