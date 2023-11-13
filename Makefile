CFLAGS = -Wall -Wextra -O2
LDLIBS = -lm
CC = gcc

.PHONY: all

all: parallel sequential

parallel: parallel.c
	$(CC) $(CFLAGS) -fopenmp -o $@ parallel.c $(LDLIBS)

sequential: sequential.c
	$(CC) $(CFLAGS) -o $@ sequential.c $(LDLIBS)

clean:
	rm -f parallel sequential *.o *.a *.bin deps.mk
