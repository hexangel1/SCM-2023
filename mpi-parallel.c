#include <stdio.h>
#include <stdarg.h>
#include <mpi.h>

#define IF_MASTER if (proc_id == 0) {
#define FI_MASTER }

#define LOGMSG(msg, ...) \
    do {\
        fprintf(stderr, "worker[%d]: " msg, proc_id, ##__VA_ARGS__);\
    } while(0)

#define IS_POWER_OF_2(n) (n && !(n & (n - 1)))

#define LOCAL_i(i) ((i) % local_grid_size_m)
#define LOCAL_j(j) ((j) % local_grid_size_n)

#define GLOBAL_i(i, domain_id) \
    (((domain_id) / y_domain_amount) * local_grid_size_m + (i)) 
#define GLOBAL_j(j, domain_id) \
    (((domain_id) % y_domain_amount) * local_grid_size_n + (j))

#define MY_GLOBAL_i(i) GLOBAL_i(i, proc_id)
#define MY_GLOBAL_j(j) GLOBAL_j(j, proc_id)

#define DOMAIN_ID(i, j) \
    (((i) / local_grid_size_m) * (y_domain_amount) + (j) / local_grid_size_n)

static int proc_id, proc_number;
static int grid_size_n = 40;
static int grid_size_m = 40;
static int local_grid_size_n;
static int local_grid_size_m;
static int x_domain_amount;
static int y_domain_amount;

void init_local_grid_sizes(void)
{
    int turn = 0, proc_count = proc_number / 2;
    local_grid_size_n = grid_size_n;
    local_grid_size_m = grid_size_m;
    while (proc_count) {
        if (turn)
            local_grid_size_n /= 2;
        else
            local_grid_size_m /= 2;
        turn = !turn;
        proc_count /= 2;
    }
    x_domain_amount = grid_size_m / local_grid_size_m;
    y_domain_amount = grid_size_n / local_grid_size_n;
}

void test_proc(int tested_id)
{
    int i, j;
    if (tested_id == proc_id) {
        for (i = 0; i < local_grid_size_m; i++) {
            for (j = 0; j < local_grid_size_n; j++) {
                int global_i = MY_GLOBAL_i(i), global_j = MY_GLOBAL_j(j);
                LOGMSG("(%d, %d)\n", global_i, global_j);
            }
        }
    }
}

void test_proc2(void)
{
    int i, j;
    IF_MASTER
    for (i = 0; i < grid_size_n; i++) {
        for (j = 0; j < grid_size_m; j++) {
            int i_loc = LOCAL_i(i), j_loc = LOCAL_j(j);
            int domain_id = DOMAIN_ID(i, j);
//            LOGMSG("(%d, %d) -> (%d, %d, %d)\n", i, j, i_loc, j_loc, domain_id);
            if (i != GLOBAL_i(i_loc, domain_id) ||
                j != GLOBAL_j(j_loc, domain_id))
                LOGMSG("TEST FAILED!!!!\n");
        }
    }
    FI_MASTER
}

void run_tests(void)
{   
    test_proc2();
    MPI_Barrier(MPI_COMM_WORLD);
    test_proc(0);
    MPI_Barrier(MPI_COMM_WORLD);
    test_proc(1);
    MPI_Barrier(MPI_COMM_WORLD);
    test_proc(2);
    MPI_Barrier(MPI_COMM_WORLD);
    test_proc(3);
}

void process_main(void)
{

}

int main(int argc, char **argv)
{
    double start_time = 0.0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_number);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    IF_MASTER
    if (!IS_POWER_OF_2(proc_number)) {
        LOGMSG("Error: process number must be a power of 2\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    FI_MASTER

    init_local_grid_sizes();

    IF_MASTER
        LOGMSG("Running on %d processes\n", proc_number);
        LOGMSG("Local grid sizes = (%d, %d)\n",
            local_grid_size_m, local_grid_size_n);
        start_time = MPI_Wtime();
    FI_MASTER

    MPI_Barrier(MPI_COMM_WORLD);
    process_main();
    MPI_Barrier(MPI_COMM_WORLD);
    IF_MASTER
        LOGMSG("Time: %lf\n", MPI_Wtime() - start_time);
    FI_MASTER
    MPI_Finalize();
    return 0;
}

