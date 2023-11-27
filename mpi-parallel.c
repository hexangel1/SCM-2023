#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define IF_MASTER if (proc_id == 0) {
#define FI_MASTER }

#define LOGMSG(msg, ...) \
    do {\
        fprintf(stderr, "worker[%d]: " msg, proc_id, ##__VA_ARGS__);\
    } while(0)

#define IS_POWER_OF_2(n) (n && !(n & (n - 1)))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

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

#define LOCAL_IDX(i, j) ((i) * local_grid_size_n + (j))
#define GLOBAL_IDX(i, j) ((i) * grid_size_n + (j))
#define DOMAIN_IDX(i, j) \
    (DOMAIN_ID(i, j) * domain_size + LOCAL_IDX(LOCAL_i(i), LOCAL_j(j)))

#define DER_XR(w, i, j) \
    ((w[DOMAIN_IDX(i + 1, j)] - w[DOMAIN_IDX(i, j)]) / grid_step_x)
#define DER_XL(w, i, j) DER_XR(w, i - 1, j)

#define DER_YR(w, i, j) \
    ((w[DOMAIN_IDX(i, j + 1)] - w[DOMAIN_IDX(i, j)]) / grid_step_y)
#define DER_YL(w, i, j) DER_YR(w, i, j - 1)

#define GET_Xi(i) (point_a1 + (i) * grid_step_x)
#define GET_Yj(j) (point_a2 + (j) * grid_step_y)

struct diff_scheme {
    double *w, *aw, *w_next;
    double *r, *ar, *b, *delta_w;
    double *aij, *bij;
    double *global_b, *global_w, *global_r;
    double delta_stop;
};

typedef double (*grid_initializer)(double x, double y);

static const int grid_size_n = 40;
static const int grid_size_m = 40;
static const double delta_stop = 0.01;
static const double point_a1 = -1.0;
static const double point_b1 = 1.0;
static const double point_a2 = -0.5;
static const double point_b2 = 0.5;
static const double grid_step_x = (point_b1 - point_a1) / (grid_size_m - 1);
static const double grid_step_y = (point_b2 - point_a2) / (grid_size_n - 1);
static const double grid_step_max = MAX(grid_step_x, grid_step_y);
static const double epsilon = grid_step_max * grid_step_max;

static int proc_id, proc_number;
static int local_grid_size_n;
static int local_grid_size_m;
static int x_domain_amount;
static int y_domain_amount;
static int domain_size;

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
    domain_size = local_grid_size_m * local_grid_size_n;
}

int inside_ellipse(double x, double y)
{
    return x * x + 4.0 * y * y < 1.0;
}

double get_y_intersection(double x)
{
    double res = (1.0 - x * x) / 4.0;
    return res >= 0.0 ? sqrt(res) : -1.0;
}

double get_x_intersection(double y)
{
    double res = 1.0 - 4.0 * y * y;
    return res >= 0.0 ? sqrt(res) : -1.0;
}

double ones_function(double x, double y)
{
    return inside_ellipse(x, y) ? 1.0 : 0.0;
}

double scalar_product(double *grid1, double *grid2)
{
    double result = 0.0;
    double weight = grid_step_x * grid_step_y;
    int i, j;
    for (i = 0; i < local_grid_size_m; ++i) {
        for (j = 0; j < local_grid_size_n; ++j) {
            result += weight * grid1[LOCAL_IDX(i, j)] * grid2[LOCAL_IDX(i, j)];
        }
    }
    return result;
}

double grid_norm(double *grid)
{
    return sqrt(scalar_product(grid, grid));
}

double grid_squared_norm(double *grid)
{
    return scalar_product(grid, grid);
}

void linear_comb(double *result, double c1, double c2,
                 double *grid1, double *grid2)
{
    int i, j;
    for (i = 0; i < local_grid_size_m; ++i) {
        for (j = 0; j < local_grid_size_n; ++j) {
            result[LOCAL_IDX(i, j)] = c1 * grid1[LOCAL_IDX(i, j)] +
                                      c2 * grid2[LOCAL_IDX(i, j)];
        }
    }
}

double get_y_length(double x, double y1, double y2)
{
    double y_max = get_y_intersection(x);
    double y_min = -y_max;
    if (y_max < 0.0)
        return inside_ellipse(x, (y1 + y2) / 2.0) ? y2 - y1 : 0.0;
    if (y1 > y_max || y2 < y_min)
        return 0.0;
    return MIN(y2, y_max) - MAX(y1, y_min);
}

double get_x_length(double y, double x1, double x2)
{
    double x_max = get_x_intersection(y);
    double x_min = -x_max;
    if (x_max < 0.0)
        return inside_ellipse((x1 + x2) / 2.0, y) ? x2 - x1 : 0.0;
    if (x1 > x_max || x2 < x_min)
        return 0.0;
    return MIN(x2, x_max) - MAX(x1, x_min);
}

void get_aij(double *result, double eps)
{
    double h1 = grid_step_x, h2 = grid_step_y;
    double inv_h2 = 1 / h2;
    int i, j;
    for (i = 1; i < grid_size_m ; ++i) {
        for (j = 1; j < grid_size_n; ++j) {
            double x = GET_Xi(i), y = GET_Yj(j);
            double l = get_y_length(x - 0.5 * h1, y - 0.5 * h2, y + 0.5 * h2);
            result[GLOBAL_IDX(i, j)] = inv_h2 * l + (1.0 - inv_h2 * l) / eps;
        }
    }
}

void get_bij(double *result, double eps)
{
    double h1 = grid_step_x, h2 = grid_step_y;
    double inv_h1 = 1 / h1;
    int i, j;
    for (i = 1; i < grid_size_m; ++i) {
        for (j = 1; j < grid_size_n; ++j) {
            double x = GET_Xi(i), y = GET_Yj(j);
            double l = get_x_length(y - 0.5 * h2, x - 0.5 * h1, x + 0.5 * h1);
            result[GLOBAL_IDX(i, j)] = inv_h1 * l + (1.0 - inv_h1 * l) / eps;
        }
    }
}

double *make_grid(size_t m, size_t n)
{
    size_t i, j;
    double *grid = malloc(sizeof(double) * n * m);
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            grid[i * n + j] = 0.0;
        }
    }
    return grid;
}

void init_grid(double *grid, grid_initializer grinit)
{
    int i, j;
    for (i = 0; i < grid_size_m; ++i) {
        for (j = 0; j < grid_size_n; ++j) {
            double x = GET_Xi(i), y = GET_Yj(j);
            grid[DOMAIN_IDX(i, j)] = grinit(x, y);
        }
    }
}

void apply_diff_operator(double *aw, double *w, double *aij, double *bij)
{
    double h1 = grid_step_x, h2 = grid_step_y;
    int i, j;
    for (i = 0; i < local_grid_size_m; ++i) {
        for (j = 0; j < local_grid_size_n; ++j) {
            double a1, a2, b1, b2;
            int global_i = MY_GLOBAL_i(i), global_j = MY_GLOBAL_j(j);
            if (global_i == 0 || global_i == grid_size_m - 1 ||
                global_j == 0 || global_j == grid_size_n - 1)
                continue;

            a2 = aij[GLOBAL_IDX(global_i + 1, global_j)];
            a1 = aij[GLOBAL_IDX(global_i, global_j)];
            b2 = bij[GLOBAL_IDX(global_i, global_j + 1)];
            b1 = bij[GLOBAL_IDX(global_i, global_j)];

            aw[LOCAL_IDX(i, j)] = -(
                (a2 * DER_XR(w, global_i, global_j) -
                 a1 * DER_XL(w, global_i, global_j)) / h1 +
                (b2 * DER_YR(w, global_i, global_j) -
                 b1 * DER_YL(w, global_i, global_j)) / h2
            );
        }
    }
}

struct diff_scheme *make_diff_scheme(void)
{
    struct diff_scheme *ds;
    ds = malloc(sizeof(*ds));

    ds->global_w = make_grid(grid_size_m, grid_size_n);
    ds->w = make_grid(local_grid_size_m, local_grid_size_n);
    ds->aw = make_grid(local_grid_size_m, local_grid_size_n);
    ds->w_next = make_grid(local_grid_size_m, local_grid_size_n);
    ds->global_r = make_grid(grid_size_m, grid_size_n);
    ds->r = make_grid(local_grid_size_m, local_grid_size_n);
    ds->ar = make_grid(local_grid_size_m, local_grid_size_n);

    ds->aij = make_grid(grid_size_m, grid_size_n);
    ds->bij = make_grid(grid_size_m, grid_size_n);
    ds->b = make_grid(local_grid_size_m, local_grid_size_n);

    ds->global_b = NULL;
IF_MASTER
    ds->global_b = make_grid(grid_size_m, grid_size_n);
    init_grid(ds->global_b, ones_function);
    get_aij(ds->aij, epsilon);
    get_bij(ds->bij, epsilon);
FI_MASTER

    return ds;
}

void free_diff_scheme(struct diff_scheme *ds)
{
    free(ds->w);
    free(ds->aw);
    free(ds->w_next);
    free(ds->global_r);
    free(ds->global_w);
    free(ds->global_b);
    free(ds->b);
    free(ds->r);
    free(ds->ar);
    free(ds->aij);
    free(ds->bij);
    free(ds);
}

void export_tsv(double *result, const char *file)
{
    int i, j;
    FILE *fp = fopen(file, "w");
    if (!fp) {
        perror("export_tsv: fopen");
        return;
    }
    for (i = 0; i < grid_size_m; ++i) {
        for (j = 0; j < grid_size_n; ++j) {
            double x = GET_Xi(i), y = GET_Yj(j);
            fprintf(fp, "%lf\t%lf\t%lf\n", x, y, result[DOMAIN_IDX(i, j)]);
        }
    }
    fclose(fp);
}

void process_main(void)
{
    double delta, tau;
    double *tmp, local_vars[3], global_vars[3];
    size_t size = grid_size_m * grid_size_n;
    struct diff_scheme *ds = make_diff_scheme();
    unsigned long iteration_num = 0;

    MPI_Scatter(ds->global_b, domain_size, MPI_DOUBLE, ds->b, domain_size,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(ds->aij, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ds->bij, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    do {

        apply_diff_operator(ds->aw, ds->global_w, ds->aij, ds->bij);
        linear_comb(ds->r, 1.0, -1.0, ds->aw, ds->b);
        MPI_Allgather(ds->r, domain_size, MPI_DOUBLE, ds->global_r, domain_size,
                      MPI_DOUBLE, MPI_COMM_WORLD);

        apply_diff_operator(ds->ar, ds->global_r, ds->aij, ds->bij);

        local_vars[0] = scalar_product(ds->ar, ds->r);
        local_vars[1] = grid_squared_norm(ds->ar);
        local_vars[2] = grid_squared_norm(ds->r);

        MPI_Allreduce(local_vars, global_vars, 3,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        delta = sqrt(global_vars[2]);

        tau = global_vars[0] / (global_vars[1] + 0.000001);

        linear_comb(ds->w_next, 1.0, -tau, ds->w, ds->r);

        MPI_Allgather(ds->w_next, domain_size, MPI_DOUBLE, ds->global_w,
                      domain_size, MPI_DOUBLE, MPI_COMM_WORLD);

        tmp = ds->w;
        ds->w = ds->w_next;
        ds->w_next = tmp;

IF_MASTER
        if (iteration_num % 1000 == 0)
            LOGMSG("delta = %lf\n", delta);
        ++iteration_num;
FI_MASTER
    } while (delta > delta_stop);

IF_MASTER
    export_tsv(ds->global_w, "result.tsv");
FI_MASTER
    free_diff_scheme(ds);
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

    process_main();
    MPI_Barrier(MPI_COMM_WORLD);

IF_MASTER
    LOGMSG("Time: %lf\n", MPI_Wtime() - start_time);
FI_MASTER

    MPI_Finalize();
    return 0;
}

