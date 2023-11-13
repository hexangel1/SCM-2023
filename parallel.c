#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define DER_XR(w, i, j) ((w->grid[i + 1][j] - w->grid[i][j]) / w->step_x)
#define DER_XL(w, i, j) DER_XR(w, i - 1, j)

#define DER_YR(w, i, j) ((w->grid[i][j + 1] - w->grid[i][j]) / w->step_y)
#define DER_YL(w, i, j) DER_YR(w, i, j - 1)

#define GET_Xi(w, i) (w->a1 + (i) * w->step_x)
#define GET_Yj(w, j) (w->a2 + (j) * w->step_y)

static int openmp_threads_num = -1;
static int save_result_iter = 0;
static int grid_size_n = 40;
static int grid_size_m = 40;
static double delta_w_stop = 0.000001;
static const char *output_file = NULL;

typedef double (*grid_initializer)(double x, double y);

struct grid_function {
    double **grid;
    size_t size_m, size_n;
    double step_x, step_y;
    double a1, b1;
    double a2, b2;
};

struct diff_scheme {
    struct grid_function *w, *aw, *w_next;
    struct grid_function *r, *ar, *b, *delta_w;
    struct grid_function *aij, *bij;
    double delta_stop;
};

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

double scalar_product(struct grid_function *gf1, struct grid_function *gf2)
{
    double result = 0.0;
    double weight = gf1->step_x * gf1->step_y;
    size_t i, j, m = gf1->size_m, n = gf1->size_n;
    assert(m == gf2->size_m && n == gf2->size_n);
#pragma omp parallel for collapse(2) reduction(+:result)
    for (i = 1; i < m; ++i) {
        for (j = 1; j < n; ++j) {
            result += weight * gf1->grid[i][j] * gf2->grid[i][j];
        }
    }
    return result;
}

double grid_norm(struct grid_function *gf)
{
    return sqrt(scalar_product(gf, gf));
}

double grid_squared_norm(struct grid_function *gf)
{
    return scalar_product(gf, gf);
}

void linear_comb(struct grid_function *result, double c1, double c2,
                 struct grid_function *gf1, struct grid_function *gf2)
{
    size_t i, j, m = gf1->size_m, n = gf1->size_n;
    assert(m == gf2->size_m && n == gf2->size_n);
#pragma omp parallel for collapse(2)
    for (i = 0; i <= m; ++i) {
        for (j = 0; j <= n; ++j) {
            result->grid[i][j] = c1 * gf1->grid[i][j] + c2 * gf2->grid[i][j];
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

void get_aij(struct grid_function *result, double eps)
{
    double h1 = result->step_x, h2 = result->step_y;
    size_t i, j, m = result->size_m, n = result->size_n;
    double inv_h2 = 1 / h2;
#pragma omp parallel for collapse(2)
    for (i = 1; i <= m; ++i) {
        for (j = 1; j <= n; ++j) {
            double x = GET_Xi(result, i), y = GET_Yj(result, j);
            double l = get_y_length(x - 0.5 * h1, y - 0.5 * h2, y + 0.5 * h2);
            result->grid[i][j] = inv_h2 * l + (1.0 - inv_h2 * l) / eps;
        }
    }
}

void get_bij(struct grid_function *result, double eps)
{
    double h1 = result->step_x, h2 = result->step_y;
    size_t i, j, m = result->size_m, n = result->size_n;
    double inv_h1 = 1 / h1;
#pragma omp parallel for collapse(2)
    for (i = 1; i <= m; ++i) {
        for (j = 1; j <= n; ++j) {
            double x = GET_Xi(result, i), y = GET_Yj(result, j);
            double l = get_x_length(y - 0.5 * h2, x - 0.5 * h1, x + 0.5 * h1);
            result->grid[i][j] = inv_h1 * l + (1.0 - inv_h1 * l) / eps;
        }
    }
}

void apply_diff_operator(struct grid_function *result, struct grid_function *gf,
                         struct grid_function *aij, struct grid_function *bij)
{
    double h1 = gf->step_x, h2 = gf->step_y;
    size_t i, j, m = gf->size_m, n = gf->size_n;
    assert(m == result->size_m && n == result->size_n);
#pragma omp parallel for collapse(2)
    for (i = 1; i < m; ++i) {
        for (j = 1; j < n; ++j) {
            double a1, a2, b1, b2;

            a2 = aij->grid[i + 1][j];
            a1 = aij->grid[i][j];
            b2 = bij->grid[i][j + 1];
            b1 = bij->grid[i][j];

            result->grid[i][j] = -(
                (a2 * DER_XR(gf, i, j) - a1 * DER_XL(gf, i, j)) / h1 +
                (b2 * DER_YR(gf, i, j) - b1 * DER_YL(gf, i, j)) / h2
            );
        }
    }
}

void init_grid(struct grid_function *gf, grid_initializer grinit)
{
    size_t i, j, m = gf->size_m, n = gf->size_n;
#pragma omp parallel for collapse(2)
    for (i = 0; i <= m; ++i) {
        for (j = 0; j <= n; ++j) {
            double x = GET_Xi(gf, i), y = GET_Yj(gf, j);
            gf->grid[i][j] = grinit(x, y);
        }
    }
}

struct grid_function *make_grid(double a1, double a2, double b1, double b2,
                                size_t m, size_t n)
{
    struct grid_function *gf;
    size_t i, j;
    gf = malloc(sizeof(*gf));
    gf->size_m = m;
    gf->size_n = n;
    gf->step_x = (b1 - a1) / (double)m;
    gf->step_y = (b2 - a2) / (double)n;
    gf->a1 = a1;
    gf->a2 = a2;
    gf->b1 = b1;
    gf->b2 = b2;
    gf->grid = malloc(sizeof(double*) * (m + 1));
    for (i = 0; i <= m; ++i) {
        gf->grid[i] = malloc(sizeof(double) * (n + 1));
        for (j = 0; j <= n; j++) {
            gf->grid[i][j] = 0.0;
        }
    }
    return gf;
}

void free_grid(struct grid_function *gf)
{
    size_t i, m = gf->size_m;
    for (i = 0; i <= m; ++i)
        free(gf->grid[i]);
    free(gf->grid);
    free(gf);
}

void debug_grid(struct grid_function *gf)
{
    size_t i, j, m = gf->size_m, n = gf->size_n;

    fprintf(stderr, "size_m = %ld, size_n = %ld\n", gf->size_m, gf->size_n);
    fprintf(stderr, "step_x = %lf, step_y = %lf\n", gf->step_x, gf->step_y);

    fprintf(stderr, "x is in [%lf, %lf]\n", gf->a1, gf->b1);
    fprintf(stderr, "y is in [%lf, %lf]\n", gf->a2, gf->b2);

    for (i = 0; i <= m; ++i) {
        fprintf(stderr, "[ ");
        for (j = 0; j <= n; ++j)
            fprintf(stderr, "%.6lf, ", gf->grid[i][j]);
        fprintf(stderr, "]\n");
    }
}

void export_tsv(struct grid_function *gf, const char *file)
{
    size_t i, j, m = gf->size_m, n = gf->size_n;
    FILE *fp = fopen(file, "w");
    if (!fp) {
        perror("export_tsv: fopen");
        return;
    }
    for (i = 0; i <= m; ++i) {
        for (j = 0; j <= n; ++j) {
            double x = GET_Xi(gf, i), y = GET_Yj(gf, j);
            fprintf(fp, "%lf\t%lf\t%lf\n", x, y, gf->grid[i][j]);
        }
    }
    fclose(fp);
}

struct diff_scheme *make_diff_scheme(double a1, double a2, double b1, double b2,
                                     size_t m, size_t n, double delta)
{
    struct diff_scheme *ds;
    double h_max, epsilon;
    ds = malloc(sizeof(*ds));
    ds->w = make_grid(a1, a2, b1, b2, m, n);
    ds->aw = make_grid(a1, a2, b1, b2, m, n);
    ds->w_next = make_grid(a1, a2, b1, b2, m, n);
    ds->r = make_grid(a1, a2, b1, b2, m, n);
    ds->ar = make_grid(a1, a2, b1, b2, m, n);
    ds->b = make_grid(a1, a2, b1, b2, m, n);
    ds->delta_w = make_grid(a1, a2, b1, b2, m, n);
    ds->aij = make_grid(a1, a2, b1, b2, m, n);
    ds->bij = make_grid(a1, a2, b1, b2, m, n);
    init_grid(ds->b, ones_function);
    h_max = MAX(ds->w->step_x, ds->w->step_y);
    epsilon = h_max * h_max;
    ds->delta_stop = delta;
    get_aij(ds->aij, epsilon);
    get_bij(ds->bij, epsilon);
    return ds;
}

void free_diff_scheme(struct diff_scheme *ds)
{
    free_grid(ds->w);
    free_grid(ds->aw);
    free_grid(ds->w_next);
    free_grid(ds->r);
    free_grid(ds->ar);
    free_grid(ds->b);
    free_grid(ds->delta_w);
    free_grid(ds->aij);
    free_grid(ds->bij);
    free(ds);
}

const char *get_checkpoint_file(unsigned long iteration_num)
{
    static char buff[128];
    snprintf(buff, sizeof(buff), "ckpt_%ld.tsv", iteration_num);
    return buff;
}

void run_scheme(double a1, double a2, double b1, double b2,
                size_t m, size_t n, double delta_stop)
{
    struct diff_scheme *ds = make_diff_scheme(a1, a2, b1, b2, m, n, delta_stop);
    struct grid_function *tmp;
    double sigma, iota, tau, delta;
    unsigned long iteration_num = 0;

#ifdef _OPENMP
    double start_time = omp_get_wtime();
#endif
    do {
        apply_diff_operator(ds->aw, ds->w, ds->aij, ds->bij);
        linear_comb(ds->r, 1.0, -1.0, ds->aw, ds->b);
        apply_diff_operator(ds->ar, ds->r, ds->aij, ds->bij);
        sigma = scalar_product(ds->ar, ds->r);
        iota = grid_squared_norm(ds->ar);
        tau = sigma / (iota + 0.000001);
        linear_comb(ds->w_next, 1.0, -tau, ds->w, ds->r);
        delta = grid_norm(ds->r);
        tmp = ds->w;
        ds->w = ds->w_next;
        ds->w_next = tmp;
        if (save_result_iter && iteration_num % save_result_iter == 0)
            export_tsv(ds->w, get_checkpoint_file(iteration_num));
        if (iteration_num % 1000 == 0)
            fprintf(stderr, "delta = %lf\n", delta);
        ++iteration_num;
    } while (delta > ds->delta_stop);
#ifdef _OPENMP
    fprintf(stderr, "time: %lf\n", omp_get_wtime() - start_time);
#endif
    fprintf(stderr, "iterations: %ld\n", iteration_num);

    if (output_file)
        export_tsv(ds->w, output_file);
    free_diff_scheme(ds);
}

int get_command_line_options(int argc, char **argv)
{
    int opt, retval = 0;
    extern char *optarg;
    extern int optopt;
    while ((opt = getopt(argc, argv, ":hw:c:m:n:d:f:")) != -1) {
        switch (opt) {
        case 'h':
            retval = -1;
            break;
        case 'w':
            openmp_threads_num = atoi(optarg);
            break;
        case 'c':
            save_result_iter = atoi(optarg);
            break;
        case 'm':
            grid_size_m = atoi(optarg);
            break;
        case 'n':
            grid_size_n = atoi(optarg);
            break;
        case 'd':
            delta_w_stop = atof(optarg);
            break;
        case 'f':
            output_file = optarg;
            break;
        case ':':
            fprintf(stderr, "Opt -%c require an operand\n", optopt);
            retval = -1;
            break;
        case '?':
            fprintf(stderr, "Unrecognized option: -%c\n", optopt);
            retval = -1;
            break;
        }
    }
    return retval;
}

int get_num_threads(void)
{
    int num_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
#endif
    return num_threads;
}

int main(int argc, char **argv)
{
    int res;
    const char *usage =
        "Usage: %s -w <threads> -m <M> -n <N> -d <delta> -f <filename>\n";
    res = get_command_line_options(argc, argv);
    if (res == -1) {
        fprintf(stderr, usage, argv[0]);
        exit(1);
    }
#ifdef _OPENMP
    if (openmp_threads_num != -1)
        omp_set_num_threads(openmp_threads_num);
    fprintf(stderr, "OpenMP supported\n");
    fprintf(stderr, "Running on %d threads\n", get_num_threads());
#else
    fprintf(stderr, "OpenMP not supported\n");
#endif
    assert(save_result_iter >= 0);
    assert(grid_size_m > 1 && grid_size_n > 1);
    assert(delta_w_stop > 0.0);
    fprintf(stderr, "M = %d, N = %d\n", grid_size_m, grid_size_n);
    run_scheme(-1.0, -0.5, 1.0, 0.5, grid_size_m, grid_size_n, delta_w_stop);
    return 0;
}

