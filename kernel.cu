#include <cuda_runtime.h>
//#include "cublas_v2.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

const double offdiagonal_threshold_cpu = 1.0E-9;

int num_blocks = 1;
int num_threads = 1 * 32;

double* alloc_matrix(int m);

double* alloc_matrix(int m)
{
    return new double[m * m];
}

double* alloc_vector(int m)
{
    return new double[m];
}

double* alloc_matrix_gpu(int m)
{
    double* res = 0;
    cudaMalloc(&res, m * m * sizeof(double));
    return res;
}

double* alloc_vector_gpu(int m)
{
    double* res = 0;
    cudaMalloc(&res, m * sizeof(double));
    return res;
}


template <
    class result_t = std::chrono::milliseconds,
    class clock_t = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

double sign(double num)
{
    if (num > 0) return 1;
    else if (num < 0) return -1;
    else return 0;
}

void print_matrix(double* matrix, int m)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            printf("%.17lf\t ", matrix[IDX2C(i, j, m)]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vector(double* vector, int m)
{
    for (int j = 0; j < m; j++)
    {
        printf("%lf ", vector[j]);
    }
    printf("\n\n");
}



double get_R(int x, int y, int j, int k, double cos_theta, double sin_theta)
{
    if (x == j && y == j)
        return cos_theta;
    else if (x == k && y == j)
        return -sin_theta;
    else if (x == j && y == k)
        return sin_theta;
    else if (x == k && y == k)
        return cos_theta;
    else if (x == y)
        return 1;
    else
        return 0;
}

double get_RT(int x, int y, int j, int k, double cos_theta, double sin_theta)
{
    if (y == j && x == j)
        return cos_theta;
    else if (y == k && x == j)
        return -sin_theta;
    else if (y == j && x == k)
        return sin_theta;
    else if (y == k && x == k)
        return cos_theta;
    else if (y == x)
        return 1;
    else
        return 0;
}

__global__ void fill_as_identity_gpu(double* matrix, int m)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m * m; i += blockDim.x * gridDim.x)
        matrix[i] = 0.0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x)
        matrix[IDX2C(i, i, m)] = 1.0;
}

void fill_as_identity(double* matrix, int m)
{
    dim3 block_dim(num_threads, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);
    fill_as_identity_gpu << <grid_dim, block_dim >> > (matrix, m);
}


void fill_as_r(double* matrix, int m, int j, int k, double c, double s)
{
    fill_as_identity(matrix, m);

    double jj = get_R(j, j, j, k, c, s);
    double jk = get_R(j, k, j, k, c, s);
    double kj = get_R(k, j, j, k, c, s);
    double kk = get_R(k, k, j, k, c, s);

    cudaMemcpy(&matrix[IDX2C(j, j, m)], &jj, sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(&matrix[IDX2C(j, k, m)], &jk, sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(&matrix[IDX2C(k, j, m)], &kj, sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(&matrix[IDX2C(k, k, m)], &kk, sizeof(double), cudaMemcpyDefault);
}

void fill_as_rt(double* matrix, int m, int j, int k, double c, double s)
{
    fill_as_identity(matrix, m);

    double jj = get_RT(j, j, j, k, c, s);
    double jk = get_RT(j, k, j, k, c, s);
    double kj = get_RT(k, j, j, k, c, s);
    double kk = get_RT(k, k, j, k, c, s);

    cudaMemcpy(&matrix[IDX2C(j, j, m)], &jj, sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(&matrix[IDX2C(j, k, m)], &jk, sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(&matrix[IDX2C(k, j, m)], &kj, sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(&matrix[IDX2C(k, k, m)], &kk, sizeof(double), cudaMemcpyDefault);
}

__global__ void getSums(double* matrix, double* s, int m)
{
    for (int column = blockIdx.x * blockDim.x + threadIdx.x; column < m; column += blockDim.x * gridDim.x)
    {
        double sum = 0;
        for (int i = 0; i < m; i++)
        {
            sum += matrix[IDX2C(column, i, m)];
        }
        s[column] = sum;
    }
}

__device__ void multiply_matrices_only_jk_gpu_row(double* first, double* second, double* res, int size, int row, int start_index, int stride)
{
    for (int b_ = start_index; b_ < size; b_ += stride)
    {
        double sum = 0;
        for (int c_ = 0; c_ < size; c_++)
            sum += first[IDX2C(row, c_, size)] * second[IDX2C(c_, b_, size)];
        res[IDX2C(row, b_, size)] = sum;
    }
}

__device__ void multiply_matrices_only_jk_gpu_column(double* first, double* second, double* res, int size, int ignore_row_1, int ignore_row_2, int column, int start_index, int stride)
{
    for (int a_ = start_index; a_ < size; a_ += stride)
    {
        if (a_ != ignore_row_1 && a_ != ignore_row_2)
        {
            double sum = 0;
            for (int c_ = 0; c_ < size; c_++)
                sum += first[IDX2C(a_, c_, size)] * second[IDX2C(c_, column, size)];
            res[IDX2C(a_, column, size)] = sum;
        }
    }
}

__global__ void multiply_matrices_only_jk_gpu(double* first, double* second, double* res, int size, int j, int k)
{

    int subblockSize = blockDim.x * gridDim.x / 4;
    int operation = (blockIdx.x * blockDim.x + threadIdx.x) / subblockSize;
    int start_index = (blockIdx.x * blockDim.x + threadIdx.x) % subblockSize;
    switch (operation)
    {
    case 0:
        multiply_matrices_only_jk_gpu_row(first, second, res, size, j, start_index, subblockSize);
        break;
    case 1:
        multiply_matrices_only_jk_gpu_row(first, second, res, size, k, start_index, subblockSize);
        break;
    case 2:
        multiply_matrices_only_jk_gpu_column(first, second, res, size, j, k, j, start_index, subblockSize);
        break;
    case 3:
        multiply_matrices_only_jk_gpu_column(first, second, res, size, j, k, k, start_index, subblockSize);
        break;
#ifdef _DEBUG
    default:
        printf("multiply_matrices_only_jk_gpu expected operation in range 1-4, got %d", operation);
        break;
#endif
    }
}

void multiply_matrices_only_jk(double* first, double* second, double* res, int size, int j, int k, bool keepFirst)
{
    // копируем неединичную матрицу в результат
    if (keepFirst)
        cudaMemcpy(res, first, size * size * sizeof(double), cudaMemcpyDefault);
    else
        cudaMemcpy(res, second, size * size * sizeof(double), cudaMemcpyDefault);


    dim3 block_dim(num_threads, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);
    multiply_matrices_only_jk_gpu << <grid_dim, block_dim >> > (first, second, res, size, j, k);

}

double get_t(double* A, int m, int j, int k)
{
    double jj, kk, jk;
    cudaMemcpy(&jj, &A[IDX2C(j, j, m)], sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(&kk, &A[IDX2C(k, k, m)], sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(&jk, &A[IDX2C(j, k, m)], sizeof(double), cudaMemcpyDefault);

    double tau = (jj - kk) / (2 * jk);
    double t = sign(tau) / (abs(tau) + sqrt(1 + tau * tau));
    return t;
}

void jacobi_rotation(double* A, int m, int j, int k,
    double* buff1, double* buff2, double* buff3, double* buff_r, double* buff_rt, double*& U, double*& V)
{
    double a_jk;
    cudaMemcpy(&a_jk, &A[IDX2C(j, k, m)], sizeof(double), cudaMemcpyDefault);
    if (fabs(a_jk) > offdiagonal_threshold_cpu)
    {
        double t = get_t(A, m, j, k);
        double c = 1 / sqrt(1 + t * t);
        double s = c * t;

        fill_as_r(buff_r, m, j, k, c, s);
        fill_as_rt(buff_rt, m, j, k, c, s);

        // A <= R * A * RT
        // V <= R * V
        // U <= U * RT

        // buff1 = R * A
        multiply_matrices_only_jk(buff_r, A, buff1, m, j, k, false);

        // buff2 = R * V
        multiply_matrices_only_jk(buff_r, V, buff2, m, j, k, false);

        // buff3 = U * RT
        multiply_matrices_only_jk(U, buff_rt, buff3, m, j, k, true);

        cudaDeviceSynchronize();

        // A = buff1 * RT
        multiply_matrices_only_jk(buff1, buff_rt, A, m, j, k, true);

        // V = buff2
        cudaMemcpy(V, buff2, m * m * sizeof(double), cudaMemcpyDefault);

        // U = buff3
        cudaMemcpy(U, buff3, m * m * sizeof(double), cudaMemcpyDefault);

        cudaDeviceSynchronize();
    }
}

__constant__ const double offdiagonal_threshold = 1.0E-9;

__global__ void is_converged_gpu(double* A, int m, bool* is_converged)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x)
    {
        if (fabs(A[i]) > offdiagonal_threshold && (i % m != i / m))
        {
            *is_converged = false;
            return;
        }
    }
}

bool is_converged(double* A, int m)
{
    bool is_converged = true;
    bool* is_converged_d;
    cudaMalloc(&is_converged_d, sizeof(bool));
    cudaMemcpy(is_converged_d, &is_converged, sizeof(is_converged), cudaMemcpyDefault);
    is_converged_gpu << <1, num_threads >> > (A, m, is_converged_d);
    cudaMemcpy(&is_converged, is_converged_d, sizeof(is_converged), cudaMemcpyDefault);
    cudaFree(is_converged_d);
    return is_converged;
}

void jacobi_svd_gpu(double* A, int m, double* s, double* u, double* vt, double* buff1, double* buff2, double* buff3, double* buff4, double* buff5)
{
    fill_as_identity(vt, m);
    fill_as_identity(u, m);

    while (!is_converged(A, m))
    {
        for (int j = 0; j < m - 1; j++)
            for (int k = j + 1; k < m; k++)
                jacobi_rotation(A, m, j, k, buff1, buff2, buff3, buff4, buff5, u, vt);
    }

    dim3 block_dim(num_threads, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    getSums << <grid_dim, block_dim >> > (A, s, m);
}


// принимает матрицу A, разлагает на u, s и vt
void jacobi_svd(double* A, int m, double* s, double* u, double* vt)
{
    double* cuda_A = alloc_matrix_gpu(m);
    double* cuda_s = alloc_vector_gpu(m);
    double* cuda_u = alloc_matrix_gpu(m);
    double* cuda_v = alloc_matrix_gpu(m);

    double* buff1 = alloc_matrix_gpu(m);
    double* buff2 = alloc_matrix_gpu(m);
    double* buff3 = alloc_matrix_gpu(m);
    double* buff4 = alloc_matrix_gpu(m);
    double* buff5 = alloc_matrix_gpu(m);

    cudaMemcpy(cuda_A, A, m * m * sizeof(double), cudaMemcpyDefault);

    jacobi_svd_gpu(cuda_A, m, cuda_s, cuda_u, cuda_v, buff1, buff2, buff3, buff4, buff5);
    cudaDeviceSynchronize();

    cudaMemcpy(s, cuda_s, m * sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(u, cuda_u, m * m * sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(vt, cuda_v, m * m * sizeof(double), cudaMemcpyDefault);


    cudaFree(buff1);
    cudaFree(buff2);
    cudaFree(buff3);
    cudaFree(buff4);
    cudaFree(buff5);

    cudaFree(cuda_A);
    cudaFree(cuda_s);
    cudaFree(cuda_u);
    cudaFree(cuda_v);
}

void fill_matrix(double* a, int m)
{
    for (int i = 0; i < m; i++)
        for (int j = i; j < m; j++)
        {
            a[IDX2C(i, j, m)] = i + j + 1;
            a[IDX2C(j, i, m)] = i + j + 1;
        }
}

bool is_symmetric(double* matrix, int m)
{
    for (int i = 0; i < m - 1; i++)
        for (int j = i + 1; j < m; j++)
        {
            if (matrix[IDX2C(i, j, m)] != matrix[IDX2C(j, i, m)])
                return false;
        }
    return true;
}

double getMean(std::vector<long long> nums)
{
    double mean = 0;
    for (int i = 0; i < nums.size(); i++)
        mean += nums[i];
    mean /= nums.size();
    return mean;
}

double getMarginOfError(std::vector<long long> nums)
{
#ifdef _DEBUG
    for (int i = 0; i < nums.size(); i++)
        printf("%d\n", nums[i]);
#endif // _DEBUG

    double mean = getMean(nums);

    double squaredDiffSum = 0;
    for (int i = 0; i < nums.size(); i++)
        squaredDiffSum += (nums[i] - mean) * (nums[i] - mean);

    double stdDeviation = sqrt(squaredDiffSum / nums.size());

    double c;
    int degreesOfFreedom = nums.size() - 1;
    switch (degreesOfFreedom) // не хочу заморачиваться с расчётом, поэтому взял пару значений из таблицы (доверительная вероятность 0.99)
    {
    case 4:
        c = 4.604;
        break;
    case 15:
        c = 2.947;
        break;
    default:
        throw new std::exception("getMeanAndDeviation: wrong sample size");
    }

    double margin = c * stdDeviation / sqrt(nums.size() * 1.0);

    return margin;
}

struct test_thread_config
{
    int thread_count;
    int block_count;
};

int main()
{
#ifdef _DEBUG
    std::cout << "DEBUG VERSION; TIME DATA INCORRECT\n";
#endif
    const int start_m = 25;
    const int max_m = 100;
    const int step_m = 25;

    const int repetition_num = 5;

    std::vector< test_thread_config> thread_configs{
        test_thread_config {32 * 1, 1},
        test_thread_config {32 * 2, 1},
        test_thread_config {32 * 1, 2},
        test_thread_config {32 * 4, 1},
        test_thread_config {32 * 2, 2},
        test_thread_config {32 * 1, 4} };

    for (int c = 0; c < thread_configs.size(); c++)
    {
        auto thread_config = thread_configs[c];
        num_threads = thread_config.thread_count;
        num_blocks = thread_config.block_count;
        std::cout << "\n----\nThread config: " << num_blocks << " blocks with " << num_threads << " threads each\n";
        for (int m = start_m; m <= max_m; m += step_m)
        {
            std::cout << "\n----\nSize: " << m << std::endl;
            double* A = alloc_matrix(m);
            double* A_copy = alloc_matrix(m);

            fill_matrix(A, m);

            for (int i = 0; i < m * m; i++)
                A_copy[i] = A[i];


            if (!is_symmetric(A, m))
            {
                std::cout << "matrix is not symmetric!";
                return 0;
            }

            double* Vt;
            double* U;
            double* S;

            Vt = alloc_matrix(m);
            U = alloc_matrix(m);
            S = alloc_vector(m);

            std::vector<long long> times;

            for (int i = 0; i <= repetition_num; i++)
            {
                for (int j = 0; j < m * m; j++)
                    A[j] = A_copy[j];

                //std::this_thread::sleep_for(std::chrono::milliseconds(200));

                auto start = std::chrono::steady_clock::now();
                jacobi_svd(A, m, S, U, Vt);

                if (i != 0) // первый — прогрев
                    times.push_back(since(start).count());
            }

            double mean = getMean(times) / 1000;

            std::cout << "Elapsed(s) = " << mean << std::endl;

            double* buff = alloc_matrix(m);
            double* s = alloc_matrix(m);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                {
                    if (i == j)
                        s[IDX2C(j, i, m)] = S[i];
                    else
                        s[IDX2C(j, i, m)] = 0;
                }

            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++)
                {
                    double sum = 0;
                    for (int c = 0; c < m; c++)
                        sum += s[IDX2C(a, c, m)] * Vt[IDX2C(c, b, m)];

                    buff[IDX2C(a, b, m)] = sum;
                }

            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++)
                {
                    double sum = 0;
                    for (int c = 0; c < m; c++)
                        sum += U[IDX2C(a, c, m)] * buff[IDX2C(c, b, m)];

                    A[IDX2C(a, b, m)] = sum;
                }

            double biggest_error = 0;
            for (int i = 0; i < m * m; i++)
                biggest_error = biggest_error > abs(A[i] - A_copy[i]) ? biggest_error : abs(A[i] - A_copy[i]);

            std::cout << "Biggest error: " << biggest_error << std::endl;

            delete A;
            delete A_copy;
            delete Vt;
            delete U;
            delete S;
            delete buff;
            delete s;
        }
    }



}
