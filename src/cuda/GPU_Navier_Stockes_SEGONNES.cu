#include <iostream>
#include "Eigen/Dense" // les solveurs ne sont pas utilisées, ils sont codés en GPU !
#include <tuple>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cusparse.h> //version optimale avec CSR representation
#include <cmath>  
#include <cstdio> 
#include <vector>

#define NB_ELEM_MAT 16
#define BLOCK_SIZE_MAT 16
#define BLOCK_DIM_VEC 16

using namespace Eigen;

const double DOMAIN_SIZE = 1.0;
const int N_POINTS = (1 << 6) - 1; // 2^6 - 1
const int N_TIME_STEPS = 1000;
const double TIME_STEP_LENGTH = 0.01;
const double element_length = DOMAIN_SIZE / (N_POINTS - 1);
const double KINEMATIC_VISCOSITY = 0.00001;
const int padding = 2 * log2(N_POINTS + 1);

struct Tridiagonal {
    Eigen::VectorXd a;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
};

void saveCurlsToCSV(const std::vector<Eigen::MatrixXd>& curls, const std::string& filename) {

    // utile pour afficher avec matplotlib sur python, je n'ai pas d'équivalent en C++ pour l'instant
    std::ofstream file(filename);

    for (const auto& curl : curls) {
        for (int i = 0; i < curl.rows(); ++i) {
            for (int j = 0; j < curl.cols(); ++j) {
                file << std::setprecision(15) << curl(i, j);
                if (!(i == curl.rows() - 1 && j == curl.cols() - 1)) {  // Ne pas ajouter de virgule après la dernière entrée
                    file << ",";
                }
            }
        }
        file << "\n";
    }
}

Tridiagonal construct_tridiagonal_for_x_CR(int N, double dt, double nu, int padding) {
    double dx = 1.0 / (N - 1);
    int padSize = log2(N + 1);

    // Create vectors with padding at both ends
    Eigen::VectorXd a = Eigen::VectorXd::Zero(N + 2 * padSize);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N + 2 * padSize);
    Eigen::VectorXd c = Eigen::VectorXd::Zero(N + 2 * padSize);

    // Main diagonal
    for (int i = padSize; i < N + padSize; ++i) {
        b(i) = 1 + nu * dt / (dx * dx);
    }

    // Upper diagonal
    for (int i = padSize; i < N + padSize - 1; ++i) {
        c(i) = -nu * dt / (2 * dx * dx);
    }

    // Lower diagonal
    for (int i = padSize + 1; i < N + padSize; ++i) {
        a(i) = -nu * dt / (2 * dx * dx);
    }

    Tridiagonal tri;
    tri.a = a;
    tri.b = b;
    tri.c = c;

    return tri;
}

Tridiagonal construct_tridiagonal_for_y_CR(int N, double dt, double nu, int padding) {
    // Identical to the x version since dx = dy for a uniform grid
    return construct_tridiagonal_for_x_CR(N, dt, nu, padding);
}

__host__ __device__ double forcing_function_x(double time, double x, double y) {

    return 0.0;
}

__host__ __device__ double forcing_function_y(double time, double x, double y) {
    double time_decay = fmax(2.0 - 0.5 * time, 0.0);
    if (0.4 < x && x < 0.6 && 0.1 < y && y < 0.3) {
        return time_decay;
    }
    return 0.0;
}

__host__ __device__ double forcing_function_x_spin(double time, double x, double y) {
    double time_decay = fmax(2.0 - 0.5 * time, 0.0);

    // Pas de force horizontale pour ces faisceaux
    return 0.0;
}

__host__ __device__ double forcing_function_y_spin(double time, double x, double y) {
    double time_decay = fmax(2.0 - 0.5 * time, 0.0);
    
    // Premier faisceau : partant du bas et se dirigeant vers le haut
    if (x > 0.35 && x < 0.45 && y > 0.1 && y < 0.5) {
        return time_decay * 1.0;
    }
    
    // Deuxième faisceau : partant du haut et se dirigeant vers le bas
    if (x > 0.55 && x < 0.65 && y > 0.5 && y < 0.9) {
        return time_decay * -1.0;
    }

    return 0.0;
}

__global__ void computeForcesX(double* d_forces_x, const double* d_x, const double* d_y, double time_current, int N_POINTS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N_POINTS && j < N_POINTS) {
        d_forces_x[i * N_POINTS + j] = forcing_function_x_spin(time_current, d_x[i], d_y[j]);
    }
}

__global__ void computeForcesY(double* d_forces_y, const double* d_x, const double* d_y, double time_current, int N_POINTS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N_POINTS && j < N_POINTS) {
        d_forces_y[i * N_POINTS + j] = forcing_function_y_spin(time_current, d_x[i], d_y[j]);
    }
}

__global__ void apply_forces_kernel(double* velocity_result, const double* velocity_prev, const double* forces, double TIME_STEP_LENGTH, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        velocity_result[idx] = velocity_prev[idx] + TIME_STEP_LENGTH * forces[idx];
    }
}

__global__ void updateVelocityWithForcesKernel(double* d_velocity_x, const double* d_forces_x, int N_POINTS, double TIME_STEP_LENGTH) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N_POINTS && j < N_POINTS) {
        int idx = i * N_POINTS + j;
        d_velocity_x[idx] += TIME_STEP_LENGTH * d_forces_x[idx];
    }
}

__device__ double interpolate(const double* field, int N_POINTS, double i, double j) {

    // interpolation bilinéaire classique
    int i0 = max(min(static_cast<int>(floor(i)), N_POINTS - 2), 0);
    int j0 = max(min(static_cast<int>(floor(j)), N_POINTS - 2), 0);
    int i1 = i0 + 1;
    int j1 = j0 + 1;

    double alpha = i - i0;
    double beta = j - j0;

    return (1.0 - alpha) * (1.0 - beta) * field[i0 * N_POINTS + j0] +
           alpha * (1.0 - beta) * field[i1 * N_POINTS + j0] +
           (1.0 - alpha) * beta * field[i0 * N_POINTS + j1] +
           alpha * beta * field[i1 * N_POINTS + j1];
}

__global__ void advectKernel(double* field_x, double* field_y, const double* vector_field_x, const double* vector_field_y, int N_POINTS, double TIME_STEP_LENGTH, double element_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < N_POINTS - 1 && j >= 1 && j < N_POINTS - 1) {
        double backtraced_i = i - TIME_STEP_LENGTH * vector_field_x[i * N_POINTS + j] / element_length;
        double backtraced_j = j - TIME_STEP_LENGTH * vector_field_y[i * N_POINTS + j] / element_length;

        backtraced_i = max(min(backtraced_i, static_cast<double>(N_POINTS - 2)), 1.0);
        backtraced_j = max(min(backtraced_j, static_cast<double>(N_POINTS - 2)), 1.0);

        // Advection for field_x
        double advected_value_x = interpolate(field_x, N_POINTS, backtraced_i, backtraced_j);
        field_x[i * N_POINTS + j] = advected_value_x;

        // Advection for field_y
        double advected_value_y = interpolate(field_y, N_POINTS, backtraced_i, backtraced_j);
        field_y[i * N_POINTS + j] = advected_value_y;
    }
}

void advect(double* d_velocity_x_prev, double* d_velocity_y_prev, const double* d_vector_field_x, const double* d_vector_field_y, int N_POINTS, dim3 gridDim, dim3 blockDim) {
    //appel au kernel
    advectKernel<<<gridDim, blockDim>>>(d_velocity_x_prev, d_velocity_y_prev, d_vector_field_x, d_vector_field_y, N_POINTS, TIME_STEP_LENGTH, element_length);
    cudaDeviceSynchronize();
}

__global__ void laplace_x_kernel(const double* field, double* diff, int rows, int cols, double element_length_inv_sq) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    
    if (i > 0 && i < rows-1 && j < cols) {
        int idx = i*cols + j;
        diff[idx] = (field[(i-1)*cols + j] - 2.0*field[idx] + field[(i+1)*cols + j]) * element_length_inv_sq;
    }
}

__global__ void laplace_y_kernel(const double* field, double* diff, int rows, int cols, double element_length_inv_sq) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;


    if (i < rows && j > 0 && j < cols-1) {
        int idx = i*cols + j;
        diff[idx] = (field[i*cols + j-1] - 2.0*field[idx] + field[i*cols + j+1]) * element_length_inv_sq;
    }
}

__device__ int log2(int x){
        return  (int)(log((double)x) / log(2.0));
}

__device__ int log2_dev(int x){
        return  (int)(log((double)x) / log(2.0));
}

__device__ void calc_dim(int size,dim3 *block,dim3 *grid){


       //ajustement des dimensions de grid et bloc pour la performance du calcul GPU
        if(size<4)              { block->x=1;block->y=1;}
        else if(size<16)        { block->x=2;block->y=2;}
        else if(size<64)        { block->x=4;block->y=4;}
        else if(size<256)       { block->x=8;block->y=8;}
        else                    { block->x=16;block->y=16;}

        // s'assurer qu'on a suffisamment de blocs avec ceil
        grid->x = (unsigned int)ceil(sqrt((double)size/block->x));
        grid->y = (unsigned int)ceil(sqrt((double)size/block->y));
}

__global__ void cr_forward(double *a, double *b, double *c, double *f, double *x,int size,int padding,int step_size,int i){


        // La premiere étape de propagation de la résolution d'un systeme tridiagonal (forward)
        int index_x = blockIdx.x * blockDim.x + threadIdx.x;
        int index_y = blockIdx.y * blockDim.y + threadIdx.y;
        int grid_width = gridDim.x * blockDim.x;
        int threadid = index_y * grid_width + index_x;

        int index1,index2,offset;
        float k1,k2;
        if(threadid>=step_size) return;
        int j = pow(2.0,i+1)*(threadid+1)-1+padding;

        offset = pow(2.0,i);
        index1 = j-offset;
        index2 = j+offset;

        k1 = a[j]/b[index1];
        k2 = c[j]/b[index2];


        if(j == size+padding - 1){
                k1 = a[j] / b[j-offset];
                b[j] = b[j] - c[j-offset] * k1;
                f[j] = f[j] - f[j-offset] * k1;
                a[j] = -a[j-offset] * k1;
                c[j] = 0.0;
        }
        else{
                k1 = a[j] / b[j-offset];
                k2 = c[j] / b[j+offset];
                b[j] = b[j] - c[j-offset] * k1 - a[j+offset] * k2;
                f[j] = f[j] - f[j-offset] * k1 - f[j+offset] * k2;
                a[j] = -a[j-offset] * k1;
                c[j] = -c[j+offset] * k2;
        }
}

__global__ void cr_backward(double *a, double *b, double *c, double *f, double *x,int size,int padding,int step_size,int i){

    // La deuxieme étape de propagation de la résolution d'un systeme tridiagonal après forward (backward)
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int threadid = index_y * grid_width + index_x;

    int index1,index2,offset;
    if(threadid>=step_size) return;

    int j = pow(2.0,i+1)*(threadid+1)-1+padding;

    offset = pow(2.0,i);
    index1 = j-offset;
    index2 = j+offset;

    if(j!=index1){
        x[index1] = (f[index1] - a[index1]*x[index1-offset] - c[index1]*x[index1+offset])/b[index1];
    }
    if(j!=index2){
        if(index2 == size-1) { // Check if it is the last index
            x[index2] = (f[index2] - a[index2]*x[index2-offset]) / b[index2];
        } else {
            x[index2] = (f[index2] - a[index2]*x[index2-offset] - c[index2]*x[index2+offset])/b[index2];
        }
    }
}

__global__ void cr_div(double *b,double *f,double *x,int index){
        x[index] = f[index]/b[index];
}

__global__ void solveCR_GPU(double* d_a, double* d_b, double* d_c, double* f, double* x, int N, int padding) {

    int z = blockIdx.x * blockDim.x + threadIdx.x;  // Index for z-direction (system index) //chaque ligne est indépendante !!

    if (z < N) {
        // Use z to index into the correct system.
        double* current_f = &f[z * (N + padding)]; //ligne
        double* current_x = &x[z * (N + padding)]; //ligne

        // Adjust the grid and block dimensions for the inner processing.
        dim3 dimBlock(32, 32); 
        dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

        int step_size;

        // Phase forward (réduction)
        for (int i = 0; i < log2(N + 1) - 1; ++i) {
            step_size = (N - pow(2.0, i + 1)) / pow(2.0, i + 1) + 1;
            calc_dim(step_size, &dimBlock, &dimGrid);
            cr_forward <<<dimGrid, dimBlock>>> (d_a, d_b, d_c, current_f, current_x, N, padding / 2, step_size, i);
        }

        // Division
        cr_div <<<1, 1>>> (d_b, current_f, current_x, padding / 2 + (N - 1) / 2);

        // Phase backward (substitution)
        for (int i = log2(N + 1) - 2; i >= 0; --i) {
            step_size = (N - pow(2.0, i + 1)) / pow(2.0, i + 1) + 1;
            calc_dim(step_size, &dimBlock, &dimGrid);
            cr_backward <<<dimGrid, dimBlock>>> (d_a, d_b, d_c, current_f, current_x, N, padding / 2, step_size, i);
        }

    }
}

__global__ void calculate_rhs_kernel(double* d_velocity, double* rhs, int N, int padding, double diffusion) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < N) {
        int idx = i * (N + padding) + j + padding/2;

        if(j == 0 || j == N - 1) {
            // Condition de bord de Dirichlet sauf erreur
            rhs[idx] = 0.0;  
        } else {
            // Éléments intérieurs
            rhs[idx] = d_velocity[i * N + j] + diffusion * (d_velocity[i * N + j - 1] - 2 * d_velocity[i * N + j] + d_velocity[i * N + j + 1]);
        } 
    }
}

__global__ void calculate_rhs_y_kernel(double* d_inter_velocity, double* rhs, int N, int padding, double diffusion) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < N) {
        int idx = j * (N + padding) + i + padding/2;

        if(i == 0 || i == N - 1) {
            // Condition de bord de Dirichlet
            rhs[idx] = 0.0;  
        } else {
            // Éléments intérieurs
            rhs[idx] = d_inter_velocity[j * N + i] + diffusion * (d_inter_velocity[j * N + i - 1] - 2 * d_inter_velocity[j * N + i] + d_inter_velocity[j * N + i + 1]);
        } 
    }
}

__global__ void update_output_velocity_kernel(double* d_output_velocity, double* x, int N, int padding) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < N) {
        // Mise à jour de d_output_velocity en utilisant x avec padding
        // Sauter le padding gauche correctement !
        d_output_velocity[j + i * N] = x[i * (N + padding) + (j + padding / 2)];
    }
}

__global__ void update_output_velocity_y_kernel(double* d_output_velocity, double* rhs, int N, int padding) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < N) {
        // Mise à jour de d_output_velocity en utilisant rhs avec padding
        // L'indexation tient compte du padding pour la dimension x
        d_output_velocity[j * N + i] = rhs[j * (N + padding) + i + padding / 2];
    }
}

void solve_x_direction_kernel(double* d_velocity, double* d_a_x, double* d_b_x, double* d_c_x, double* d_output_velocity, double* rhs, double* x, int N, int padding, double diffusion) {

    dim3 dimBlock2D(32, 32);  // Configuration pour les kernels qui fonctionnent sur la grille 2D
    dim3 dimGrid2D((N + dimBlock2D.x - 1) / dimBlock2D.x, (N + dimBlock2D.y - 1) / dimBlock2D.y);

    dim3 dimBlock1D(256);  // Configuration pour le kernel qui fonctionne sur une grille 1D
    dim3 dimGrid1D((N + dimBlock1D.x - 1) / dimBlock1D.x);

    // Étape 1: Calculer le nouveau rhs
    calculate_rhs_kernel<<<dimGrid2D, dimBlock2D>>>(d_velocity, rhs, N, padding, diffusion);

    // Étape 2: Appel du solveur
    solveCR_GPU<<<dimGrid1D, dimBlock1D>>>(d_a_x, d_b_x, d_c_x, rhs, x, N, padding);

    // Synchronisation
    cudaDeviceSynchronize();

    // Étape 3: Mettre à jour d_output_velocity
    update_output_velocity_kernel<<<dimGrid2D, dimBlock2D>>>(d_output_velocity, x, N, padding);

    // d_velo_intermediate is computed now : we can compute d_velo_final

}

void solve_y_direction_kernel(double* d_velocity, double* d_a_y, double* d_b_y, double* d_c_y, double* d_output_velocity, double* rhs, double* x, int N, int padding, double diffusion) {

    dim3 dimBlock2D(32, 32);  // Configuration pour les kernels qui fonctionnent sur la grille 2D
    dim3 dimGrid2D((N + dimBlock2D.x - 1) / dimBlock2D.x, (N + dimBlock2D.y - 1) / dimBlock2D.y);

    dim3 dimBlock1D(256);  // Configuration pour le kernel qui fonctionne sur une grille 1D
    dim3 dimGrid1D((N + dimBlock1D.x - 1) / dimBlock1D.x);

    // Étape 1: Calculer le nouveau rhs pour la direction y
    calculate_rhs_y_kernel<<<dimGrid2D, dimBlock2D>>>(d_velocity, rhs, N, padding, diffusion);

    // Synchronisation
    cudaDeviceSynchronize();

    // Étape 2: Appel du solveur
    solveCR_GPU<<<dimGrid1D, dimBlock1D>>>(d_a_y, d_b_y, d_c_y, rhs, x, N, padding);

    // Synchronisation
    cudaDeviceSynchronize();

    // Étape 3: Mettre à jour d_output_velocity pour la direction y
    update_output_velocity_y_kernel<<<dimGrid2D, dimBlock2D>>>(d_output_velocity, x, N, padding);
}


void solve_ADI_CR_gpu(double* d_velocity_advected, double dt, double nu, int N,
                      double* d_a_x, double* d_b_x, double* d_c_x,
                      double* d_a_y, double* d_b_y, double* d_c_y, int padding,
                      double* d_intermediate_velocity,
                      double* rhs, double* x)
{
    double diffusion = KINEMATIC_VISCOSITY * TIME_STEP_LENGTH /(element_length * element_length);
    // Step 1: Solve in the x direction
    
    solve_x_direction_kernel(d_velocity_advected, d_a_x, d_b_x, d_c_x, d_intermediate_velocity, rhs, x, N, padding, diffusion);

    // Step 2: Solve in the y direction
    solve_y_direction_kernel(d_intermediate_velocity, d_a_y, d_b_y, d_c_y, d_velocity_advected, rhs, x, N, padding, diffusion);
}

__global__ void partial_derivative_x_kernel(double* diff, const double* field, int rows, int cols, double element_length) {

    // calcul classique de la dérivée partielle selon x par différence finie ordre simple
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1) {
        int index = i * cols + j;
        int index_plus = (i+1) * cols + j;
        int index_minus = (i-1) * cols + j;

        diff[index] = (field[index_plus] - field[index_minus]) / (2.0 * element_length);
    }
}

__global__ void partial_derivative_y_kernel(double* diff, const double* field, int rows, int cols, double element_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1) {
        int index = i * cols + j;
        int index_plus = i * cols + (j+1);
        int index_minus = i * cols + (j-1);

        diff[index] = (field[index_plus] - field[index_minus]) / (2.0 * element_length);
    }
}

__global__ void combine_derivatives_kernel(double* divergence, const double* partial_x, const double* partial_y, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        int index = i * cols + j;
        divergence[index] = partial_x[index] + partial_y[index];
    }
}

void computeDivergenceOnGPU(double* d_divergence, const double* d_velocity_x, const double* d_velocity_y,
                            double* d_partial_x, double* d_partial_y,
                            int N_POINTS, dim3 gridDim, dim3 blockDim, double element_length) {

    partial_derivative_x_kernel<<<gridDim, blockDim>>>(d_partial_x, d_velocity_x, N_POINTS, N_POINTS, element_length);
    partial_derivative_y_kernel<<<gridDim, blockDim>>>(d_partial_y, d_velocity_y, N_POINTS, N_POINTS, element_length);
    // calcul de la divergence
    combine_derivatives_kernel<<<gridDim, blockDim>>>(d_divergence, d_partial_x, d_partial_y, N_POINTS, N_POINTS);
}


// Une fonction d'aide pour l'indexation aplatit pour notre matrice N^2 x N^2 de sorte à eviter les erreurs 
__device__ int idx(int row, int col, int N) {
    return row * N * N + col;
}

__global__ void construct_A_pressure_kernel(double* A, int N, double dx) {

    // construction de A dans Ax = b, s'obtiens après un calcul de différence finie (cf equation diffusion 2D)
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    // Stride
    for (int i = tx; i < N; i += blockDim.x * gridDim.x) {
        for (int j = ty; j < N; j += blockDim.y * gridDim.y) {
            // Diagonal principale
            A[idx(i * N + j, i * N + j, N)] = -4.0 / (dx * dx);


            // Diagonale gauche
            if (j > 0) {
                A[idx(i * N + j, i * N + j - 1, N)] = 1.0 / (dx * dx);
            }

            // Diagonale droite
            if (j < N - 1) {
                A[idx(i * N + j, i * N + j + 1, N)] = 1.0 / (dx * dx);
            }

            // Diagonale supérieure
            if (i > 0) {
                A[idx(i * N + j, (i - 1) * N + j, N)] = 1.0 / (dx * dx);
            }

            // Diagonale inférieure
            if (i < N - 1) {
                A[idx(i * N + j, (i + 1) * N + j, N)] = 1.0 / (dx * dx);
            }
        }
    }
}

double* create_A_pressure_on_GPU(int N, dim3 gridDim, dim3 blockDim) {

    double* d_A;
    cudaMalloc(&d_A, N * N * N * N * sizeof(double));

    // Set the entire matrix to zero
    cudaMemset(d_A, 0, N * N * N * N * sizeof(double));

    double dx = 1.0 / (N - 1);

    construct_A_pressure_kernel<<<gridDim, blockDim>>>(d_A, N, dx);
    return d_A;
}

__global__ void matrixVectorMultiplyOptimizedKernel(const double* A, const double* x, double* result, int n) {

    // fonction (idée) trouvée sur git et adaptée légèrement pour calculer de manière performate le produit vecteur/matrice
    // D'après mes recherches, atomicAdd fonctionne dorénavant sur les doubles

    __shared__ double x_shared[NB_ELEM_MAT];

    int effective_block_width;
    if ((blockIdx.x + 1) * NB_ELEM_MAT <= n) {
        effective_block_width = NB_ELEM_MAT;
    } else {
        effective_block_width = n % NB_ELEM_MAT;
    }

    if (threadIdx.x < effective_block_width) {
        x_shared[threadIdx.x] = x[blockIdx.x * NB_ELEM_MAT + threadIdx.x];
    }

    __syncthreads();

    int idx = blockIdx.y * BLOCK_SIZE_MAT + threadIdx.x;
    double tmp_sum = 0.0;
    if (idx < n) {
        for (int j = 0; j < effective_block_width; j++) {
            tmp_sum += x_shared[j] * A[idx * n + blockIdx.x * NB_ELEM_MAT + j];
        }
        atomicAdd(&result[idx], tmp_sum);
    }
}

__global__ void vectorAdditionKernel(const double* a, const double* b, double* alpha, double* result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        result[i] = a[i] + *alpha * b[i];
    }
}

__global__ void vectorAdditionKernel2(const double* a, double* b, double* result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        result[i] = a[i] + b[i];
        b[i] = 0.0;
    }
}

__global__ void vectorSubtractionKernel(const double* a, const double* b, double* alpha, double* result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    
    for (int i = index; i < n; i += stride) {

        result[i] = a[i] - *alpha * b[i];
    }
}

__global__ void dotProductOptimizedKernel(const double* a, const double* b, double* out, int n) {

        // each block has it's own shared_tmp of size BLOCK_DIM_VEC
        __shared__ float shared_tmp[BLOCK_DIM_VEC];

        // nnécessaire pour atomicAdd
        if (threadIdx.x + blockDim.x * blockIdx.x == 0) {
                *out = 0.0;
        }


        if (blockIdx.x * blockDim.x + threadIdx.x < n) {
                shared_tmp[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x]
                                * b[blockIdx.x * blockDim.x + threadIdx.x];
        } else {
                
                shared_tmp[threadIdx.x] = 0.0;
        }

        // reduction within block
        for (int i = blockDim.x / 2; i >= 1; i = i / 2) {
                // threads access memory position written by other threads so sync is needed
                __syncthreads();
                if (threadIdx.x < i) {
                        shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + i];
                }
        }

        // atomic add the partial reduction in out
        if (threadIdx.x == 0) {
                atomicAdd(out, shared_tmp[0]);
        }
}

__global__ void scalarVectorMultiplyKernel(const double* scalar, const double* vec, double* result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        result[i] = vec[i] * (*scalar);
    }
}

__global__ void computeQuotientKernel(const double* numerator, const double* denominator, double* quotient) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *quotient = *numerator / *denominator;
    }
}

__global__ void vectorCopyKernel(const double* src, double* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void matrixVectorMultiplyCSRKernel(

    
    const double* val, const int* colInd, const int* rowPtr,
    const double* x, double* result, int n
) {

    //Première fonction simple non optimale
    int thread_id = blockIdx.y * blockDim.y + threadIdx.y;
    int stride = gridDim.y * blockDim.y;

    for (int row = thread_id; row < n; row += stride) {
        double dot = 0.0;
        int row_start = rowPtr[row];
        int row_end = rowPtr[row + 1];

        for (int j = row_start; j < row_end; j++) {
            dot += val[j] * x[colInd[j]];
        }

        result[row] = dot;
    }
}

void solveCG_cuda(cusparseHandle_t handle, cusparseSpMatDescr_t matDescr, 
                  cusparseDnVecDescr_t vecInputDescr, cusparseDnVecDescr_t vecOutputDescr, 
                  void* d_buffer, double* d_val, int* d_colInd, int* d_rowPtr, 
                  double* d_b, double* d_x, double* d_p, double* d_r, double* d_temp,
                  double* d_alpha, double* d_beta, double* d_r_norm, double* d_r_norm_old,
                  double* d_temp_scal, double* h_r_norm, int n)
{

    int MAX_ITER = 100000000;
    double EPS = 1e-8;
    
    dim3 vec_block_dim(BLOCK_DIM_VEC);
    dim3 vec_grid_dim((n + BLOCK_DIM_VEC - 1) / BLOCK_DIM_VEC);

    dim3 mat_grid_dim((n + NB_ELEM_MAT - 1) / NB_ELEM_MAT, (n + BLOCK_SIZE_MAT - 1) / BLOCK_SIZE_MAT);
    dim3 mat_block_dim(BLOCK_SIZE_MAT);


    cudaMemset(d_r_norm_old, 0, sizeof(double));
    dotProductOptimizedKernel<<<vec_grid_dim, vec_block_dim>>>(d_r, d_r, d_r_norm_old, n);

    int k = 0;
    cudaError_t err;

    double h_temp_scal;
    double alpha;
    double beta;

    double alphaSpMV = 1.0;
    double betaSpMV = 0.0;

    while ((k < MAX_ITER) && (*h_r_norm > EPS)) {

        // Calcul de d_temp = A * d_p 
        //cudaMemset(d_temp, 0, sizeof(double) * n);
        //matrixVectorMultiplyCSRKernel<<<mat_grid_dim, mat_block_dim>>>(d_val, d_colInd, d_rowPtr, d_p, d_temp, n);
        // betaspmv = 0 donc pas besoin de mettre d_Ap à 0


        // On utilise ici la représentation CSR pour les matrices creuses, le gain de performance est énorme !
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
             &alphaSpMV, matDescr, vecInputDescr, &betaSpMV, vecOutputDescr, 
             CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);

        // alpha_k = ...
        cudaMemset(d_temp_scal, 0, sizeof(double));
        dotProductOptimizedKernel<<<vec_grid_dim, vec_block_dim>>>(d_p, d_temp, d_temp_scal, n);

        computeQuotientKernel<<<1, 1>>>(d_r_norm_old, d_temp_scal, d_alpha);

        vectorSubtractionKernel<<<vec_grid_dim, vec_block_dim>>>(d_r, d_temp, d_alpha, d_r, n);

        // d_x_{k+1} = ...
        vectorAdditionKernel<<<vec_grid_dim, vec_block_dim>>>(d_x, d_p, d_alpha, d_x, n);

        cudaMemset(d_r_norm, 0, sizeof(double));
        dotProductOptimizedKernel<<<vec_grid_dim, vec_block_dim>>>(d_r, d_r, d_r_norm, n);

        computeQuotientKernel<<<1, 1>>>(d_r_norm, d_r_norm_old, d_beta);

        // d_p_{k+1} = ...
        scalarVectorMultiplyKernel<<<vec_grid_dim, vec_block_dim>>>(d_beta, d_p, d_temp, n);

        vectorAdditionKernel2<<<vec_grid_dim, vec_block_dim>>>(d_r, d_temp, d_p, n);

        // Copie de d_r_norm à d_r_norm_old
        vectorCopyKernel<<<vec_grid_dim, vec_block_dim>>>(d_r_norm, d_r_norm_old, n);

        // Copie de d_r_norm à la mémoire hôte (pour évaluer la condition d'arrêt)
        cudaMemcpy(h_r_norm, d_r_norm, sizeof(double), cudaMemcpyDeviceToHost);

        k++;
        
    }
}

__global__ void gradient_x_kernel(double* velocity, const double* pressure, int rows, int cols, double element_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1) {
        int index = i * cols + j;
        int index_plus = (i+1) * cols + j;
        int index_minus = (i-1) * cols + j;

        double gradient = (pressure[index_plus] - pressure[index_minus]) / (2.0 * element_length);
        velocity[index] -= gradient;
    }
}

__global__ void gradient_y_kernel(double* velocity, const double* pressure, int rows, int cols, double element_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1) {
        int index = i * cols + j;
        int index_plus = i * cols + (j+1);
        int index_minus = i * cols + (j-1);

        double gradient = (pressure[index_plus] - pressure[index_minus]) / (2.0 * element_length);
        velocity[index] -= gradient;
    }
}

__global__ void curlKernel(const double* velocity_x, const double* velocity_y, double* curl, int rows, int cols, double element_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1) {
        int index = i * cols + j;

        int index_x_plus = (i+1) * cols + j;
        int index_x_minus = (i-1) * cols + j;
        int index_y_plus = i * cols + (j+1);
        int index_y_minus = i * cols + (j-1);

        double dvy_dx = (velocity_y[index_x_plus] - velocity_y[index_x_minus]) / (2.0 * element_length);
        double dvx_dy = (velocity_x[index_y_plus] - velocity_x[index_y_minus]) / (2.0 * element_length);

        curl[index] = dvy_dx - dvx_dy;
    }
}

__global__ void curlKernelHigherOrder(


    // pas très utile en fin de compte, l'ordre de base suffit
    const double* velocity_x, const double* velocity_y,
    double* curl, int rows, int cols, double element_length) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 1 && i < rows - 2 && j > 1 && j < cols - 2) {
        int index = i * cols + j;

        int index_x_plus_1 = (i+1) * cols + j;
        int index_x_plus_2 = (i+2) * cols + j;
        int index_x_minus_1 = (i-1) * cols + j;
        int index_x_minus_2 = (i-2) * cols + j;

        int index_y_plus_1 = i * cols + (j+1);
        int index_y_plus_2 = i * cols + (j+2);
        int index_y_minus_1 = i * cols + (j-1);
        int index_y_minus_2 = i * cols + (j-2);

        double dvy_dx = (- velocity_y[index_x_plus_2] + 8 * velocity_y[index_x_plus_1] - 8 * velocity_y[index_x_minus_1] + velocity_y[index_x_minus_2]) / (12.0 * element_length);
        double dvx_dy = (- velocity_x[index_y_plus_2] + 8 * velocity_x[index_y_plus_1] - 8 * velocity_x[index_y_minus_1] + velocity_x[index_y_minus_2]) / (12.0 * element_length);

        curl[index] = dvy_dx - dvx_dy;
    }
}

void createCSRPentadiagonal(double *d_val, int *d_colInd, int *d_rowPtr, int N, double dx) {
    int N2 = N * N; // Taille de la matrice N x N
    int nnz = 5 * N2 - 4 * N; // Nombre d'éléments non nuls

    // Allocation temporaire sur le CPU
    double *val = (double *)malloc(nnz * sizeof(double));
    int *colInd = (int *)malloc(nnz * sizeof(int));
    int *rowPtr = (int *)malloc((N2 + 1) * sizeof(int));

    // Remplissage des tableaux val, colInd, et rowPtr
    int counter = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            rowPtr[idx] = counter;

            // Diagonale gauche
            if (j > 0) {
                val[counter] = 1.0 / (dx * dx);
                colInd[counter++] = idx - 1;
            }

            // Diagonale supérieure
            if (i > 0) {
                val[counter] = 1.0 / (dx * dx);
                colInd[counter++] = idx - N;
            }

            // Diagonale principale
            val[counter] = -4.0 / (dx * dx);
            colInd[counter++] = idx;

            // Diagonale inférieure
            if (i < N - 1) {
                val[counter] = 1.0 / (dx * dx);
                colInd[counter++] = idx + N;
            }

            // Diagonale droite
            if (j < N - 1) {
                val[counter] = 1.0 / (dx * dx);
                colInd[counter++] = idx + 1;
            }
        }
    }
    rowPtr[N2] = counter;

    // Copie des données remplies sur le GPU
    cudaMemcpy(d_val, val, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, colInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, rowPtr, (N2 + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Libération de la mémoire sur le CPU
    free(val);
    free(colInd);
    free(rowPtr);
}

int main() {

    double* d_x;
    double* d_y;
    double* d_forces_x;
    double* d_forces_y;
    double* d_velocity_x_prev;
    double* d_velocity_y_prev;
    double *d_a_x, *d_b_x, *d_c_x, *d_a_y, *d_b_y, *d_c_y; 
    double* d_divergence;
    double* d_pressure;
    double* d_curl;
    double* d_r;
    double* d_p;
    double* d_Ap;
    double* d_partial_x;
    double* d_partial_y;
    double* d_result;

    double* d_alpha;
    double* d_beta;
    double* d_r_norm;
    double* d_r_norm_old;
    double* d_temp_scal;
    double *h_r_norm = (double *) malloc(sizeof(double));

    *h_r_norm = 1.0;
    double* d_intermediate_velocity;

    double* rhs;
    double* x_adi;

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N_POINTS, 0.0, DOMAIN_SIZE);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(N_POINTS, 0.0, DOMAIN_SIZE);

    cudaMalloc(&d_x, N_POINTS * sizeof(double));
    cudaMalloc(&d_y, N_POINTS * sizeof(double));
    cudaMemcpy(d_x, x.data(), N_POINTS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), N_POINTS * sizeof(double), cudaMemcpyHostToDevice);


    cudaMalloc(&d_forces_x, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_forces_y, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_velocity_x_prev, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_velocity_y_prev, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_divergence, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_pressure, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_curl, N_POINTS * N_POINTS * sizeof(double));
    cudaMemset(d_curl, 0, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_r, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_p, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_Ap, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_partial_x, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&d_partial_y, N_POINTS * N_POINTS * sizeof(double));

    cudaMemset(d_partial_x, 0, N_POINTS * N_POINTS * sizeof(double));
    cudaMemset(d_partial_y, 0, N_POINTS * N_POINTS * sizeof(double));

    cudaMalloc(&d_result, sizeof(double));

    cudaMalloc(&d_alpha, sizeof(double));
    cudaMalloc(&d_beta, sizeof(double));
    cudaMalloc(&d_r_norm, sizeof(double));
    cudaMalloc(&d_r_norm_old, sizeof(double));
    cudaMalloc(&d_temp_scal, sizeof(double));
    cudaMalloc(&d_intermediate_velocity, N_POINTS * N_POINTS * sizeof(double));
    cudaMalloc(&rhs, ((N_POINTS + padding) * N_POINTS) * sizeof(double));
    cudaMalloc(&x_adi, ((N_POINTS + padding) * N_POINTS) * sizeof(double));

    cudaMemset(rhs, 0, ((N_POINTS + padding) * N_POINTS) * sizeof(double));
    cudaMemset(x_adi, 0, ((N_POINTS + padding) * N_POINTS) * sizeof(double));
    cudaMemset(d_pressure, 0, N_POINTS * N_POINTS * sizeof(double));


    Eigen::VectorXd a_x, b_x, c_x, a_y, b_y, c_y;
    Tridiagonal tri_x = construct_tridiagonal_for_x_CR(N_POINTS, TIME_STEP_LENGTH, KINEMATIC_VISCOSITY, padding);
    a_x = tri_x.a;
    b_x = tri_x.b;
    c_x = tri_x.c;

    Tridiagonal tri_y = construct_tridiagonal_for_y_CR(N_POINTS, TIME_STEP_LENGTH, KINEMATIC_VISCOSITY, padding);
    a_y = tri_y.a;
    b_y = tri_y.b;
    c_y = tri_y.c;

    int paddedSize = N_POINTS + padding;

    cudaMalloc(&d_a_x, paddedSize * sizeof(double));
    cudaMalloc(&d_b_x, paddedSize * sizeof(double));
    cudaMalloc(&d_c_x, paddedSize * sizeof(double));
    cudaMalloc(&d_a_y, paddedSize * sizeof(double));
    cudaMalloc(&d_b_y, paddedSize * sizeof(double));
    cudaMalloc(&d_c_y, paddedSize * sizeof(double));

    cudaMemcpy(d_a_x, a_x.data(), paddedSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_x, b_x.data(), paddedSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_x, c_x.data(), paddedSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_y, a_y.data(), paddedSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_y, b_y.data(), paddedSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_y, c_y.data(), paddedSize * sizeof(double), cudaMemcpyHostToDevice);

    Eigen::MatrixXd velocity_x_prev = Eigen::MatrixXd::Zero(N_POINTS, N_POINTS);
    Eigen::MatrixXd velocity_y_prev = Eigen::MatrixXd::Zero(N_POINTS, N_POINTS);

    cudaMemcpy(d_velocity_x_prev, velocity_x_prev.data(), N_POINTS * N_POINTS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity_y_prev, velocity_y_prev.data(), N_POINTS * N_POINTS * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N_POINTS + blockDim.x - 1) / blockDim.x, (N_POINTS + blockDim.y - 1) / blockDim.y);

    double time_current = 0.0;

    /*****************************************************************************************************************\

    /*
    SPARSE MATRIX (cf documentation)
    */

    double dx = DOMAIN_SIZE / (N_POINTS - 1);
    int nnz = 5 * N_POINTS * N_POINTS - 4 * N_POINTS;

    double *d_val;
    int *d_colInd, *d_rowPtr;

    // Allocation de mémoire sur le GPU
    cudaMalloc((void**)&d_val, nnz * sizeof(double));
    cudaMalloc((void**)&d_colInd, nnz * sizeof(int));
    cudaMalloc((void**)&d_rowPtr, (N_POINTS * N_POINTS + 1) * sizeof(int));

    // CSR representation
    createCSRPentadiagonal(d_val, d_colInd, d_rowPtr, N_POINTS, dx);

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Création du descripteur de matrice CSR
    cusparseSpMatDescr_t matDescr;
    cusparseCreateCsr(&matDescr, N_POINTS * N_POINTS, N_POINTS * N_POINTS, nnz, 
                    d_rowPtr, d_colInd, d_val, 
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, 
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    double alphaSpMV = 1.0;
    double betaSpMV = 0.0;
    size_t bufferSize = 0;
    void* d_buffer = nullptr;

    // Préparation des descripteurs de vecteurs pour l'opération SpMV
    cusparseDnVecDescr_t vecInputDescr_p, vecOutputDescr_Ap;
    cusparseCreateDnVec(&vecInputDescr_p, N_POINTS * N_POINTS, d_p, CUDA_R_64F);
    cusparseCreateDnVec(&vecOutputDescr_Ap, N_POINTS * N_POINTS, d_Ap, CUDA_R_64F);

    // Détermination de la taille du buffer pour cusparseSpMV
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            &alphaSpMV, matDescr, vecInputDescr_p, &betaSpMV, vecOutputDescr_Ap, 
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    // Allocation du buffer
    cudaMalloc(&d_buffer, bufferSize);

    //**************************************************************************************************************

    double* h_curl = new double[N_POINTS * N_POINTS];

    std::vector<Eigen::MatrixXd> curls;
    double *d_transposed;
    cudaMalloc(&d_transposed, N_POINTS * N_POINTS * sizeof(double));

    for (int step = 0; step < N_TIME_STEPS; ++step) {

        time_current += TIME_STEP_LENGTH;

        std::cout << step << std::endl;

        //forcesx et forcesy sont calculées suivant les conditions pour la simulation
        computeForcesX<<<gridDim, blockDim>>>(d_forces_x, d_x, d_y, time_current, N_POINTS);
        computeForcesY<<<gridDim, blockDim>>>(d_forces_y, d_x, d_y, time_current, N_POINTS);

        // Ici, les forces sont déjà sur le GPU, donc nous devons juste les utiliser pour mettre à jour les vitesses.
        updateVelocityWithForcesKernel<<<gridDim, blockDim>>>(d_velocity_x_prev, d_forces_x, N_POINTS, TIME_STEP_LENGTH);
        updateVelocityWithForcesKernel<<<gridDim, blockDim>>>(d_velocity_y_prev, d_forces_y, N_POINTS, TIME_STEP_LENGTH);


        /***********************************************      ADVECTION     ***********************************************************************/

        advect(d_velocity_x_prev, d_velocity_y_prev, d_velocity_x_prev, d_velocity_y_prev, N_POINTS, gridDim, blockDim);


        /***********************************************      DIFFUSION     ***********************************************************************/

        solve_ADI_CR_gpu(d_velocity_x_prev, TIME_STEP_LENGTH, KINEMATIC_VISCOSITY, N_POINTS, d_a_x, d_b_x, d_c_x, d_a_y, d_b_y, d_c_y, padding, d_intermediate_velocity, rhs, x_adi);
        solve_ADI_CR_gpu(d_velocity_y_prev, TIME_STEP_LENGTH, KINEMATIC_VISCOSITY, N_POINTS, d_a_x, d_b_x, d_c_x, d_a_y, d_b_y, d_c_y, padding, d_intermediate_velocity, rhs, x_adi);


        /***********************************************      PRESSURE      ***********************************************************************/

        computeDivergenceOnGPU(d_divergence, d_velocity_x_prev, d_velocity_y_prev, d_partial_x, d_partial_y, N_POINTS, gridDim, blockDim, element_length);

        //initialisation variables
        cudaMemcpy(d_p, d_divergence, N_POINTS * N_POINTS * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_r, d_divergence, N_POINTS * N_POINTS * sizeof(double), cudaMemcpyDeviceToDevice);
        
        *h_r_norm = 1.0;
        cudaMemset(d_pressure, 0, N_POINTS * N_POINTS * sizeof(double));
        solveCG_cuda(handle, matDescr, vecInputDescr_p, vecOutputDescr_Ap, d_buffer, d_val, d_colInd, d_rowPtr, d_divergence, d_pressure, d_p, d_r, d_Ap, d_alpha, d_beta, d_r_norm, d_r_norm_old, d_temp_scal, h_r_norm, N_POINTS * N_POINTS);


        /******************************************************************************************************************************************/
       
        // Update d_velocity_x_prev in-place x-= gradient
        gradient_x_kernel<<<gridDim, blockDim>>>(d_velocity_x_prev, d_pressure, N_POINTS, N_POINTS, element_length);

        // Update d_velocity_y_prev in-place y-= gradient
        gradient_y_kernel<<<gridDim, blockDim>>>(d_velocity_y_prev, d_pressure, N_POINTS, N_POINTS, element_length);

        curlKernel<<<gridDim, blockDim>>>(d_velocity_x_prev, d_velocity_y_prev, d_curl, N_POINTS, N_POINTS, element_length);

        cudaMemcpy(h_curl, d_curl, N_POINTS * N_POINTS * sizeof(double), cudaMemcpyDeviceToHost);



        // copy dans le csv pour python
        Eigen::MatrixXd curlMatrix(N_POINTS, N_POINTS);

        for (int i = 0; i < N_POINTS; ++i) {
            for (int j = 0; j < N_POINTS; ++j) {
                curlMatrix(i, j) = h_curl[i * N_POINTS + j];

            }
        }
        curls.push_back(curlMatrix);
    }


    // Libération de la mémoire sur le GPU
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_forces_x);
    cudaFree(d_forces_y);
    cudaFree(d_velocity_x_prev);
    cudaFree(d_velocity_y_prev);
    cudaFree(d_a_x);
    cudaFree(d_b_x);
    cudaFree(d_c_x);
    cudaFree(d_a_y);
    cudaFree(d_b_y);
    cudaFree(d_c_y);
    cudaFree(d_divergence);
    cudaFree(d_pressure);
    cudaFree(d_curl);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_partial_x);
    cudaFree(d_partial_y);
    cudaFree(d_result);
    cudaFree(d_intermediate_velocity);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_r_norm);
    cudaFree(d_r_norm_old);
    cudaFree(d_temp_scal);
    cudaFree(rhs);
    cudaFree(x_adi);

    free(h_r_norm);

    saveCurlsToCSV(curls, "gpu_final.csv");

    cudaFree(d_val);
    cudaFree(d_colInd);
    cudaFree(d_rowPtr);

    cusparseDestroySpMat(matDescr);
    cusparseDestroyDnVec(vecInputDescr_p);
    cusparseDestroyDnVec(vecOutputDescr_Ap);
    cudaFree(d_buffer);
    cusparseDestroy(handle);
    cudaFree(d_transposed);



    }







