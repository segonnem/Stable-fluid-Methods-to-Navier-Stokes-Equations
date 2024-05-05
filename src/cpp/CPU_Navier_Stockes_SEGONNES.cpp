#include <iostream>
#include "Eigen/Dense"
#include <tuple>
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace Eigen;

const double DOMAIN_SIZE = 1.0;
const int N_POINTS = (1 << 7) - 1; // 2^6 - 1
const int N_TIME_STEPS = 15;
const double TIME_STEP_LENGTH = 0.1;
const double element_length = DOMAIN_SIZE / (N_POINTS - 1);
const double KINEMATIC_VISCOSITY = 0.0001;

void saveCurlsToCSV(const std::vector<Eigen::MatrixXd>& curls, const std::string& filename) {
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

extern "C" {

struct Tridiagonal {
    Eigen::VectorXd a;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
};

MatrixXd laplace_x(const MatrixXd& field) {
    MatrixXd diff = MatrixXd::Zero(field.rows(), field.cols());
    diff.block(1, 0, field.rows() - 2, field.cols()) =
        (field.block(0, 0, field.rows() - 2, field.cols())
        - 2 * field.block(1, 0, field.rows() - 2, field.cols())
        + field.block(2, 0, field.rows() - 2, field.cols())) / (element_length * element_length);
    return diff;
}

MatrixXd laplace_y(const MatrixXd& field) {
    MatrixXd diff = MatrixXd::Zero(field.rows(), field.cols());
    diff.block(0, 1, field.rows(), field.cols() - 2) =
        (field.block(0, 0, field.rows(), field.cols() - 2)
        - 2 * field.block(0, 1, field.rows(), field.cols() - 2)
        + field.block(0, 2, field.rows(), field.cols() - 2)) / (element_length * element_length);
    return diff;
}

MatrixXd build_laplacian_matrix(int N) {
    double dx = 1.0 / (N - 1);
    MatrixXd L = MatrixXd::Zero(N * N, N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = i * N + j;

            // Main diagonal
            L(k, k) = -4 / (dx * dx);

            // Off diagonals
            if (i > 0)      L(k, k - N) = 1 / (dx * dx);
            if (i < N - 1) L(k, k + N) = 1 / (dx * dx);
            if (j > 0)     L(k, k - 1) = 1 / (dx * dx);
            if (j < N - 1) L(k, k + 1) = 1 / (dx * dx);
        }
    }

    // Adjusting the main diagonal for Neumann boundary conditions
    L(0, 0) = L(N - 1, N - 1) = L(N * (N - 1), N * (N - 1)) = L(N * N - 1, N * N - 1) = -2 / (dx * dx);
    for (int k = 1; k < N - 1; ++k) {
        L(k, k) = L(N * N - N + k, N * N - N + k) = -3 / (dx * dx);
    }

    return L;
}

Tridiagonal construct_tridiagonal_for_x_CR(int N, double dt, double nu) {
    double dx = 1.0 / (N - 1);
    
    // Main diagonal
    Eigen::VectorXd b = Eigen::VectorXd::Ones(N) * (1 + 2 * nu * dt / (dx * dx));
    
    // Upper diagonal
    Eigen::VectorXd c = Eigen::VectorXd::Ones(N) * (-nu * dt / (dx * dx));
    c(N - 1) = 0;  // Setting the last element to 0
    
    // Lower diagonal
    Eigen::VectorXd a = Eigen::VectorXd::Ones(N) * (-nu * dt / (dx * dx));
    a(0) = 0;  // Setting the first element to 0
    
    Tridiagonal tri;
        tri.a = a;
        tri.b = b;
        tri.c = c;

    return tri;
    
}

Tridiagonal construct_tridiagonal_for_y_CR(int N, double dt, double nu) {
    // Identical to the x version since dx = dy for a uniform grid
    return construct_tridiagonal_for_x_CR(N, dt, nu);
}


VectorXd cyclic_reduction_cpu(const VectorXd& a_const, const VectorXd& b_const, const VectorXd& c_const, const VectorXd& F_const) {
    int size = b_const.size();
    VectorXd x = VectorXd::Zero(size);

    VectorXd a = a_const;
    VectorXd b = b_const;
    VectorXd c = c_const;
    VectorXd F = F_const;

    // Part 1 - Forward Reduction
    for (int i = 0; i < static_cast<int>(std::log2(size + 1)) - 1; ++i) {
        int step = static_cast<int>(std::pow(2, i + 1));
        for (int j = step - 1; j < size; j += step) {
            int offset = static_cast<int>(std::pow(2, i));
            int index1 = j - offset;
            int index2 = j + offset;

            if (j == size - 1) {
                double k1 = a[j] / b[j - offset];
                b[j] -= c[j - offset] * k1;
                F[j] -= F[j - offset] * k1;
                a[j] = -a[j - offset] * k1;
                c[j] = 0.0;
            } else {
                double k1 = a[j] / b[j - offset];
                double k2 = c[j] / b[j + offset];
                b[j] -= (c[j - offset] * k1 + a[j + offset] * k2);
                F[j] -= (F[j - offset] * k1 + F[j + offset] * k2);
                a[j] = -a[j - offset] * k1;
                c[j] = -c[j + offset] * k2;
            }
        }
    }

    // Part 2 - Find the middle
    int index = (size - 1) / 2;
    x[index] = F[index] / b[index];

    // Part 3 - Back substitution
    for (int i = static_cast<int>(std::log2(size + 1)) - 2; i >= 0; --i) {
        int step = static_cast<int>(std::pow(2, i + 1));
        for (int j = step - 1; j < size; j += step) {
            int offset = static_cast<int>(std::pow(2, i));
            int index1 = j - offset;
            int index2 = j + offset;

            if (j != index1 && index1 >= 0) {
                if (index1 - offset < 0) {
                    x[index1] = (F[index1] - c[index1] * x[index1 + offset]) / b[index1];
                } else {
                    x[index1] = (F[index1] - a[index1] * x[index1 - offset] - c[index1] * x[index1 + offset]) / b[index1];
                }
            }

            if (j != index2 && index2 < size) {
                if (index2 + offset >= size) {
                    x[index2] = (F[index2] - a[index2] * x[index2 - offset]) / b[index2];
                } else {
                    x[index2] = (F[index2] - a[index2] * x[index2 - offset] - c[index2] * x[index2 + offset]) / b[index2];
                }
            }
        }
    }
    x[size - 1] = (F[size - 1] - a[size - 1] * x[size - 2]) / b[size - 1];
    return x;
}

MatrixXd solve_ADI_CR(const MatrixXd& velocity, double dt, double nu, int N) {
    VectorXd a_x, b_x, c_x, a_y, b_y, c_y;

    Tridiagonal tri_x = construct_tridiagonal_for_x_CR(N, dt, nu);
    a_x = tri_x.a;
    b_x = tri_x.b;
    c_x = tri_x.c;

    Tridiagonal tri_y = construct_tridiagonal_for_y_CR(N, dt, nu);
    a_y = tri_y.a;
    b_y = tri_y.b;
    c_y = tri_y.c;

    
    // Step 1: Solve in the x direction for each row
    MatrixXd intermediate_velocity(N, N);
    for (int i = 0; i < N; ++i) {
        VectorXd laplace_term_y = nu * dt * laplace_y(velocity).row(i);
        
        VectorXd b_y_ = velocity.row(i).transpose() + laplace_term_y;

        intermediate_velocity.row(i) = cyclic_reduction_cpu(a_x, b_x, c_x, b_y_);
    }

    // Step 2: Solve in the y direction for each column
    MatrixXd final_velocity(N, N);
    for (int j = 0; j < N; ++j) {
        VectorXd laplace_term_x = nu * dt * laplace_x(intermediate_velocity).col(j);
        
        VectorXd b_x_ = intermediate_velocity.col(j) + laplace_term_x;
        final_velocity.col(j) = cyclic_reduction_cpu(a_y, b_y, c_y, b_x_);
        
    }

    return final_velocity;
}

MatrixXd construct_A_pressure(int N) {
    MatrixXd A = MatrixXd::Zero(N * N, N * N);
    double dx = 1.0 / (N - 1);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = i * N + j;

            // Center
            A(k, k) = -4.0 / (dx * dx);

            // Left
            if (j > 0)
                A(k, k-1) = 1.0 / (dx * dx);
            // Right
            if (j < N-1)
                A(k, k+1) = 1.0 / (dx * dx);
            // Up
            if (i > 0)
                A(k, k-N) = 1.0 / (dx * dx);
            // Down
            if (i < N-1)
                A(k, k+N) = 1.0 / (dx * dx);
        }
    }

    return A;
}


VectorXd jacobi_method(const MatrixXd& A, const VectorXd& b, int max_iterations = 1000, double tol = 1e-6) {
    int n = b.size();

    // Initial guess
    VectorXd x = VectorXd::Zero(n);

    // Diagonal of A and its inverse
    VectorXd D = A.diagonal();
    
    // L_U matrix: A without its diagonal
    MatrixXd L_U = A;
    L_U.diagonal().setZero();


    for (int i = 0; i < max_iterations; ++i) {

        VectorXd x_new = (b - L_U * x).array() / D.array();
        
        if ((x_new - x).norm() < tol) {
            return x_new;
        }

        x = x_new;
    }

    return x;
}

Eigen::VectorXd gradientConjugate(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) {
    int n = b.size();
    Eigen::VectorXd x(n); // Solution initiale
    x.setZero();
    
    Eigen::VectorXd r = b - A * x;
    Eigen::VectorXd p = r;
    Eigen::VectorXd Ap(n);
    
    double rDotOld = r.dot(r);
    double tol = 1e-6; // Vous pouvez ajuster cette tolérance
    int maxIter = 1000; // Vous pouvez ajuster le nombre maximum d'itérations

    for (int i = 0; i < maxIter; ++i) {
        Ap = A * p;
        double alpha = rDotOld / p.dot(Ap);
        x += alpha * p;
        r -= alpha * Ap;

        double rDotNew = r.dot(r);

        if (sqrt(rDotNew) < tol) {
            break;
        }

        p = r + (rDotNew / rDotOld) * p;
        rDotOld = rDotNew;
    }

    return x;
}
double forcing_function_x(double time, double x, double y) {
    double time_decay = std::max(2.0 - 0.5 * time, 0.0);

    // Pas de force horizontale pour ces faisceaux
    return 0.0;
}

double forcing_function_y(double time, double x, double y) {
    double time_decay = std::max(2.0 - 0.5 * time, 0.0);
    
    // Appliquer une force verticale dans cette région spécifique
    if (x > 0.4 && x < 0.6 && y > 0.1 && y < 0.3) {
        return time_decay * 1.0; // Force verticale appliquée
    }

    return 0.0; // Par défaut, aucune force verticale
}



MatrixXd partial_derivative_x(const MatrixXd& field) {
    MatrixXd diff = MatrixXd::Zero(field.rows(), field.cols());

    diff.block(1, 1, field.rows() - 2, field.cols() - 2) = 
        (field.block(2, 1, field.rows() - 2, field.cols() - 2) -
        field.block(0, 1, field.rows() - 2, field.cols() - 2)) / 
        (2 * element_length);

    return diff;
}

MatrixXd partial_derivative_y(const MatrixXd& field) {
    MatrixXd diff = MatrixXd::Zero(field.rows(), field.cols());

    diff.block(1, 1, field.rows() - 2, field.cols() - 2) =
        (field.block(1, 2, field.rows() - 2, field.cols() - 2) -
        field.block(1, 0, field.rows() - 2, field.cols() - 2)) / 
        (2 * element_length);

    return diff;
}

MatrixXd divergence(const MatrixXd& velocity_x, const MatrixXd& velocity_y) {

    return partial_derivative_x(velocity_x) + partial_derivative_y(velocity_y);
}

MatrixXd gradient_x(const MatrixXd& field) {
    return partial_derivative_x(field);
}

MatrixXd gradient_y(const MatrixXd& field) {
    return partial_derivative_y(field);
}

MatrixXd curl_2d(const MatrixXd& velocity_x, const MatrixXd& velocity_y) {
    return partial_derivative_x(velocity_y) - partial_derivative_y(velocity_x);
}

double interpolate(const MatrixXd& field, double i, double j) {
    int i0 = std::max(std::min(static_cast<int>(std::floor(i)), N_POINTS - 2), 0);
    int j0 = std::max(std::min(static_cast<int>(std::floor(j)), N_POINTS - 2), 0);
    int i1 = i0 + 1;
    int j1 = j0 + 1;

    double alpha = i - i0;
    double beta = j - j0;

    double value = (1.0 - alpha) * (1.0 - beta) * field(i0, j0) +
                   alpha * (1.0 - beta) * field(i1, j0) +
                   (1.0 - alpha) * beta * field(i0, j1) +
                   alpha * beta * field(i1, j1);

    return value;
}

MatrixXd advect(const MatrixXd& field, const MatrixXd& vector_field_x, const MatrixXd& vector_field_y) {
    MatrixXd advected = MatrixXd::Zero(N_POINTS, N_POINTS);

    for (int i = 1; i < N_POINTS - 1; ++i) {
        for (int j = 1; j < N_POINTS - 1; ++j) {
            double backtraced_i = i - TIME_STEP_LENGTH * vector_field_x(i, j) / element_length;
            double backtraced_j = j - TIME_STEP_LENGTH * vector_field_y(i, j) / element_length;
            
            backtraced_i = std::max(std::min(backtraced_i, static_cast<double>(N_POINTS - 2)), 1.0);
            backtraced_j = std::max(std::min(backtraced_j, static_cast<double>(N_POINTS - 2)), 1.0);
            advected(i, j) = interpolate(field, backtraced_i, backtraced_j);
        }
    }

    return advected;
}

}



int main() {

    std::vector<Eigen::MatrixXd> curls;
    std::vector<Eigen::MatrixXd> curls2;
    std::vector<Eigen::MatrixXd> curls3;

    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N_POINTS, 0.0, DOMAIN_SIZE);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(N_POINTS, 0.0, DOMAIN_SIZE);


    Eigen::MatrixXd velocity_x_prev = Eigen::MatrixXd::Zero(N_POINTS, N_POINTS);
    Eigen::MatrixXd velocity_y_prev = Eigen::MatrixXd::Zero(N_POINTS, N_POINTS);

    double time_current = 0.0;

    
    for (int step = 0; step < N_TIME_STEPS; ++step) {
        time_current += TIME_STEP_LENGTH;

        std::cout << step << std::endl;
        Eigen::MatrixXd forces_x = Eigen::MatrixXd::Zero(N_POINTS, N_POINTS);
        Eigen::MatrixXd forces_y = Eigen::MatrixXd::Zero(N_POINTS, N_POINTS);

        for (int i = 0; i < N_POINTS; ++i) {
            for (int j = 0; j < N_POINTS; ++j) {
                forces_x(i, j) = forcing_function_x(time_current, x(i), y(j));
                forces_y(i, j) = forcing_function_y(time_current, x(i), y(j));
            }
        }


        Eigen::MatrixXd velocity_x_forces_applied = velocity_x_prev + TIME_STEP_LENGTH * forces_x;
        Eigen::MatrixXd velocity_y_forces_applied = velocity_y_prev + TIME_STEP_LENGTH * forces_y;

        Eigen::MatrixXd velocity_x_advected = advect(velocity_x_forces_applied, velocity_x_forces_applied, velocity_y_forces_applied);
        Eigen::MatrixXd velocity_y_advected = advect(velocity_y_forces_applied, velocity_x_forces_applied, velocity_y_forces_applied);

    
        Eigen::MatrixXd velocity_x_diffused = solve_ADI_CR(velocity_x_advected, TIME_STEP_LENGTH, KINEMATIC_VISCOSITY, N_POINTS);
        Eigen::MatrixXd velocity_y_diffused = solve_ADI_CR(velocity_y_advected, TIME_STEP_LENGTH, KINEMATIC_VISCOSITY, N_POINTS);

        Eigen::MatrixXd velocity_x_diffused = velocity_x_advected;
        Eigen::MatrixXd velocity_y_diffused = velocity_y_advected;
        Eigen::MatrixXd A_for_pressure = construct_A_pressure(N_POINTS);
        Eigen::VectorXd b_pressure = Eigen::Map<Eigen::VectorXd>(divergence(velocity_x_diffused, velocity_y_diffused).data(), velocity_x_diffused.size());
        Eigen::VectorXd pressure_flat = gradientConjugate(A_for_pressure, b_pressure);

        Eigen::MatrixXd pressure = Eigen::MatrixXd::Map(pressure_flat.data(), N_POINTS, N_POINTS);
        
       
        Eigen::MatrixXd velocity_x_projected = velocity_x_diffused - gradient_x(pressure);
        Eigen::MatrixXd velocity_y_projected = velocity_y_diffused - gradient_y(pressure);

        velocity_x_prev = velocity_x_projected;
        velocity_y_prev = velocity_y_projected;

        Eigen::MatrixXd curl = curl_2d(velocity_x_projected, velocity_y_projected);
        

    
    }
    


}