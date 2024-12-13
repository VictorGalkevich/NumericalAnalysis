#include <complex>
#include <iostream>
#include <vector>
#include "gaussian.h"

#include "../../../../opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current/gcc/aarch64-apple-darwin23/14/include-fixed/math.h"

size_t iterations = 0;

std::vector<std::vector<double> > generateTransformationMatrix(
    const std::vector<std::vector<double> > &A,
    const size_t k
) {
    const size_t n = A.size();

    const size_t posi = k + 1;
    const double a_n_n1 = A[posi][k];

    std::vector transformer(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        transformer[i][i] = 1;
    }

    for (int i = 0; i < n; ++i) {
        transformer[k][i] = -A[posi][i] / a_n_n1;
    }
    transformer[k][k] = 1 / a_n_n1;
    return transformer;
}

std::vector<std::vector<double> > TransposeSquareMatrix(const std::vector<std::vector<double> > &initial) {
    const size_t n = initial.size();
    std::vector transposed(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transposed[j][i] = initial[i][j];
        }
    }
    return transposed;
}

std::vector<std::vector<double> > MultiplySquareMatrices(
    const std::vector<std::vector<double> > &A,
    const std::vector<std::vector<double> > &B) {
    const size_t n = A.size();

    std::vector<std::vector<double> > res(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                res[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return res;
}

void ReadMatrix(std::vector<std::vector<double> > &a) {
    const size_t n = a.size();
    for (size_t i = 0; i != n; ++i) {
        for (size_t j = 0; j != n; ++j) {
            std::cin >> a[i][j];
        }
    }
}

std::vector<double> MultiplyMatrixVector(
    const std::vector<std::vector<double> > &A,
    const std::vector<double> &v
) {
    const size_t n = A.size();
    std::vector<double> result(n, 0);

    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[i][j] * v[j];
        }
        result[i] = sum;
    }

    return result;
}

double dotProduct(const std::vector<double> &vec1, const std::vector<double> &vec2) {
    double result = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

std::vector<double> MultiplyVectorMatrix(
    const std::vector<std::vector<double> > &A,
    const std::vector<double> &v
) {
    const size_t n = A.size();
    std::vector<double> result(n, 0);

    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += v[j] * A[j][i];
        }
        result[i] = sum;
    }

    return result;
}

double CalculateVectorMaximumNorm(const std::vector<double> &x) {
    double max = -1;
    const size_t n = x.size();
    for (int i = 0; i < n; i++) {
        if (double value = std::abs(x[i]); value > max) {
            max = value;
        }
    }
    return max;
}

void Print(const std::vector<double> &numbers) {
    const size_t rows = numbers.size();
    for (size_t i = 0; i != rows; ++i) {
        std::cout << numbers[i];
        std::cout << "\n";
    }
}

std::vector<double> CalculateResidualVector(const std::vector<std::vector<double> > &a,
                                            const std::vector<double> &x,
                                            const double lambda) {
    const size_t n = x.size();
    std::vector<double> residual(x.size());

    residual = MultiplyMatrixVector(a, x);

    for (int i = 0; i < n; ++i) {
        residual[i] -= lambda * x[i];
    }
    return residual;
}

void transform(
    std::vector<std::vector<double> > &A,
    const size_t k
) {
    const std::vector transformer = generateTransformationMatrix(A, k);
    const std::vector inversedTransformer = FindInverseMatrix(transformer);
    A = MultiplySquareMatrices(inversedTransformer, A);
    A = MultiplySquareMatrices(A, transformer);
}

void Print(const std::vector<std::vector<double> > &numbers) {
    const size_t rows = numbers.size();
    const size_t columns = numbers[0].size();
    for (size_t i = 0; i != rows; ++i) {
        for (size_t j = 0; j != columns; ++j) {
            if (j > 0) {
                std::cout << "\t";
            }
            std::cout << numbers[i][j];
        }
        std::cout << "\n";
    }
}

std::vector<double> FindEigenPolynome(std::vector<std::vector<double>>& A) {
    for (int i = A.size() - 2; i > -1; i--) {
        transform(A, i);
    }
    return A[0];
}

std::vector<double> FindXmax(double lmax, size_t n) {
    std::vector<double> xmax(n, 0);
    for (int i = 0; i < n; i++) {
        xmax[i] = std::pow(lmax, n - 1 - i);
    }
    return xmax;
}

double calculatePolynomeValue(std::vector<double>& vec, double value) {
    const size_t n = vec.size();
    double res = std::pow(value, n);
    for (int i = n - 1; i > -1; i--) {
        res -= vec[n - 1 - i] * std::pow(value, i);
    }
    return res;
}

double trace(const std::vector<std::vector<double>>& A) {
    double result = 0;
    for (int i = 0; i < A.size(); ++i) {
        result += A[i][i];
    }
    return result;
}

double determinant(std::vector<std::vector<double>> A) {
    double det = 1.0;
    const size_t n = A.size();
    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int j = i + 1; j < n; j++) {
            if (abs(A[j][i]) > abs(A[pivot][i])) {
                pivot = j;
            }
        }
        if (pivot != i) {
            std::swap(A[i], A[pivot]);
            det *= -1;
        }
        if (A[i][i] == 0) {
            return 0;
        }
        det *= A[i][i];
        for (int j = i + 1; j < n; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i + 1; k < n; k++) {
                A[j][k] -= factor * A[i][k];
            }
        }
    }
    return det;
}

int main() {
    size_t n;
    std::cin >> n;

    std::vector a(n, std::vector<double>(n));
    ReadMatrix(a);

    std::vector a_transposed = TransposeSquareMatrix(a);

    std::vector a_t_by_a = MultiplySquareMatrices(a_transposed, a);

    std::vector<double> eigen = FindEigenPolynome(a_t_by_a);
    std::cout << "Eigen polynome: " << "\n\n";
    Print(eigen);

    std::cout << "\nMatrix: " << "\n\n";
    Print(a_t_by_a);
    //lmax: 1.48755

    std::cout << "\nfi_1: " << eigen[0] - trace(a_t_by_a) << "\n\n";
    std::cout << "\nfi_2: " << eigen[n - 1] - determinant(a_t_by_a) << "\n\n";

    std::cout << "\npsi: " << calculatePolynomeValue(eigen, 1.48751) << "\n\n";

    std::cout << "\nEigen vector: " << "\n\n";
    std::vector<double> xmax = FindXmax(1.48755, n);
    Print(xmax);

    const std::vector<double> residual = CalculateResidualVector(a_t_by_a, xmax, 1.48755);
    std::cout << "\nResidual vector: " << "\n\n";
    Print(residual);
}
