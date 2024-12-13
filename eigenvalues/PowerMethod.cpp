#include <complex>
#include <iostream>
#include <vector>

#include "../../../../opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current/gcc/aarch64-apple-darwin23/14/include-fixed/math.h"

size_t iterations = 0;

std::vector<std::vector<double> > TransposeSquareMatrix(const std::vector<std::vector<double> > &initial) {
    const size_t n = initial.size();
    std::vector<std::vector<double> > transposed(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transposed[j][i] = initial[i][j];
        }
    }
    return transposed;
}

std::vector<std::vector<double>> MultiplySquareMatrices(
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

void normalize(std::vector<double> &vec) {
    const double norm = CalculateVectorMaximumNorm(vec);
    for (double & i : vec) {
        i /= norm;
    }
}

bool check(
    const std::vector<double> &y_k,
    const std::vector<double> &y_k1,
    const std::vector<double> &y_k2,
    double precision
) {
    double max = -INFINITY;
    for (int i = 0; i < y_k1.size(); ++i) {
        if (const double value = std::fabs(std::fabs(y_k2[i] / y_k1[i]) - std::fabs(y_k1[i]/y_k[i])); value > max) {
            max = value;
        }
    }
    return max <= precision;
}

double Iterate(
    const std::vector<std::vector<double> > &A,
    std::vector<double> &start,
    const double precision
) {
    const size_t n = start.size();
    std::vector<double> y_k(n);
    std::vector<double> y_k1 = start;
    std::vector<double> y_k2(n);
    std::vector<double> ycurr(n);
    std::vector<double> yprev(n);

    y_k2 = MultiplyMatrixVector(A, y_k1);
    ycurr = y_k2;
    normalize(y_k2);

    do {
        yprev = ycurr;
        iterations++;
        y_k = y_k1;
        y_k1 = y_k2;
        y_k2 = MultiplyMatrixVector(A, y_k1);
        ycurr = y_k2;
        normalize(y_k2);
    } while (!check(y_k, y_k1, y_k2, precision));

    start = ycurr;

    return ycurr[0] / (yprev[0] / CalculateVectorMaximumNorm(yprev));
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

int main() {
    size_t n;
    std::cin >> n;

    std::vector a(n, std::vector<double>(n));

    std::vector<double> y_0(n, 0);
    y_0[0] = 1;
    ReadMatrix(a);

    std::vector a_transposed = TransposeSquareMatrix(a);

    const std::vector a_t_by_a = MultiplySquareMatrices(a_transposed, a);

    constexpr double precision = 1e-10;

    std::cout << "Transposed: " << "\n\n";
    Print(a_t_by_a);
    const double lmax = Iterate(a_t_by_a, y_0, precision);

    std::cout << "\n\nLambda_max: " << lmax << "\n\n";

    const std::vector<double> residual = CalculateResidualVector(a_t_by_a, y_0, lmax);
    std::cout << "Residual vector: " << "\n\n";
    Print(residual);

    std::cout << "\nXmax Vector: " << "\n\n";
    Print(y_0);

    std::cout << "\nIterations: ";
    std::cout << iterations;

}
