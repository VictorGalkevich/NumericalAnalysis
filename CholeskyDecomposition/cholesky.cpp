//
// Created by Victor Galkevich on 11/15/24.
//
#include <cmath>
#include <vector>
#include <iostream>

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

std::vector<std::vector<double> > BuildLowerTrinagular(const std::vector<std::vector<double> > &initial) {
    const size_t n = initial.size();
    std::vector<std::vector<double> > lowerTriangular(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        double temp;

        for (int j = 0; j < i; j++) {
            temp = 0;
            for (int k = 0; k < j; k++) {
                temp += lowerTriangular[i][k] * lowerTriangular[j][k];
            }
            lowerTriangular[i][j] = (initial[i][j] - temp) / lowerTriangular[j][j];
        }

        temp = initial[i][i];
        for (int k = 0; k < i; k++) {
            temp -= lowerTriangular[i][k] * lowerTriangular[i][k];
        }
        lowerTriangular[i][i] = std::sqrt(temp);
    }

    return lowerTriangular;
}

void Print(const std::vector<double> &numbers) {
    const size_t rows = numbers.size();
    for (size_t i = 0; i != rows; ++i) {
        std::cout << numbers[i];
        std::cout << "\n";
    }
}

void ReadLinearSystem(std::vector<std::vector<double> > &a, std::vector<double> &b) {
    const size_t n = a.size();
    for (size_t i = 0; i != n; ++i) {
        for (size_t j = 0; j != n; ++j) {
            std::cin >> a[i][j];
        }
        std::cin >> b[i];
    }
}

std::vector<double> solve(const std::vector<std::vector<double> > &lowerTrinagular,
                          const std::vector<double> &b) {
    const size_t n = b.size();
    std::vector<double> y(n);
    for (size_t i = 0; i < n; i++) {
        y[i] = b[i];

        for (size_t j = 0; j < i; ++j) {
            y[i] -= lowerTrinagular[i][j] * y[j];
        }

        y[i] /= lowerTrinagular[i][i];
    }

    const std::vector<std::vector<double> > transposed = TransposeSquareMatrix(lowerTrinagular);
    std::vector<double> x(n);
    for (int32_t i = n - 1; i > -1; i--) {
        x[i] = y[i];

        for (size_t j = i + 1; j < n; ++j) {
            x[i] -= transposed[i][j] * x[j];
        }

        x[i] /= transposed[i][i];
    }
    return x;
}

std::vector<std::vector<double> > MultiplyTwoMatrices(const std::vector<std::vector<double> > &first,
                                                      const std::vector<std::vector<double> > &second) {
    const size_t n = first.size();
    std::vector<std::vector<double> > result(n, std::vector<double>(n));

    for (size_t i = 0; i != n; i++) {
        for (size_t j = 0; j != n; j++) {
            result[i][j] = 0;

            for (size_t k = 0; k != n; k++) {
                result[i][j] += first[i][k] * second[k][j];
            }
        }
    }
    return result;
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

std::vector<double> CalculateResidualVector(const std::vector<std::vector<double> > &a,
                                            const std::vector<double> &b,
                                            const std::vector<double> &x) {
    const size_t n = x.size();
    std::vector<double> residual(x.size());

    for (size_t i = 0; i != n; ++i) {
        for (size_t j = 0; j != n; ++j) {
            residual[i] += a[i][j] * x[j];
        }
        residual[i] -= b[i];
    }
    return residual;
}

int main() {
    size_t n;
    std::cin >> n;

    std::vector<std::vector<double> > a(n, std::vector<double>(n));
    std::vector<double> b(n);

    ReadLinearSystem(a, b);

    const auto transposed = TransposeSquareMatrix(a);
    std::cout << "Transposed Matrix: " << "\n";
    Print(transposed);
    const auto symmetric = MultiplyTwoMatrices(transposed, a);
    std::cout << "Symmetric Matrix: " << "\n";
    Print(symmetric);

    const auto lower = BuildLowerTrinagular(symmetric);
    std::cout << "Matrix L: " << "\n";
    Print(lower);
    const auto x = solve(lower, b);
    std::cout << "X vector: " << "\n";
    Print(x);
    const auto residual = CalculateResidualVector(symmetric, b, x);
    std::cout << "Residual vector: " << "\n";
    Print(residual);
}
