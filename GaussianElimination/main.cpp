#include <iostream>
#include <vector>

std::vector<size_t> permutation;
int32_t swapCount = 0;

void Rearrange(const std::vector<double> &x) {
    for (size_t i = 0; i != x.size(); ++i) {
        if (i != permutation[i]) {
            const size_t in1 = permutation[i];
            permutation[i] = permutation[in1];
            permutation[in1] = in1;
        }
    }
}

size_t FindPivotInRow(const std::vector<std::vector<double> > &numbers,
                      const size_t row) {
    size_t pivotIndex = row;

    for (size_t i = row + 1; i < row; ++i) {
        if (std::abs(numbers[row][i]) > std::abs(numbers[row][pivotIndex])) {
            pivotIndex = i;
        }
    }
    return pivotIndex;
}

void SwapColumns(std::vector<std::vector<double> > &numbers,
                 const size_t prev,
                 const size_t best) {
    if (prev == best) {
        return;
    }

    const size_t num_rows = numbers.size();

    for (int row = 0; row < num_rows; row++) {
        const double temp = numbers[row][prev];
        numbers[row][prev] = numbers[row][best];
        numbers[row][best] = temp;
    }

	int tmp = permutation[prev];
	permutation[prev] = permutation[best];
	permutation[best] = tmp;
    swapCount++;
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

void Print(const std::vector<double> &numbers) {
    const size_t rows = numbers.size();
    for (size_t i = 0; i != rows; ++i) {
        std::cout << numbers[i];
        std::cout << "\n";
    }
}

void Forward(std::vector<std::vector<double> > &a,
             std::vector<double> &b) {
    const size_t n = b.size();

    for (size_t k = 0; k < n - 1; k++) {
        const size_t pivotIndex = FindPivotInRow(a, k);
        SwapColumns(a, k, pivotIndex);
        for (size_t i = k + 1; i < n; ++i) {
            const double li = a[i][k] / a[k][k];

            b[i] -= li * b[k];
            for (size_t j = k + 1; j < n; ++j) {
                a[i][j] -= li * a[k][j];
            }

            a[i][k] = 0;
        }
    }
}

std::vector<double> Backward(const std::vector<std::vector<double> > &a,
                             std::vector<double> &b,
                             std::vector<double> &x) {
    const int32_t n = b.size();

    for (int32_t i = n - 1; i > -1; i--) {
        x[i] = b[i];

        for (size_t j = i + 1; j < n; ++j) {
            x[i] -= a[i][j] * x[j];
        }

        x[i] /= a[i][i];
    }
    return b;
}

std::vector<std::vector<double> > GaussianElimination(std::vector<std::vector<double> > a,
                                                      std::vector<double> b,
                                                      std::vector<double> &x) {
    Forward(a, b);
    Backward(a, b, x);
    Rearrange(x);
    for (size_t i = 0; i != a.size(); ++i) {
        permutation[i] = i;
    }
    return a;
}

double CalculateUpperTrinagleMatrixDeterminant(const std::vector<std::vector<double> > &a) {
    double res = 1.0;

    for (size_t i = 0; i != a.size(); ++i) {
        res *= a[i][i];
    }
    if (swapCount % 2 != 0) {
        res *= -1;
    }
    return res;
}

std::vector<double> CalculateResidualVector(const std::vector<std::vector<double> > &a,
                                            const std::vector<double> &b,
                                            const std::vector<double> &x) {
    const int32_t n = x.size();
    std::vector<double> residual(x.size());

    for (size_t i = 0; i != n; ++i) {
        for (size_t j = 0; j != n; ++j) {
            residual[i] += a[i][j] * x[j];
        }
        residual[i] -= b[i];
    }
    return residual;
}

std::vector<std::vector<double> > FindInverseMatrix(const std::vector<std::vector<double> > &a) {
    const int32_t n = a.size();

    std::vector<std::vector<double> > inverse(n, std::vector<double>(n));
    std::vector<double> b(n, 0);
    b[0] = 1;
    std::vector<double> x(n);

    GaussianElimination(a, b, x);
    uint32_t col = 0;
    for (size_t j = 0; j != n; ++j) {
        inverse[j][col] = x[j];
    }
    col++;
    for (size_t i = 1; i != n; ++i) {
        b[i - 1] = 0;
        b[i] = 1;
        GaussianElimination(a, b, x);
        for (size_t j = 0; j != n; ++j) {
            inverse[j][col] = x[j];
        }
        col++;
    }
    return inverse;
}

std::vector<std::vector<double> > CalculateResidualMatrix(const std::vector<std::vector<double> > &a,
                                                          const std::vector<std::vector<double> > &x) {
    const int32_t n = x.size();
    std::vector<std::vector<double> > residual(n, std::vector<double>(n));

    for (size_t i = 0; i != n; i++) {
        for (size_t j = 0; j != n; j++) {
            residual[i][j] = 0;

            for (size_t k = 0; k != n; k++) {
                residual[i][j] += a[i][k] * x[k][j];
            }
        }
        residual[i][i] -= 1;
    }

    return residual;
}

double CalculateMatrixRowNorm(const std::vector<std::vector<double> > &a) {
    double max = 0;
    for (int i = 0; i < a.size(); ++i) {
        max += std::abs(a[0][i]);
    }
    for (int i = 1; i < a.size(); ++i) {
        double tmp = 0;
        for (int j = 0; j < a.size(); ++j) {
            tmp += std::abs(a[0][i]);
        }
        if (tmp > max) {
            max = tmp;
        }
    }
    return max;
}

double CalculateConditionNumber(const std::vector<std::vector<double> > &a,
                                const std::vector<std::vector<double> > &inverse) {
    return CalculateMatrixRowNorm(a) * CalculateMatrixRowNorm(inverse);
}

int main() {
    size_t n;
    std::cin >> n;

    std::vector<std::vector<double> > a(n, std::vector<double>(n));
    std::vector<double> b(n);
    std::vector<double> x(n);

    permutation.resize(n);
    for (size_t i = 0; i != n; ++i) {
        permutation[i] = i;
    }

    for (size_t i = 0; i != n; ++i) {
        for (size_t j = 0; j != n; ++j) {
            std::cin >> a[i][j];
        }
        std::cin >> b[i];
    }

    const std::vector<std::vector<double> > forward = GaussianElimination(a, b, x);

    std::cout << "Matrix A determinant is: " << CalculateUpperTrinagleMatrixDeterminant(forward) << "\n\n";

    std::cout << "X vector: \n\n";
    Print(x);
    std::cout << "\nResidual vector: \n\n";
    const std::vector<double> residual_vector = CalculateResidualVector(a, b, x);
    Print(residual_vector);
    std::cout << "\nInverse matrix: \n\n";
    const std::vector<std::vector<double> > inverse = FindInverseMatrix(a);
    Print(inverse);
    std::cout << "\nResidual matrix: \n\n";
    const std::vector<std::vector<double> > residual_matrix = CalculateResidualMatrix(a, inverse);
    Print(residual_matrix);
    std::cout << "\nCondition number: " << CalculateConditionNumber(a, inverse);
}
