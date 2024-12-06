#include <iostream>
#include <vector>

int iterations = 0;
double bNorm = 0;

std::vector<size_t> permutation;
int32_t swapCount = 0;

double CalculateMatrixRowNorm(const std::vector<std::vector<double> > &a) {
    double max = 0;
    for (int i = 0; i < a.size(); ++i) {
        max += std::abs(a[0][i]);
    }
    for (int i = 1; i < a.size(); ++i) {
        double tmp = 0;
        for (int j = 0; j < a.size(); ++j) {
            tmp += std::abs(a[i][j]);
        }
        if (tmp > max) {
            max = tmp;
        }
    }
    return max;
}

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

    const size_t tmp = permutation[prev];
    permutation[prev] = permutation[best];
    permutation[best] = tmp;
    swapCount++;
}

void generateMatrixB(const std::vector<std::vector<double> > &A,
                     std::vector<std::vector<double> > &B,
                     const std::vector<double> &f,
                     std::vector<double> &b) {
    const size_t n = A.size();
    for (int i = 0; i < n; i++) {
        double aii = A[i][i];
        for (int j = 0; j < n; j++) {
            B[i][j] = -A[i][j] / aii;
        }
        B[i][i] = 0;
        b[i] = f[i] / aii;
    }
}

void generateMatrixL(const std::vector<std::vector<double> > &B,
                     std::vector<std::vector<double> > &L) {
    const size_t n = B.size();
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            L[i][j] = B[i][j];
        }
    }
}

void generateSubtractionMatrix(std::vector<std::vector<double> > &L) {
    const size_t n = L.size();
    for (int i = 0; i < n; i++) {
        L[i][i] = 1;
        for (int j = 0; j < i; j++) {
            L[i][j] = -L[i][j];
        }
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

std::vector<std::vector<double>> FindInverseMatrix(const std::vector<std::vector<double> > &a) {
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

std::vector<std::vector<double>> MultiplySquareMatrices(
    const std::vector<std::vector<double> > &A,
    const std::vector<std::vector<double> > &B) {
    const int32_t n = A.size();

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

std::vector<double> MultiplyMatrixVector(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& v
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

void generateMatrixR(const std::vector<std::vector<double> > &B,
                     std::vector<std::vector<double> > &R) {
    const size_t n = B.size();
    for (int i = n - 2; i > -1; i--) {
        for (int j = n - 1; j > i; j--) {
            R[i][j] = B[i][j];
        }
    }
}



void CreateGauusSeidelMatrix(
    const std::vector<std::vector<double> > &A,
    std::vector<std::vector<double> > &Bs,
    const std::vector<double> &f,
    std::vector<double> &b
) {
    generateMatrixB(A, Bs, f, b);
    bNorm = CalculateMatrixRowNorm(Bs);
    const size_t n = A.size();
    std::vector<std::vector<double> > L(n, std::vector<double>(n, 0));
    std::vector<std::vector<double> > R(n, std::vector<double>(n, 0));
    generateMatrixL(Bs, L);
    generateMatrixR(Bs, R);
    generateSubtractionMatrix(L);
    const std::vector<std::vector<double> > inverse = FindInverseMatrix(L);
    Bs = MultiplySquareMatrices(inverse, R);
    b = MultiplyMatrixVector(inverse, b);
}


double CalculateVectorMaximumNorm(const std::vector<double> &x, const std::vector<double> &y) {
    double max = -1;
    const size_t n = x.size();
    for (int i = 0; i < n; i++) {
        if (const double value = std::abs(x[i] - y[i]); value > max) {
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

void ReadLinearSystem(std::vector<std::vector<double> > &a, std::vector<double> &b) {
    const size_t n = a.size();
    for (size_t i = 0; i != n; ++i) {
        for (size_t j = 0; j != n; ++j) {
            std::cin >> a[i][j];
        }
        std::cin >> b[i];
    }
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

std::vector<double> SimpleIteration(
    const std::vector<std::vector<double> > &B,
    const std::vector<double> &b,
    const double precision
) {
    const int n = B.size();

    std::vector<double> xk = b;
    std::vector<double> xk_1 = b;

    double normValue;

    do {
        iterations++;
        xk = xk_1;
        for (int i = 0; i < n; i++) {
            double tmp = 0;
            for (int j = 0; j < n; j++) {
                tmp += B[i][j] * xk[j];
            }
            tmp += b[i];
            xk_1[i] = tmp;
        }
        normValue = CalculateVectorMaximumNorm(xk, xk_1);
    } while (normValue > precision);

    return xk_1;
}


int main() {
    size_t n;
    std::cin >> n;

    std::vector<std::vector<double> > a(n, std::vector<double>(n));
    std::vector<double> f(n);

    ReadLinearSystem(a, f);

    constexpr double precision = 1e-5;

    std::vector<std::vector<double> > B(n, std::vector<double>(n));
    std::vector<double> b(n);

    permutation.resize(n);
    for (size_t i = 0; i != n; ++i) {
        permutation[i] = i;
    }

    CreateGauusSeidelMatrix(a, B, f, b);

    std::cout << "Vector b: " << "\n";
    Print(b);

    std::cout << "Matrix B: " << "\n";
    Print(B);

    const std::vector<double> x = SimpleIteration(B, b, precision);

    std::cout << "Vector X: " << "\n";
    Print(x);

    const std::vector<double> residual = CalculateResidualVector(a, f, x);
    std::cout << "Residual vector: " << "\n";
    Print(residual);

    std::cout << "Number of iterations: " << iterations << "\n";

    std::cout << "q: " << bNorm << "\n";
}
