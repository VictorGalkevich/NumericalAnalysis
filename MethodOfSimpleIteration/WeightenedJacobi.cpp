#include <iostream>
#include <vector>

int iterations = 0;

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

void CreateBMatrix(
  const std::vector<std::vector<double>>& A,
  std::vector<std::vector<double>>& B,
  const std::vector<double>& f,
  std::vector<double>& b
  )
{
  const size_t n = A.size();
  const double norm = CalculateMatrixRowNorm(A);

  for (int i = 0; i < n; i++) {
    B[i][i] = 1;
    for (int j = 0; j < n; j++) {
      B[i][j] += -A[i][j] / norm;
    }
    b[i] = f[i] / norm;
  }
}

double CalculateVectorMaximumNorm(std::vector<double>& x, std::vector<double>& y) {
  double max = -1;
  int n = x.size();
  for (int i = 0; i < n; i++) {
    double value = std::abs(x[i] - y[i]);
    if (value > max) {
      max = value;
    }
  }
  return max;
}

void Print(const std::vector<double>& numbers) {
  const size_t rows = numbers.size();
  for (size_t i = 0; i != rows; ++i) {
    std::cout << numbers[i];
    std::cout << "\n";
  }
}

void ReadLinearSystem(std::vector<std::vector<double> >& a, std::vector<double>& b) {
  const size_t n = a.size();
  for (size_t i = 0; i != n; ++i) {
    for (size_t j = 0; j != n; ++j) {
      std::cin >> a[i][j];
    }
    std::cin >> b[i];
  }
}

void Print(const std::vector<std::vector<double> >& numbers) {
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

std::vector<double> CalculateResidualVector(const std::vector<std::vector<double> >& a,
  const std::vector<double>& b,
  const std::vector<double>& x) {
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
  const std::vector<std::vector<double>>& B,
  const std::vector<double>& b,
  const double precision,
  const std::vector<double>& start
)
{
  const size_t n = B.size();

  std::vector<double> xk = start;
  std::vector<double> xk_1 = start;

  double normValue;

  do {
    iterations++;
    xk = xk_1;
    for (int i = 0; i < n; i++) {
      double tmp = b[i];
      for (int j = 0; j < n; j++) {
        tmp += B[i][j] * xk[j];
      }
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

  const std::vector<std::vector<double> > a_transposed = TransposeSquareMatrix(a);
  const std::vector<std::vector<double> > a_transposed_by_a = MultiplySquareMatrices(a_transposed, a);

  const std::vector<double> a_transposed_by_f = MultiplyMatrixVector(a_transposed, f);


  constexpr double precision = 1e-5;

  std::vector<std::vector<double> > B(n, std::vector<double>(n, 0));
  std::vector<double> b(n);

  CreateBMatrix(a_transposed_by_a, B, a_transposed_by_f, b);

  std::cout << "Vector b: " << "\n";
  Print(b);

  std::cout << "Matrix B: " << "\n";
  Print(B);

  std::cout << "Matrix A_t*A: " << "\n";
  Print(a_transposed_by_a);

  std::cout << "Vector a_t*f: " << "\n";
  Print(a_transposed_by_f);

  const std::vector<double> x = SimpleIteration(B, b, precision, a_transposed_by_f);

  std::cout << "Vector X: " << "\n";
  Print(x);

  const std::vector<double> residual = CalculateResidualVector(a, f, x);
  std::cout << "Residual vector: " << "\n";
  Print(residual);

  std::cout << "Number of iterations: " << iterations << "\n";

  const double norm = CalculateMatrixRowNorm(a_transposed_by_a);
  std::cout << "||A||: " << norm << "\n";

  const double bnorm = CalculateMatrixRowNorm(B);
  std::cout << "||B||: " << bnorm << "\n";
}