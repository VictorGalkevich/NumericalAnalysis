#include <iostream>
#include <vector>

int iterations = 0;

void CreateJacobiMatrix(
  std::vector<std::vector<double>>& A,
  std::vector<std::vector<double>>& B,
  std::vector<double>& f,
  std::vector<double>& b
  )
{
  int n = A.size();
  for (int i = 0; i < n; i++) {
    double aii = A[i][i];
    for (int j = 0; j < n; j++) {
      B[i][j] = -A[i][j] / aii;
    }
    B[i][i] = 0;
    b[i] = f[i] / aii;
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
  std::vector<std::vector<double>>& B,
  std::vector<double>& b,
  double precision
)
{
  int n = B.size();

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

  double precision = 1e-5;

  std::vector<std::vector<double> > B(n, std::vector<double>(n));
  std::vector<double> b(n);

  CreateJacobiMatrix(a, B, f, b);

  std::cout << "Vector b: " << "\n";
  Print(b);

  std::cout << "Matrix B: " << "\n";
  Print(B);

  std::vector<double> x = SimpleIteration(B, b, precision);

  std::cout << "Vector X: " << "\n";
  Print(x);

  std::vector<double> residual = CalculateResidualVector(a, f, x);
  std::cout << "Residual vector: " << "\n";
  Print(residual);

  std::cout << "Number of iterations: " << iterations << "\n";
}