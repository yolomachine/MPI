#include <mpi.h>
#include <iostream>
#include <iomanip>

double f(double x) {
    return x * x * x;
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto a = std::strtod(argv[1], nullptr);
    auto b = std::strtod(argv[2], nullptr);
    auto n = std::strtoull(argv[3], nullptr, 10);
    auto sum = 0., dx = (b - a) / n;

    int h = n / size;
    int m = rank != size - 1 ? h * (rank + 1) : n;

    for (int i = h * rank; i < m; ++i)
        sum += f(a + i * dx);

    if (rank == 0) {
        double temp;
        for (int i = 0; i < size - 1; ++i) {
            MPI_Status status;
            MPI_Recv(&temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            sum += temp;
        }
        std::cout << std::setprecision(9) << sum * dx << std::endl;
    } else
        MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}