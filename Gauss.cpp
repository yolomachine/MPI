#include <iostream>
#include <mpi.h>
#include <random>
#include <numeric>
#include <exception>
#include <algorithm>

using namespace std;

int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    const double start_time = MPI_Wtime();
    size_t n = strtoul(argv[1], NULL, 10);

    const size_t m = n / size;
    double* matrix_a = NULL;
    size_t* order = NULL;

    if (rank == 0) {

        matrix_a = new double[n * (n + 1)];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n + 1; ++j) {
                matrix_a[i * (n + 1) + j] = rand() % 5;
            }
        }

        order = new size_t[n];
        memset(order, 0, n * sizeof(size_t));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n + 1; ++j) {
                cout << matrix_a[i * (n + 1) + j] << ' ';
            }
            cout << endl;
        }
        cout << endl;

    }

    auto* rows = new double[m * (n + 1)];
    auto* used = new bool[n];
    memset(used, 0, n * sizeof(bool));

    MPI_Scatter(matrix_a, m * (n + 1), MPI_DOUBLE, rows, m * (n + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < n; ++i) {

        double max = 0;
        int arg_max = -1;

        for (size_t j = 0; j < m; ++j) {
            if (!used[j + m * rank] && abs(rows[j * (n + 1) + i]) >= max) {
                max = abs(rows[j * (n + 1) + i]);
                arg_max = static_cast<int>(j);
            }
        }

        auto* max_row = new double[n + 1];
        if (rank != 0) {
            MPI_Send(&arg_max, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            if (arg_max >= 0) {
                MPI_Send(&rows[arg_max * (n + 1)], n + 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        } else {
            if (arg_max >= 0) {
                copy(rows + arg_max * (n + 1), rows + (arg_max + 1) * (n + 1), max_row);
            }

            for (size_t j = 1; j < size; ++j) {
                int arg_maxx;
                auto* data = new double[n + 1];

                MPI_Recv(&arg_maxx, 1, MPI_INT, j, 0, MPI_COMM_WORLD, &status);

                if (arg_maxx >= 0) {
                    arg_maxx += m * j;
                    MPI_Recv(data, n + 1, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, &status);

                    if (abs(data[i]) >= max) {
                        arg_max = arg_maxx;
                        max = abs(data[i]);
                        copy(data, data + n + 1, max_row);
                    }
                }
                delete[] data;
            }
            order[i] = static_cast<size_t>(arg_max);
        }

        MPI_Bcast(&arg_max, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(max_row, n + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        used[arg_max] = true;
        for (size_t j = 0; j < m; ++j) {
            if (used[j + rank * m]) {
                continue;
            }

            auto &temp = rows[j * (n + 1) + i];
            for (size_t k = i + 1; k < n + 1; ++k) {
                rows[j * (n + 1) + k] -= max_row[k] * temp / max_row[i];
            }

            temp = 0;
        }
    }

    MPI_Gather(rows, m * (n + 1), MPI_DOUBLE, rank != 0 ? nullptr : matrix_a, rank != 0 ? 1 : m*(n + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n + 1; ++j)
                std::cout << matrix_a[i * (n + 1) + j] << ' ';
            std::cout << std::endl;
        }
        std::cout << std::endl;

        for (size_t i = 0; i < n; ++i) {
            if (!used[i]) {
                order[n - 1] = i;
                break;
            }
        }

        memset(used, 0, n * sizeof(bool));
        for (size_t i = 0; i < n; ++i) {
            const auto ind = n - i - 1;
            const auto it = order[ind];
            matrix_a[it * (n + 1) + n] /= matrix_a[it * (n + 1) + ind];
            matrix_a[it * (n + 1) + ind] = 1;
            used[it] = true;

            for (size_t j = 0; j < n; ++j) {
                if (used[j]) {
                    continue;
                }

                matrix_a[j * (n + 1) + n] -= matrix_a[j * (n + 1) + ind] * matrix_a[it * (n + 1) + n];
                matrix_a[j * (n + 1) + ind] = 0;
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n + 1; ++j) {
                std::cout << matrix_a[i * (n + 1) + j] << ' ';
            }

            std::cout << std::endl;
        }

        std::cout << std::endl;

        cout << MPI_Wtime() - start_time << endl;
    }
}
