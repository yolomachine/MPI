#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = std::strtoul(argv[1], nullptr, 10);
    size_t m = std::strtoul(argv[2], nullptr, 10);
    const int rows_amount = n / size;
    const long send_count = rows_amount * m;

    std::vector<int> matrix(n * m);
    std::vector<int> rows(send_count);
    std::vector<int> vec(m);
    std::vector<int> local_res(rows_amount, 0);
    std::vector<int> res(n);

    const double start_time = MPI_Wtime();

    if (n % size != 0 || size > n ) {
        if (rank == 0) {
            std::ostringstream s;
            s << "\n=====================================\n"
              << "  N is not aliquot to MPI_Comm_size"
              << "\n=====================================\n";
            std::cerr << s.str();
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                matrix[i * m + j] = 1;

        for (int j = 0; j < m; ++j)
            vec[j] = j + 1;
    }

    MPI_Scatter(matrix.data(), send_count, MPI_INT, rows.data(), send_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vec.data(), m, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_amount; ++i)
        for (int j = 0; j < m; ++j)
            local_res[i] += rows[i*m + j] * vec[j];

    MPI_Gather(local_res.data(), rows_amount, MPI_INT, res.data(), rows_amount, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::ostringstream s;
        //for (auto i: res)
        //    s << i << "\n";
        s << *res.begin() << "\nVector size: " << res.size() << "\nElapsed time: " << MPI_Wtime() - start_time << " seconds\n";
        std::cout << s.str();
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}