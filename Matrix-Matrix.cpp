#include <mpi.h>

#include <iostream>
#include <sstream>
#include <vector>

using std::cout;
using std::cerr;
using std::endl;
using std::ostringstream;
using std::vector;

typedef long int Matrix_Elements_Type;

int main(int argc, char * argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    // A rows
    size_t l = std::strtoul(argv[1], nullptr, 10);

    // B rows -- A cols
    size_t m = std::strtoul(argv[2], nullptr, 10);

    // B cols
    size_t n = std::strtoul(argv[3], nullptr, 10);
    MPI_Datatype MPI_Matrix_Elements_Type = MPI_LONG;
    vector<Matrix_Elements_Type> A;
    vector<Matrix_Elements_Type> B;
    vector<Matrix_Elements_Type> C;

    const double start_time = MPI_Wtime();

    if (rank == 0) {

        if (l % size != 0) {
            cerr << "Number of rows in matrix A is not aliquot to MPI_Comm_size" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (m % size != 0) {
            cerr << "Number of rows in matrix B is not aliquot to MPI_Comm_size" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (l != n) {
            cerr << "Number of rows in matrix A is not equal to number of cols in matrix B" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        A.resize(l * m);
        B.resize(m * n);
        C.resize(l * n);

        cout << "A: \n";
        for (size_t i = 0; i < l; ++i) {
            for (size_t j = 0; j < m; ++j) {
                A[i * m + j] = (i + 1) * (j + 1);
                cout << i * j << " ";
            }
            cout << endl;
        }
        cout << endl;

        cout << "B: \n";
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                B[i * n + j] = 2 + i;
                cout << 2 + i << " ";
            }
            cout << endl;
        }
        cout << endl;


        vector<Matrix_Elements_Type> tmp(n * m);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                tmp[i * m + j] = B[j * n + i];
            }
        }
        B = tmp;
    }

    const size_t proc_rows = l / size;
    vector<Matrix_Elements_Type> a(proc_rows * m);
    MPI_Scatter(A.data(), proc_rows * m, MPI_Matrix_Elements_Type, a.data(), proc_rows * m, MPI_Matrix_Elements_Type,
                0, MPI_COMM_WORLD);

    const size_t proc_cols = n / size;
    vector<Matrix_Elements_Type> b(m * proc_cols);
    MPI_Scatter(B.data(), proc_cols * m, MPI_Matrix_Elements_Type, b.data(), proc_cols * m, MPI_Matrix_Elements_Type,
                0, MPI_COMM_WORLD);

    vector<Matrix_Elements_Type> c(proc_rows * n, 0);
    for (int p = 0; p < size; ++p) {

        cout << endl;
        for (size_t i = 0; i < proc_rows; ++i)
            for (size_t j = 0; j < proc_cols; ++j)
                for (size_t k = 0; k < m; ++k) {
                    c[i * n + j + ((rank + p) * proc_cols % n)] += a[i * m + k] * b[j * m + k];
                }

        MPI_Sendrecv_replace(b.data(), proc_cols * m, MPI_Matrix_Elements_Type, ((rank - 1) % size + size) % size,
                             0, (rank + 1) % size, 0, MPI_COMM_WORLD, &status);
    }

    MPI_Gather(c.data(), proc_rows * n, MPI_Matrix_Elements_Type, C.data(), proc_rows * n, MPI_Matrix_Elements_Type,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < l; ++i) {
            for (int j = 0; j < n; ++j)
                cout << C[i * n + j] << " ";
            cout << endl;
        }
        cout << "\nElapsed time: " << MPI_Wtime() - start_time << "s" << endl;
    }

    MPI_Finalize();
    return 0;
}