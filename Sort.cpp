#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstddef>
#include <algorithm>
#include <cmath>

using namespace std;

int main(int argc, char * argv[]) {
    srand(42);
    int size, rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    const double start_time = MPI_Wtime();
    size_t n = strtoul(argv[1], NULL, 10);

    double *data = NULL;
    int *lengths = NULL;
    int *displ = NULL;
    double minn = RAND_MAX;
    double maxx = 0;
    double step;

    if (rank == 0) {
        lengths = new int[size];
        displ = new int[size];
        data = new double[n];

        for (int i = 0; i < n; i++) {
            data[i] = static_cast<double>(rand());
            minn = min(data[i], minn);
            maxx = max(maxx, data[i]);
            //cout << data[i] << ' ';
        }

        cout << '\n';

        step = (maxx - minn) / size;
        vector< vector<double> > bins(size);

        for (int i = 0; i < n; i++) {
            size_t index = static_cast<size_t>(floor(data[i] / step));
            if (index == bins.size()) {
                --index;
            }
            bins[index].push_back(data[i]);
        }

        int index = 0;
        for (int i = 0; i < bins.size(); i++) {
            lengths[i] = bins[i].size();
            displ[i] = i >= 1 ? displ[i - 1] + lengths[i - 1] : 0;

            for (int j = 0; j < bins[i].size(); j++) {
                data[index++] = bins[i][j];
            }
        }
    }

    int part_size;
    MPI_Scatter(lengths, 1, MPI_INT, &part_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double* bucket = new double[part_size];
    MPI_Scatterv(data, lengths, displ, MPI_DOUBLE, bucket, part_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    sort(bucket, bucket + part_size);

    MPI_Gatherv(bucket, part_size, MPI_DOUBLE, data, lengths, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Elapsed time: " << MPI_Wtime() - start_time << "s" << endl;
    }

    MPI_Finalize();
    return 0;
}