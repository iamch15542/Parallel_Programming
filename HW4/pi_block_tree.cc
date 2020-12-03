#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;

    // init variable
    long per_p = tosses / world_size, cnt = 0, recv_cnt;

    // run monte carlo
    double x, y;
    unsigned int seed = time(NULL) * world_rank;
    for(size_t i = 0; i < per_p; ++i) {
        x = (double)rand_r(&seed) / RAND_MAX;
        y = (double)rand_r(&seed) / RAND_MAX;
        if ((x * x + y * y) <= 1.0) {
            cnt++;
        }
    }

    // TODO: binary tree redunction
    int active = 1;
    int log_size = (int)log2(world_size);

    for(int k = 0; k < log_size; ++k) {
        if (active) {
            //if bit k is set in rank
            if ((1 << k) & world_rank) {
                MPI_Send(&cnt, 1, MPI_LONG, world_rank - ((int) pow(2,k)), 0, MPI_COMM_WORLD);
                active = 0;
            } else {
                MPI_Recv(&recv_cnt, 1, MPI_LONG, world_rank + ((int) pow(2,k)), 0, MPI_COMM_WORLD, &status);
                cnt += recv_cnt;
            }
        }
    }

    if (world_rank == 0) {
        // TODO: PI result
        pi_result = 4 * ((double)cnt / (double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
