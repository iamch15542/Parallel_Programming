#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <random>       // for random_device

unsigned int xorshift32(unsigned int& x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // init variable
    long per_p = tosses / world_size, cnt = 0, recv[world_size];

    // run monte carlo
    std::random_device rd;
    unsigned int rdn = rd();
    long long check = 1073741824;
    for(size_t i = 0; i < per_p; ++i) {
        unsigned int seed = xorshift32(rdn);
        unsigned int x = (seed & 0x7FFF0000) >> 16;
        unsigned int y = (seed & 0x00007FFF);
        if ((x * x + y * y) <= check) {
            cnt++;
        }
    }

    if(world_rank > 0) {
        // TODO: MPI workers
        MPI_Send(&cnt, 1, MPI_LONG, 0, 1, MPI_COMM_WORLD);
    } else if (world_rank == 0) {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        recv[0] = cnt;
        MPI_Request requests[world_size];
        for(int i = 1; i < world_size; ++i) {
            MPI_Irecv(&recv[i], 1, MPI_LONG, i, 1, MPI_COMM_WORLD, &requests[i]);
        }

        MPI_Waitall(world_size - 1, &requests[1], MPI_STATUSES_IGNORE);
    }

    if(world_rank == 0) {
        // TODO: PI result
        long total = 0;
        for(int i = 0; i < world_size; ++i) total += recv[i];
        pi_result = 4 * ((double)total / (double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
