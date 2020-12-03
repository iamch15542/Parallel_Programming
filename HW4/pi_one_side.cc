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

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // init variable
    long per_p = tosses / world_size, cnt = 0;
    long *recv = (long*) malloc(world_size * sizeof(long));
    MPI_Win_create(recv, world_size * sizeof(long), sizeof(long), MPI_INFO_NULL,MPI_COMM_WORLD, &win);

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

    if(world_rank == 0) {
        // Master
        MPI_Win_fence(0, win);
        MPI_Put(&cnt, 1, MPI_LONG, 0, world_rank, 1, MPI_LONG, win);
        MPI_Win_fence(0, win);
    } else {
        // Workers
        MPI_Win_fence(0, win);
        MPI_Put(&cnt, 1, MPI_LONG, 0, world_rank, 1, MPI_LONG, win);
        MPI_Win_fence(0, win);
    }

    MPI_Win_free(&win);

    if(world_rank == 0) {
        // TODO: handle PI result
        for(int i = 1; i < world_size; ++i) cnt += recv[i];
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