#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <random>       // for random_device

typedef long long int ll;

unsigned int xorshift32(unsigned int x) {
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
    int per_cnt[world_size];
    ll per_p = tosses / world_size, cnt = 0;
    double x, y;
    unsigned int seed = time(NULL) * world_rank;

    if (world_rank > 0)
    {
        // TODO: handle workers
        // main loop
        for(ll i = 0; i < per_p; ++i) {
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            if ((x * x + y * y) <= 1.0) {
                cnt++;
            }
        }
        MPI_Send(&cnt, 1, MPI_LONG_LONG, 0, 1, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: master
        for(ll i = 0; i < per_p; ++i) {
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            if ((x * x + y * y) <= 1.0) {
                cnt++;
            }
        }
        per_cnt[0] = cnt;
        for(long long int i = 1; i < world_size; ++i) {
            MPI_Recv(&per_cnt[i], world_size, MPI_LONG_LONG, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        ll total = 0;
        for(ll i = 0; i < world_size; ++i) {
            total += per_cnt[i];
        }
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
