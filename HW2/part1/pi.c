#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

// define global variable
long long int ans = 0;

// define mutex && initial
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

struct PARAMS {
    unsigned int a; // seed value
    long long b;    // per thread need to run
};

// count
void* cnt_thread(void* data) {
    PARAMS *para;
    para = (PARAMS *) data;
    long long int per_p = para->b, cnt = 0;
    double x, y;
    unsigned int seed = para->a;

    for(long long i = 0; i < per_p; ++i) {
        x = (double)rand_r(&seed) / RAND_MAX;
        y = (double)rand_r(&seed) / RAND_MAX;
        if ((x * x + y * y) <= 1.0) {
            cnt++;
        }
    }

    // write cnt into sum
    pthread_mutex_lock(&mutex);
    ans += cnt;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}


int main(int argc, char *argv[]) {

    // initial
    int thread_num = atoi(argv[1]);                // the number of CPU cores
    long long int point_number = atoll(argv[2]);   // the number of tosses

    // define thread variable
    pthread_t t[thread_num];
    struct PARAMS params[thread_num];

    // per thread need to run
    long long int per_point = point_number / thread_num;

    // get random seed
    unsigned int random_data[thread_num];
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd > 0) {
        ssize_t result = read(fd, random_data, sizeof(random_data));
        if(result < 0) printf("Fail\n");
    }
    close(fd);

    // creating thread_num threads 
    for(int i = 0; i < thread_num; ++i) {
        params[i].a = random_data[i];
        params[i].b = per_point;
        pthread_create(&t[i], NULL, cnt_thread, (void*)&params[i]);
    }

    // wait for thread
    for(int i = 0; i < thread_num; ++i) {
        pthread_join(t[i], NULL);
    }

    // print answer
    printf("%.10lf\n", 4.0 * (double)ans / (double)point_number);
}