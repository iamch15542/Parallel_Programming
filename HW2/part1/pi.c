#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

// define global variable
long long ans = 0;

// define mutex && initial
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// define thread variable
pthread_t t[MAX_THREAD];
long long int random_data[MAX_THREAD];

struct PARAMS {
    int a;           // which thread
    long long int b; // per thread need to run
    long long int c; // record point num
};

// count
void* cnt_thread(void* data) {
    PARAMS *para;
    para = (PARAMS *) data;
    int num = para->a;
    long long int per_p = para->b, cnt = 0;
    double x, y;
    unsigned int seed = random_data[num];

    for(long long int i = 0; i < per_p; ++i) {
        x = (double)rand_r(&seed) / RAND_MAX;
        y = (double)rand_r(&seed) / RAND_MAX;
        if ((x * x + y * y) <= 1.0) {
            cnt++;
        }
    }

    // write cnt into sum
    pthread_mutex_lock(&mutex);
    ans += cnt;
    para->c = cnt;
    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}


int main(int argc, char *argv[]) {

    // initial
    int thread_num = atoi(argv[1]);           // the number of CPU cores
    long long int point_number = atoi(argv[2]);   // the number of tosses
    struct PARAMS params[thread_num];

    int fd = open("/dev/urandom", O_RDONLY);
    if (fd > 0) {
        ssize_t result = read(fd, random_data, sizeof(random_data));
        if(result < 0) printf("Fail\n");
    }
    close(fd);

    // per thread need to run
    long long int per_point = point_number / thread_num;

    // creating thread_num threads 
    for(int i = 0; i < thread_num; ++i) {
        params[i].a = i;
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