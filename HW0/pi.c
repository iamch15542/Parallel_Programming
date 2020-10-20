#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

// define global variable
long long ans = 0;

// define mutex && initial
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// define thread variable    
pthread_t t[8];

struct PARAMS {
    int a; // which thread
    long long b; // per thread need to run
    long long c; // record point num
};

// count
void* cnt_thread(void* data) {
    PARAMS *para;
    para = (PARAMS *) data;
    int num = para->a;
    long long per_p = para->b, cnt = 0;
    double x, y;
    unsigned int seed = time(NULL) + num;

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
    para->c = cnt;
    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}


int main() {

    // initial
    int thread_num = 0;
    long long point_number = 0;
    struct PARAMS params[8];

    // input param
    scanf("%d", &thread_num);
    scanf("%lld", &point_number);

    // per thread need to run
    long per_point = point_number / thread_num;

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

    // print thread
    for(int i = 0; i < thread_num; ++i) {
        printf("Thread %d, There are %lld points in the circle\n", i, params[i].c);
    }

    // print answer
    printf("Pi : %.10lf\n", 4.0 * (double)ans / (double)point_number);
}