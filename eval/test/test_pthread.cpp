/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pthread.h>  
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <vector>

pthread_mutex_t mutex;

void *print_msg(void *arg)
{  
    int num = *(int *)arg;
    int i=0;  
    pthread_mutex_lock(&mutex);
    // sleep(2);
    for(i=0;i<16;i++)
    {  
        printf("output : %d", num);
    }
    printf("\n");
    pthread_mutex_unlock(&mutex);
    return((void *)0);
}

int main(int argc,char** argv)
{  
    int a;
    void * p = &a;
    printf("%p\n", p);
    printf("%p\n", (void *) ((float *)p + 1) );
    int num1 = 1;
    int num2 = 2;
    std::vector<pthread_t> vec;
    pthread_mutex_init(&mutex, 0);
    int n = 4;
    vec.resize(n);
    for (int i = 0; i < n; i++){
        if (i % 2 == 0)
            pthread_create(&vec[i], NULL, print_msg, &num1);
        else
            pthread_create(&vec[i], NULL, print_msg, &num2);
        printf("Thread %d: ID %d\n",i , vec[i]);
    }

    for (int i = 0; i < n; i++){
        int res = pthread_join(vec[i], NULL);
        if (res == 0)
            printf("Sucessfully reclaim\n");
        printf("Thread %d: ID %d\n",i , vec[i]);
    }

    printf("\n\nMain thread \n\n");
    // pthread_join(id2, NULL);
    pthread_mutex_destroy(&mutex);
    return 0;
}  