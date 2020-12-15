#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

extern void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr) {

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // int *tmp_ptr = new int[n*m*sizeof(int)]
    
    // root to read data
    if(rank == 0) {
        std::cin >> *n_ptr >> *m_ptr >> *l_ptr;
        *a_mat_ptr = new int [*n_ptr * *m_ptr];
        for(int i = 0; i < *n_ptr; ++i) {
            for(int j = 0; j < *m_ptr; ++j) {
                std::cin >> *((*a_mat_ptr + i * *m_ptr) + j);
            }
        }
        *b_mat_ptr = new int [*m_ptr * *l_ptr];
        for(int i = 0; i < *m_ptr; ++i) {
            for(int j = 0; j < *l_ptr; ++j) {
                std::cin >> *((*b_mat_ptr + i * *l_ptr) + j);
            }
        }
    }

    // broadcast arg to all processes
    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank != 0) {
        *a_mat_ptr = new int [*n_ptr * *m_ptr];
        *b_mat_ptr = new int [*m_ptr * *l_ptr];
    }

    // broadcast data to all processes
    MPI_Bcast(*a_mat_ptr, *n_ptr * *m_ptr, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, *m_ptr * *l_ptr, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

extern void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat) {
    
    // get MPI info
    int rank, size, *result, *tmp;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // check cin
    // if(rank == 0) {
    //     std::cout << n << ' ' << m << ' ' << l << '\n';
    //     for(int i = 0; i < n; ++i) {
    //         for(int j = 0; j < m; ++j) {
    //             std::cout << *(a_mat + i * m + j) << ' ';
    //         }
    //         std::cout << '\n';
    //     }
    //     for(int i = 0; i < m; ++i) {
    //         for(int j = 0; j < l; ++j) {
    //             std::cout << *(b_mat + i * l + j) << ' ';
    //         }
    //         std::cout << '\n';
    //     }
    // }
    if(rank == 0) {
        result = new int [n * l];
    }

    // perform multiplication by all processes
    int per_r = n / size;
    int tmp[per_r][l];
    for(int k = 0; k < l; ++k) {
        for(int i = 0; i < per_r; ++i) {
            tmp[i][k] = 0;
            for(int j = 0; j < m; ++j) {
                tmp[i][k] = tmp[i][k] + *(a_mat + (i + rank * per_r) * m + j) * *(b_mat + j * l + k);
            }
        }
    }
    MPI_Gather(tmp, per_r * l, MPI_INT, result, per_r * l, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        int extra = n % size;
        for(int k = 0; k < l; ++k) {
            for(int i = 0; i < extra; ++i) {
                int row = i + size * per_r;
                *(result + row * l + k) = 0;
                for(int j = 0; j < m; ++j) {
                    *(result + row * l + k) = *(result + row * l + k) + *(a_mat + (row) * m + j) * *(b_mat + j * l + k);
                }
            }
        }
        for(int i = 0; i < n; ++i) {
            std::cout << *(result + i * l);
            for(int j = 1; j < l; ++j) {
                std::cout << ' ' << *(result + i * l + j);
            }
            std::cout << '\n';
        }
        delete [] result;
    }
}

extern void destruct_matrices(int *a_mat, int *b_mat) {

    // free memory
    delete [] a_mat;
    delete [] b_mat;
}