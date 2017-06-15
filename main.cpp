

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <cmath>

using namespace std;
using namespace std::chrono;

void block_matrix_mul_parallel(float **A, float **B, float **C, int size, int block_size, int threads_num)
{
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
	float tmp;
	int chunk = 1;
	omp_set_dynamic(0);
	omp_set_num_threads(threads_num);
#pragma omp parallel shared(A, B, C, size, chunk) private(i, j, k, jj, kk, tmp)
	{

#pragma omp for schedule (static, chunk)
		for (jj = 0; jj < size; jj += block_size)
		{
			for (kk = 0; kk < size; kk += block_size)
			{
				for (i = 0; i < size; i++)
				{
					for (j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
					{
						tmp = 0.0f;
						for (k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
						{
							tmp += A[i][k] * B[k][j];
						}
						C[i][j] += tmp;
					}
				}
			}
		}
	}
}

void output_matrix(float **matrix, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			matrix[i][j] = 0.0f;
}
int main(int argc, char **argv)
{
	FILE **files = new FILE*[8];
	for (int i = 0; i < 8; i++)
	{
		char file_name[30];
		sprintf(file_name, "output/threads_num_%d", i + 1);
		files[i] = fopen(file_name, "a+");
	}

	for (int size = 100; size < 1100; size += 100)
	{
		int block_size = 10;
		float **A = new float*[size];
		float **B = new float*[size];
		float **C = new float*[size];
		cout << "Init has begun" << endl;
		for (int i = 0; i < size; i++)
		{
			A[i] = new float[size];
			B[i] = new float[size];
			C[i] = new float[size];
			for (int j = 0; j < size; j++)
			{
				float random_numberA = 0.0f + rand() % (RAND_MAX / 2);
				float random_numberB = 0.0f + rand() % (RAND_MAX / 2);
				if (i == j) // diagonal element                            
				{
					random_numberA = (float)rand();
					random_numberB = (float)rand();
				}
				A[i][j] = random_numberA;
				B[i][j] = random_numberB;
				C[i][j] = 0.0f;
			}
		}
		cout << "Computation has begun" << endl;
		double start, end;
		for (int i = 1; i < 9; i++)
		{
			cout << "Number of threads: " << i << endl;
			double sum_time = 0.0;
			int iter = 200;
			for (int j = 0; j < iter; j++)
			{
				start = omp_get_wtime();
				block_matrix_mul_parallel(A, B, C, size, block_size, i);
				end = omp_get_wtime();
				sum_time += (end - start);
			}
			char buff[13];
			sprintf(buff, "%d;%f\n", size, sum_time / iter);
			fwrite(buff, sizeof(char), sizeof(buff), files[i - 1]);
			cout << "Time elapsed: " << sum_time / iter << endl;
		}
	}
	for (int i = 0; i < 8; i++)
		fclose(files[i]);
	delete[] files;
	return 0;
}

