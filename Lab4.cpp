#include <stdio.h>                                                                                                                          
#include <math.h>                                                                                                                           
#include <time.h>                                                                                                                           
#include <stdlib.h>                                                                                                                         
#include <mpi.h>                                                                                                                         


// user defined function below                                                                                                              
float f(float x) { return exp(cos(x)); }

//function to calculate a definite integral given bounds of integration (xmin/max) & bounds of function (ymin/ymax)                         
float calc_integral(float(*f)(float), float xmin, float xmax, float ymin, float ymax) {

	float total, inBox;

	for (int count = 0; count < 1000000; count++) {
		float u1 = (float)rand() / (float)RAND_MAX;
		float u2 = (float)rand() / (float)RAND_MAX;
		float xcoord = ((xmax - xmin)*u1) + xmin;
		float ycoord = ((ymax - ymin)*u2) + ymin;
		float val = f(xcoord);
		total++;
		if (val > ycoord) {
			inBox++;
		}
	}

	float density = inBox / total;

	return (xmax - xmin) * (ymax - ymin) * density;
}

int main(int argc, char* argv[]) {

	int size, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double *limits = NULL, *local_limits;
	double integral = 0, local_integral;

	double start_time;

	if (rank == 0)
	{
		double lower_limit = atoi(argv[1]);
		double upper_limit = atoi(argv[2]);
		double limit_step = (upper_limit - lower_limit) / size;

		limits = (double*)malloc(size * 2 * sizeof(double));

		for (int i = 0, double current_limit = lower_limit; i < size; i++) {
			limits[i * 2] = current_limit;
			current_limit += limit_step;
			limits[i * 2] = current_limit;
		}

		start_time = MPI_Wtime();
	}

	local_limits = (double *)malloc(2 * sizeof(double));

	MPI_Scatter(limits, 2, MPI_DOUBLE, local_limits, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	local_integral = calc_integral(f, local_limits[0], local_limits[1], 0, 4);

	free(local_limits);

	printf("Local integral of process #%d is: %f\n", rank, local_integral);
	fflush(stdout);

	MPI_Reduce(&local_integral, &integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		double end_time = MPI_Wtime();
		double time = ((double)(end_time - start_time));

		printf("Processing time: %f, s\n", time);
		printf("Integral: %f\n", integral);
	}

	MPI_Finalize();

	return 0;
}