#include <mpi.h>
#include <stdio.h>

const double PI25DT = 3.141592653589793238462643;

inline double f(double x)
{
    return 4.0 / (1.0 + x * x);
}

int main(int argc, char **argv)
{
    int rank;
    int size;
    int intervals_num;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        printf("MPI_Init error\n");
        return 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    while (true)
    {
        if (rank == 0)
        {
            printf("Enter the number of intervals: (0 quits)");
            scanf("%d", &intervals_num);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        double duration = -MPI_Wtime();
        MPI_Bcast(&intervals_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (intervals_num <= 0)
        {
            break;
        }

        double sum = 0.0;
        for (int i = rank; i < intervals_num; i += size)
        {
            double x = (i + 0.5) / intervals_num;
            sum += f(x);
        }
        double local_pi = sum / intervals_num;

        double PI;
        MPI_Reduce(&local_pi, &PI, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            printf("pi is approximately %.16f, Error is %.16f\n", PI, PI - PI25DT);
            printf("wall clock time = %f seconds\n", duration + MPI_Wtime());
        }
    }

    MPI_Finalize();
    return 0;
}
