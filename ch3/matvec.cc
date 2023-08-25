#include <mpi.h>
#include <memory>
#include <cmath>
#include <algorithm>
#include <spdlog/spdlog.h>

const auto rows = 10000;
const auto cols = 10000;

int main(int argc, char **argv)
{
    spdlog::set_level(spdlog::level::off);
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        spdlog::error("MPI_Init error\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    if (np == 1)
    {
        spdlog::error("This program requires at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    std::unique_ptr<double[]> b(new double[rows]);
    std::unique_ptr<double[]> A(new double[rows * cols]);

    const int num_worker = np - 1;
    const int manager_rank = (np - 1);

    MPI_Barrier(MPI_COMM_WORLD);

    double duration = -MPI_Wtime();
    // use the last process as the master process
    if (rank == manager_rank)
    {
        std::unique_ptr<double[]> c(new double[rows]);
        for (auto j = 0; j < cols; ++j)
        {
            b[j] = 1;
            for (auto i = 0; i < rows; ++i)
            {
                A[i * cols + j] = j + 1;
            }
        }
        // Send the vector b to all processes
        spdlog::info("process %d: sending b to all processes\n", rank);
        MPI_Bcast(b.get(), cols, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        int num_sent_rows = 0;
        // Send A row-by-row to each worker process, i.e. row 0 to process 0
        for (auto i = 0; i < std::min(rows, num_worker); ++i)
        {
            // tag is row
            spdlog::info("process %d: sending row %d to rank %d\n", rank, i, i);
            MPI_Send(A.get() + i * cols, cols, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            spdlog::info("process %d: sent row %d to rank %d\n", rank, i, i);
            ++num_sent_rows;
        }

        // Receive the result from each worker process
        MPI_Status status;
        double ans = 0.0;
        // ensure receive the result from each row
        for (int i = 0; i < rows; ++i)
        {
            MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            spdlog::info("process %d: received ans %lf from rank %d\n", rank, ans, status.MPI_SOURCE);
            const auto row = status.MPI_TAG;
            const auto sender = status.MPI_SOURCE;
            c[row] = ans;
            // Now sender is free, send it a new row or STOP it
            if (num_sent_rows < rows)
            {
                MPI_Send(A.get() + num_sent_rows * cols, cols, MPI_DOUBLE, sender, num_sent_rows, MPI_COMM_WORLD);
                num_sent_rows++;
            }
            else
            {
                MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, sender, rows, MPI_COMM_WORLD);
            }
        }

        // Print the result
        spdlog::info("process %d: c =\n", rank);
        for (auto i = 0; i < rows; ++i)
        {
            spdlog::info("%lf ", c[i]);
        }
        spdlog::info("\n");
        printf("process %d: duration = %lf\n", rank, duration + MPI_Wtime());
    }
    else
    {
        std::unique_ptr<double[]> buffer(new double[cols]);
        // Receive the vector b from master
        MPI_Bcast(b.get(), cols, MPI_DOUBLE, manager_rank, MPI_COMM_WORLD);
        // process 0, ..., rows - 1 receive A[0], ..., A[rows - 1],
        // any process with rank >= rows will not receive anything
        spdlog::info("process %d: received b from %d\n", rank, manager_rank);
        if (rank >= rows)
        {
            goto _exit;
        }
        for (;;)
        {
            // Receive the matrix A row-by-row from process 0
            spdlog::info("process %d: waiting for a row\n", rank);
            MPI_Status status;
            MPI_Recv(buffer.get(), cols, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            // No more job, stop!
            if (status.MPI_TAG == rows)
            {
                spdlog::info("process %d: received STOP signal\n", rank);
                break;
            }
            const auto row = status.MPI_TAG;
            spdlog::info("process %d: received row %d\n", rank, row);
            double ans = 0.0;
            for (auto j = 0; j < cols; ++j)
            {
                ans += buffer[j] * b[j];
            }
            spdlog::info("process %d: calculated ans %lf\n", rank, ans);
            MPI_Send(&ans, 1, MPI_DOUBLE, manager_rank, row, MPI_COMM_WORLD);
        }
    }

_exit:
    MPI_Finalize();
    return 0;
}