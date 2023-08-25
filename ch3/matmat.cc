#include <mpi.h>
#include <memory>
#include <spdlog/spdlog.h>

const auto A_ROWS = 20;
const auto A_COLS = 10000;

const auto STOP_TAG = A_ROWS;

const auto B_ROWS = 10000;
const auto B_COLS = 20;

void manager(int manager_rank)
{
    auto B = std::unique_ptr<double[]>(new double[B_ROWS * B_COLS]);
    // initialize B, each entry in a column is the same
    for (auto j = 0; j < B_COLS; ++j)
    {
        for (auto i = 0; i < B_ROWS; ++i)
        {
            B[i * B_COLS + j] = j + 1;
        }
    }

    // send B to all processes
    spdlog::debug("manager: sending B to all processes");
    MPI_Bcast(B.get(), B_ROWS * B_COLS, MPI_DOUBLE, manager_rank, MPI_COMM_WORLD);

    auto A = std::unique_ptr<double[]>(new double[A_ROWS * A_COLS]);
    // initialize A, each entry in a column is the same
    for (auto j = 0; j < A_COLS; ++j)
    {
        for (auto i = 0; i < A_ROWS; ++i)
        {
            A[i * A_COLS + j] = j + 1;
        }
    }

    auto duration = -MPI_Wtime();
    // send a row of A to each process
    spdlog::debug("manager: sending A row-by-row to all processes");
    int num_sent_rows = 0;
    for (auto i = 0; i < std::min(A_ROWS, manager_rank); ++i)
    {
        // tag is row number
        MPI_Send(A.get() + i * A_COLS, A_COLS, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
        spdlog::debug("process %d: sent row %d to rank %d\n", manager_rank, i, i);
        ++num_sent_rows;
    }

    auto C = std::unique_ptr<double[]>(new double[A_ROWS * B_COLS]);
    auto buffer = std::unique_ptr<double[]>(new double[B_COLS]);
    MPI_Status status;
    // ensure that all rows of C are received
    for (auto i = 0; i < A_ROWS; ++i)
    {
        MPI_Recv(buffer.get(), B_COLS, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
        const auto row = status.MPI_TAG;
        const auto sender = status.MPI_SOURCE;
        spdlog::debug("manager: received ans row {:d} from rank {:d}\n",
                      row, sender);
        std::memcpy(C.get() + row * B_COLS, buffer.get(), B_COLS * sizeof(double));
        // now sender is free, send it a new row or STOP it
        if (num_sent_rows < A_ROWS)
        {
            MPI_Send(A.get() + num_sent_rows * A_COLS, A_COLS, MPI_DOUBLE,
                     sender, num_sent_rows, MPI_COMM_WORLD);
            spdlog::debug("manager: sent row {:d} to rank {:d}\n",
                          num_sent_rows, sender);
            ++num_sent_rows;
        }
        else
        {
            MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, sender, STOP_TAG, MPI_COMM_WORLD);
        }
    }

    printf("manager: duration = %lf\n", duration + MPI_Wtime());
    // output C
    printf("C =\n");
    for (auto i = 0; i < A_ROWS; ++i)
    {
        for (auto j = 0; j < B_COLS; ++j)
        {
            printf("%lf ", C[i * B_COLS + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void worker(int rank, int manager_rank)
{
    auto B = std::unique_ptr<double[]>(new double[B_ROWS * B_COLS]);
    // receive B from the manager
    MPI_Bcast(B.get(), B_ROWS * B_COLS, MPI_DOUBLE, manager_rank, MPI_COMM_WORLD);

    spdlog::debug("worker %d: received B from manager\n", rank);

    auto buffer = std::unique_ptr<double[]>(new double[A_COLS]);

    // receive A row-by-row from the manager
    MPI_Status status;

    auto answer = std::unique_ptr<double[]>(new double[B_COLS]);

    for (;;)
    {
        MPI_Recv(buffer.get(), A_COLS, MPI_DOUBLE, manager_rank, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
        const auto row = status.MPI_TAG;
        if (row == STOP_TAG)
        {
            spdlog::debug("worker {:d}: received STOP signal\n", rank);
            break;
        }
        spdlog::debug("worker {:d}: received row {:d} from manager\n", rank, row);
        std::memset(answer.get(), 0, B_COLS * sizeof(double));
        // perform multiplication by row perspective
        for (auto i = 0; i < A_COLS; ++i)
        {
            for (auto j = 0; j < B_COLS; ++j)
            {
                answer[j] += buffer[i] * B[i * B_COLS + j];
            }
        }
        MPI_Send(answer.get(), B_COLS, MPI_DOUBLE, manager_rank, row, MPI_COMM_WORLD);
        spdlog::debug("worker {:d}: sent ans row {:d} to manager\n", rank, row);
    }
}

int main(int argc, char **argv)
{
    spdlog::set_level(spdlog::level::critical);
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        printf("MPI_Init error\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size <= 1)
    {
        spdlog::critical("Error: size must be greater than 1\n");
        return 1;
    }

    const int manager_rank = size - 1;

    if (rank == manager_rank)
    {
        manager(manager_rank);
    }
    else
    {
        worker(rank, manager_rank);
    }

    MPI_Finalize();
    return 0;
}