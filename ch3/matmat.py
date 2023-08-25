#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from loguru import logger
from mpi4py import MPI

A_ROWS = 20
A_COLS = 10000
STOP_TAG = A_COLS

B_ROWS = A_COLS
B_COLS = 20


def manager():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    # initialize matrix B
    B = np.empty((B_ROWS, B_COLS), dtype=np.float64)
    B[:, :] = range(1, B_COLS + 1)

    comm = MPI.COMM_WORLD

    # send matrix B to all workers

    comm.Bcast(B, root=manager_rank)
    logger.info("manager: sent matrix B to all workers.")

    # initialize matrix A
    A = np.empty((A_ROWS, A_COLS), dtype=np.float64)
    A[:, :] = range(1, A_COLS + 1)

    duration = -MPI.Wtime()
    # send one row of matrix A to each worker
    num_rows_sent = 0
    for i in range(min(A_ROWS, size - 1)):
        comm.Send(A[i, :], dest=i, tag=i)
        num_rows_sent += 1

    C = np.empty((A_ROWS, B_COLS), dtype=np.float64)
    buffer = np.empty(B_COLS, dtype=np.float64)
    # ensure all rows of C are received
    for i in range(A_ROWS):
        status = MPI.Status()
        comm.Recv(buffer, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        row = status.tag
        sender = status.source
        logger.info(f"manager: received answer row {row} from worker {sender}.")
        C[row, :] = buffer[:]

        # now sender is free to receive another row, send it one if available
        # or stop it
        if num_rows_sent < A_ROWS:
            comm.Send(A[num_rows_sent, :], dest=sender, tag=num_rows_sent)
            num_rows_sent += 1
        else:
            comm.send(None, dest=sender, tag=STOP_TAG)

    print(f"manager: finished in {duration + MPI.Wtime()} seconds.")
    print("C = \n", C)


def worker(manager_rank: int, rank: int):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    B = np.empty((B_ROWS, B_COLS), dtype=np.float64)
    comm.Bcast(B, root=manager_rank)
    logger.info(f"worker {rank}: received matrix B from manager.")

    buffer = np.empty(A_COLS, dtype=np.float64)
    while True:
        status = MPI.Status()
        comm.Recv(buffer, source=manager_rank, tag=MPI.ANY_TAG, status=status)
        if status.tag == STOP_TAG:
            logger.info(f"worker {rank}: received stop tag from manager.")
            break

        row = status.tag
        logger.info(f"worker {rank}: received row {row} from manager. 😀")
        answer = np.dot(buffer, B)
        logger.info(f"worker {rank}: sending answer back to manager. 😎")
        comm.Send(answer, dest=manager_rank, tag=row)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size <= 1:
        print("Please run with at least 2 processes.")
        exit(1)

    manager_rank = size - 1
    comm.Barrier()

    if rank == manager_rank:
        manager()
    else:
        worker(manager_rank, rank)

    MPI.Finalize()
