#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate PI using monte-carlo algorithm.

Usage:
    monte_carlo.py [-h | --help]
    monte_carlo.py -e <epsilon>

Options:
    -h --help   show this screen.
    -e <epsilon>    specify epsilon [default: 1e-5]
"""

import sys

import numpy as np
from docopt import docopt
from loguru import logger
from mpi4py import MPI
from numpy import sqrt

REQUEST_CHUNK = 1
CHUNK_SIZE = 5000
STOP_TAG = 999
PI25 = 3.141592653589793238462643
MAX_ITER = 1e5

MINIMUM_LOG_LEVEL = "DEBUG"


def create_comm_worker():
    comm_world = MPI.COMM_WORLD
    worker_group = comm_world.group.Excl([0])
    comm_worker = MPI.COMM_WORLD.Create_group(worker_group)
    return comm_worker


def root(epsilon: float):
    """estimate PI using monte-carlo method

    Args:
        epsilon (float, optional): tolerance. Defaults to 1e-5.
    """
    logger.remove(0)
    logger.add(
        sys.stderr,
        format="{time} | {level} | ROOT | {message}",
        level=MINIMUM_LOG_LEVEL,
    )
    comm_world = MPI.COMM_WORLD
    comm_world.bcast(epsilon)
    logger.trace(f"send epsilon {epsilon}")
    status = MPI.Status()
    duration = -MPI.Wtime()
    while True:
        done = comm_world.recv(None, tag=REQUEST_CHUNK, status=status)
        logger.trace(f"recieve chunk request from {status.source}")

        # prepare one chunk and dispatch it
        chunk = np.random.rand(CHUNK_SIZE)
        logger.trace(f"send a chunk to {status.source}")
        comm_world.Send(chunk, dest=status.source)

        if done:
            logger.trace("job done")
            break
    logger.success(f"Finished in {duration + MPI.Wtime()} secs.")


def worker():
    comm_world = MPI.COMM_WORLD
    rank = comm_world.rank
    logger.remove(0)
    logger.add(
        sys.stderr,
        format=f"{{time}} | {{level}} | WORKER {rank} | {{message}}",
        level=MINIMUM_LOG_LEVEL,
    )
    epsilon = comm_world.bcast(None)
    logger.trace(f"receive epsilon {epsilon}")
    comm_world.send(False, dest=0, tag=REQUEST_CHUNK)
    comm_worker = create_comm_worker()
    chunk = np.empty(CHUNK_SIZE)
    total_in_circle_cnt = 0
    total_out_circle_cnt = 0
    in_circle_cnt = 0
    out_circle_cnt = 0
    iter_cnt = 0
    while True:
        iter_cnt += 1
        status = MPI.Status()
        comm_world.Recv(chunk, source=0, status=status)
        if status.tag == STOP_TAG:
            break
        logger.trace("receive a chunk")
        for i in range(0, CHUNK_SIZE, 2):
            # put in [0, 2] and shift left for 1
            x = chunk[i] * 2 - 1
            y = chunk[i + 1] * 2 - 1

            in_ = sqrt(x**2 + y**2) <= 1.0
            if in_:
                in_circle_cnt += 1
            else:
                out_circle_cnt += 1
        total_in_circle_cnt = comm_worker.allreduce(in_circle_cnt, MPI.SUM)
        total_out_circle_cnt = comm_worker.allreduce(out_circle_cnt, MPI.SUM)

        pi = 4 * total_in_circle_cnt / (total_in_circle_cnt + total_out_circle_cnt)
        r = abs(pi - PI25)
        done = r < epsilon or iter_cnt >= MAX_ITER

        if done:
            # let rank 1 tell root
            if rank == 1:
                logger.success(f"rank 1: PI is {pi}")
                comm_world.send(True, 0, REQUEST_CHUNK)
            break
        comm_world.send(False, dest=0, tag=REQUEST_CHUNK)


def main(epsilon: float = 1e-5):
    comm_world = MPI.COMM_WORLD
    rank = comm_world.rank
    size = comm_world.size
    if rank == 0:
        if size <= 1:
            print("At least 2 processes should be used", flush=True)
            comm_world.Abort(1)
        args = docopt(doc=__doc__, help=False)
        if args.get("--help"):
            print(__doc__, flush=True)
            comm_world.Abort(1)
        root(float(args.get("-e", "1e-5")))
    else:
        worker()

    MPI.Finalize()


if __name__ == "__main__":
    main()
# end main
