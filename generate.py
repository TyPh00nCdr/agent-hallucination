from multiprocessing import Pool
from os import sched_getaffinity

from race import rollout

AVAILABLE_CORES = len(sched_getaffinity(0))

if __name__ == "__main__":
    print(f'Available threads: {AVAILABLE_CORES}')

    with Pool(AVAILABLE_CORES) as pool:
        pool.map(rollout, range(AVAILABLE_CORES))
