Run python3 python/solve_all.py inputs outputs

# Spring 2022 CS170 Project | TeamID 048 

## Overview 

This project submission was built off the provided [CS170 Project Skeleton](https://github.com/Berkeley-CS170/project-sp22-skeleton). The algorithm implemented utilizes a greedy approach with tie-breaking optimizations to determine tower placements that aim to minimize total penalty accumulated.

_Contributors: Andrew Tran, Elizabeth Yeh, Jason Zhao_

## Modifications

The changes we made to the original skeleton include the following:

python/solve.py | solve_greedy: this function and its associated helpers implement the above described algorithm for deciding tower placements.

python/solve_all.py | solver: we modified the function to use our code instead of the naive implementation.

## Usage 

We added our greedy implementation at [`python/solve.py`].
```bash
python3 solve.py case.in –solver=greedy case.out
```

The code at [`python/solve_all.py`] has already been modified to use solve_greedy. To run our implementation on all inputs, use
```bash
python3 python/solve_all.py inputs outputs
```
in the root directory.

