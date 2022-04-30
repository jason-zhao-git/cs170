"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from asyncio.windows_events import NULL
from ctypes.wintypes import BOOLEAN
from pathlib import Path
from typing import Callable, Dict
from xmlrpc.client import boolean


from instance import Instance
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper
from point import Point

def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

def solve(instance: Instance) -> Solution:
    side_len = instance.D
    score_matrix = [[0 for i in range(side_len)] for j in range(side_len)]
    r = instance.R_s
    print(r)
    cities = instance.cities.copy()
    for city in cities:
        x = city.x
        y = city.y
        for i in range(x-r, x+r):
            for j in range(y-r, y+r):
                if i >= 0 and j >= 0 and i < side_len and j < side_len:
                    currPoint = Point(x=int(i), y=int(j))
                    if currPoint.distance_sq(city) <= r*r:
                        score_matrix[i][j] = score_matrix[i][j] + 1
    
    for i in range(len(score_matrix)):
        print(score_matrix[i])
                    
    #find max of the matrix
    towerss = []
    while cities != [NULL]*len(cities):
        t = maxmat(score_matrix, side_len)
        towerss.append(t)
        update(t, score_matrix, side_len, cities, r)
    

    return Solution(
        instance=instance,
        towers=towerss,
    )


def maxmat(score_matrix, leng):
    maxmat_ind = Point(x=0, y=0)
    maxmat_val = 0
    for i in range(leng):
        for j in range(leng):
            if score_matrix[i][j] > maxmat_val:
                maxmat_val = score_matrix[i][j]
                maxmat_ind = Point(x=i, y=j)
    return maxmat_ind

def update(tower, score_matrix, side, cities, r):
    x = tower.x
    y = tower.y
    for i in range(x-r, x+r):
        for j in range(y-r, y+r):
            if i >= 0 and j >= 0 and i < side and j < side:
                currPoint = Point(x=int(i), y=int(j))
                if currPoint.distance_sq(tower) <= r*r:
                    for index in range(len(cities)):
                        city = cities[index]
                        if city != 0 and city.x == currPoint.x and city.y == currPoint.y:
                            update_score(city, score_matrix, side, r)
                            cities[index] = NULL

def update_score(city, score_matrix, side, r):
    x = city.x
    y = city.y
    for i in range(x-r, x+r):
        for j in range(y-r, y+r):
            if i >= 0 and j >= 0 and i < side and j < side:
                currPoint = Point(x=int(i), y=int(j))
                if currPoint.distance_sq(city) <= r*r:
                    score_matrix[i][j] = score_matrix[i][j] - 1







SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "greedy": solve
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")


def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")


def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str,
                        help="The output file. Use - for stdout.",
                        default="-")
    main(parser.parse_args())
