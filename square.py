import numpy as np
from math import gcd
from typing import List, Tuple
import time
import math


def square(number: float) -> float:
    """Return the square of a given number."""
    return number * number

def solve_equation(y,z):
    return np.sqrt(square(z)-square(y))

def _to_tuples(triples):
    """Ensure output is list of tuples with plain Python ints."""
    return [(int(x), int(y), int(z)) for x, y, z in triples]


# ---- 1) Slow scan (integer math; no floats) ----
def solve_slow(limit: int):
    solutions = []
    for z in range(1, limit + 1):
        z2 = z * z
        for y in range(1, z):
            s = z2 - y * y
            if s <= 0:
                continue
            x = math.isqrt(s)
            if x * x == s:
                solutions.append((x, y, z))
    return _to_tuples(solutions)


def solve_slow(limit: int):
    solutions = []
    for z in range(1, limit + 1):
        z2 = z * z
        for y in range(1, z):
            s = z2 - y * y
            if s <= 0:
                continue
            x = math.isqrt(s)
            if x * x == s:
                solutions.append((x, y, z))
    return _to_tuples(solutions)


def solve_chatGPT(limit: int):
    out = []
    m = 2
    while True:
        if m*m + 1 > limit:
            break
        for n in range(1, m):
            if ((m - n) & 1) == 0 or gcd(m, n) != 1:
                continue
            a = m*m - n*n
            b = 2*m*n
            c = m*m + n*n
            if c > limit:
                break
            k = 1
            while k * c <= limit:
                out.append((k*a, k*b, k*c))
                k += 1
        m += 1
    return _to_tuples(out)


def solve_chatGPT_jonas(limit: int):
    solutions = []
    for z in range(1, limit + 1):
        z2 = z * z
        for x in range(1, z):
            y2 = z2 - x * x
            if y2 <= 0:
                continue
            y = math.isqrt(y2)
            if y * y == y2:
                solutions.append((x, y, z))
    return _to_tuples(solutions)


def solve_meshgrid(limit: int):
    y_vals = np.arange(1, limit)
    z_vals = np.arange(1, limit + 1)
    Y, Z = np.meshgrid(y_vals, z_vals, indexing="ij")
    X2 = Z**2 - Y**2
    mask = X2 > 0
    X = np.zeros_like(X2, dtype=float)
    np.sqrt(X2, where=mask, out=X)
    mask &= (X == np.floor(X))
    x_vals = X[mask].astype(int)
    y_out = Y[mask].astype(int)
    z_out = Z[mask].astype(int)
    triples = zip(x_vals, y_out, z_out)
    return _to_tuples(triples)



def benchmark(upper_limit: int):
    solvers = [
        ("slow", solve_slow),
        ("chatGPT", solve_chatGPT),
        ("meshgrid", solve_meshgrid),
        ("chatGPT_jonas", solve_chatGPT_jonas),
    ]

    for name, func in solvers:
        start_time = time.time()
        solution = func(upper_limit)
        elapsed = time.time() - start_time
        print(f"{name} time: {elapsed:.4f} seconds")
        # Uncomment if you also want to inspect solutions
        print(f"{name} solution: {solution}")

if __name__ == "__main__":
    upper_limit = 5
    benchmark(upper_limit)


  
