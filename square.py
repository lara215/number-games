import math
import time
from math import gcd, isqrt
from typing import List, Tuple
import numpy as np


# ---------- Helpers ----------
def square(number: float) -> float:
    """Return the square of a given number."""
    return number * number


def solve_equation(leg: float, hypotenuse: float) -> float:
    """Compute sqrt(hypotenuse^2 - leg^2). Assumes hypotenuse^2 >= leg^2."""
    return np.sqrt(square(hypotenuse) - square(leg))


def _to_tuples(triples) -> List[Tuple[int, int, int]]:
    """Convert an iterable of triples into a list of plain integer tuples."""
    return [(int(a), int(b), int(c)) for a, b, c in triples]


# ---------- Solvers ----------
def solve_slow(limit: int) -> List[Tuple[int, int, int]]:
    """
    Brute-force solver:
    Loop over all possible hypotenuses (c) and legs (a, b),
    check if a^2 + b^2 = c^2.
    """
    solutions = []
    for hypotenuse in range(1, limit + 1):
        c2 = hypotenuse * hypotenuse
        for leg_b in range(1, hypotenuse):
            remainder = c2 - leg_b * leg_b
            if remainder <= 0:
                continue
            leg_a = isqrt(remainder)
            if leg_a * leg_a == remainder:
                solutions.append((leg_a, leg_b, hypotenuse))
    return _to_tuples(solutions)


def solve_chatGPT(limit: int) -> List[Tuple[int, int, int]]:
    """
    Generate primitive Pythagorean triples using Euclid's formula:
        a = m^2 - n^2, b = 2mn, c = m^2 + n^2
    Then scale them by k until the limit is reached.
    """
    triples = []
    m = 2
    while True:
        if m * m + 1 > limit:
            break
        for n in range(1, m):
            # Must be coprime and one even/one odd
            if ((m - n) & 1) == 0 or gcd(m, n) != 1:
                continue
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n
            if c > limit:
                break
            k = 1
            while k * c <= limit:
                triples.append((k * a, k * b, k * c))
                k += 1
        m += 1
    return _to_tuples(triples)


def solve_chatGPT_jonas(limit: int) -> List[Tuple[int, int, int]]:
    """
    Alternate brute-force:
    Loop over hypotenuse (c) and one leg (a),
    derive the other leg (b) using integer square root.
    """
    solutions = []
    for hypotenuse in range(1, limit + 1):
        c2 = hypotenuse * hypotenuse
        for leg_a in range(1, hypotenuse):
            b2 = c2 - leg_a * leg_a
            if b2 <= 0:
                continue
            leg_b = isqrt(b2)
            if leg_b * leg_b == b2:
                solutions.append((leg_a, leg_b, hypotenuse))
    return _to_tuples(solutions)


def solve_meshgrid(limit: int) -> List[Tuple[int, int, int]]:
    """
    Vectorized solver using NumPy:
    Create a grid of possible legs (b) and hypotenuses (c),
    compute a^2 = c^2 - b^2, check which are perfect squares.
    """
    leg_b_values = np.arange(1, limit)
    hypotenuse_values = np.arange(1, limit + 1)

    # Create 2D arrays of b and c values
    B, C = np.meshgrid(leg_b_values, hypotenuse_values, indexing="ij")

    # Compute a^2 = c^2 - b^2
    A_squared = C**2 - B**2

    # Mask invalid (negative or zero) values
    mask = A_squared > 0

    # Compute a only where valid
    A = np.zeros_like(A_squared, dtype=float)
    np.sqrt(A_squared, where=mask, out=A)

    # Keep only integer solutions
    mask &= (A == np.floor(A))

    # Extract solutions
    leg_a_values = A[mask].astype(int)
    leg_b_out = B[mask].astype(int)
    hypotenuse_out = C[mask].astype(int)

    triples = zip(leg_a_values, leg_b_out, hypotenuse_out)
    return _to_tuples(triples)

def add_squares(list1: np.ndarray, list2: np.ndarray) -> dict[tuple, int]:
    dictionary ={}
    for x1 in list1:
        for x2 in list2:
            dictionary[(x1,x2)]=x1+x2

    return dictionary

def solve_savling_list(limit: int) -> List[Tuple[int,int,int]]:
    numbers=np.arange(1,limit+1)
    square_numbers=numbers**2
    summed_squares=add_squares(square_numbers,square_numbers)
    #print(summed_squares)
    results=[]
    for key,value in summed_squares.items():
        if value in square_numbers:
          results.append((key[0]**0.5, key[1]**0.5, value**0.5))
            
                    
                 
    return(results)


# ---------- Benchmark ----------
def benchmark(upper_limit: int, show_solutions: bool = False) -> None:
    """Run each solver, measure execution time, and optionally print solutions."""
    solvers = [
        ("Brute-force (slow)", solve_slow),
        ("Euclid's formula", solve_chatGPT),
        ("Vectorized meshgrid", solve_meshgrid),
        ("Brute-force (Jonas)", solve_chatGPT_jonas),
        ("saving squares", solve_savling_list)
    ]

    for solver_name, solver_func in solvers:
        start_time = time.time()
        solutions = solver_func(upper_limit)
        elapsed = time.time() - start_time

        print(f"{solver_name:>20}  time: {elapsed:.6f} s  (found {len(solutions)} triples)")
        if show_solutions:
            print(f"{solver_name:>20}  solutions: {solutions}")


# ---------- Main ----------
def main():
    # Set parameters here
    upper_limit = 200       # maximum hypotenuse to consider
    show_solutions = False   # set True to print all triples

    benchmark(upper_limit, show_solutions)


if __name__ == "__main__":
    main()


  
