import numpy as np
from math import gcd
from typing import List, Tuple
from time import time


def square(number: float) -> float:
    """Return the square of a given number."""
    return number * number

def solve_equation(y,z):
    return np.sqrt(square(z)-square(y))

def solve_slow(upper_limit):
    solutions=[]
    for z in range(1,upper_limit+1):
        for y in range(1,z-1):
            x= solve_equation(y,z)
            if x.is_integer():
                solutions.append([int(x),y,z])
    return (solutions)

def solve_chatGPT(limit: int) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    m = 2
    while True:
        z0 = m*m + 1  # smallest n=1 gives this z
        if z0 > limit:
            break
        for n in range(1, m):
            if ((m - n) & 1) == 0 or gcd(m, n) != 1:
                continue
            a = m*m - n*n
            b = 2*m*n
            c = m*m + n*n
            if c > limit:
                break
            # add all multiples k*(a,b,c) with c*k <= limit
            k = 1
            while k * c <= limit:
                x, y, z = k*a, k*b, k*c
                x=int(x)
                y=int(y)
                z=int(z)
                # include both leg orders (x,y) and (y,x) if you need all pairs
                out.append(tuple(sorted((x, y)) + [z]))  # keep (x<=y) for consistency
                k += 1
        m += 1
    # make unique & sorted
    out = sorted(set(out))
    return out


def solve_meshgrid(upper_limit: int):
    # Create 2D grids for y and z
    y_vals = np.arange(1,upper_limit)
    z_vals = np.arange(1,upper_limit+1)
    Y, Z = np.meshgrid(y_vals, z_vals, indexing="ij")

    # Compute candidate x^2 = z^2 - y^2
    X2 = Z**2 - Y**2

    # Mask out negatives (invalid)
    mask = X2 >= 1

    # Compute integer square roots where valid
    X = np.sqrt(X2, where=mask, out=np.zeros_like(X2, dtype=float))

    # Mask those that are perfect squares
    mask &= (X == np.floor(X))

    # Extract valid integer triples
    x_vals = X[mask].astype(int)
    y_vals = Y[mask]
    z_vals = Z[mask]

    return list(zip(x_vals, y_vals, z_vals))


if __name__ == "__main__":
    upper_limit  = 200
    start_time=time()
    solution=solve_slow(upper_limit)
    #print("slow solution: ", solution)
    print("slow time: ", time()-start_time)

    start_time=time()
    solution=solve_chatGPT(upper_limit)
    #print("chatGPT solution: ", solution)
    print("chatGPT time: ", time()-start_time)

    start_time=time()
    solution=solve_meshgrid(upper_limit)                           
    #print("meshgrid solution: ", solution)
    print("meshgrid time: ", time()-start_time)


  
