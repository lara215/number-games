import numpy as np

def square(number: float) -> float:
    """Return the square of a given number."""
    return number * number

def solve_equation(y,z):
    return np.sqrt(square(z)-square(y))

def main():
    solutions=[]
    for z in range(upper_limit):
        for y in range(upper_limit):
            x= solve_equation(y,z)
            print(x)
            if x.is_integer():
                solutions.append([int(x),y,z])
    print(solutions)

if __name__ == "__main__":
    upper_limit  = 5
    main()
  
