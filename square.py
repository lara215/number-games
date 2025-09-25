def square(number: float) -> float:
    """Return the square of a given number."""
    return number * number


if __name__ == "__main__":
    num = 3.3
    result = square(num)
    print(f"The square of {num} is {result}")
