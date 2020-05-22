def fmin(a, b):
    return a if a < b else b


def prod(a, b):
    return a * b


def lukasiewicz(a, b):
    return max(a + b - 1, 0.0)


def drastic(a, b):
    result = 0
    if a == 1:
        result = b
    elif b == 1:
        result = a
    return result


def nilpotent(a, b):
    result = 0
    if a + b > 1:
        result = fmin(a, b)
    return result


def hamacher(a, b):
    if a == b and b == 0:
        result = 0
    else:
        prod = a * b
        result = prod / (a + b - prod)
    return result
