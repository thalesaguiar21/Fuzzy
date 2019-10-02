def check_none_tconorm(a, b, funcname):
    if a is None or b is None:
        raise ValueError('None value passed to tconorm.' + funcname)


def fmax(a, b):
    check_none_tconorm(a, b, 'fmax')
    return a if a > b else b


def probabilistic_sum(a, b):
    check_none_tconorm(a, b, 'probabilistic_sum')
    return a + b - a * b


def bounded_sum(a, b):
    check_none_tconorm(a, b, 'bounded_sum')
    return min(a + b, 1.0)


def drastic(a, b):
    check_none_tconorm(a, b, 'drastic')
    result = None
    if a == 0:
        result = b
    elif b == 0:
        result = a
    else:
        result = 1.0
    return result


def nilpotent_max(a, b):
    check_none_tconorm(a, b, 'nilpotent_max')
    result = fmax(a, b)
    return result if a + b < 1.0 else 1.0


def einstein_sum(a, b):
    check_none_tconorm(a, b, 'einstein_sum')
    return (a + b) / (1.0 + a * b)
