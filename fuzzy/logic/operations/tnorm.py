def check_tnorm_none(a, b, funcname):
    if a is None or b is None:
        raise ValueError('None value passed to tnorm.' + funcname)


def fmin(a, b):
    check_tnorm_none(a, b, 'fmin')
    return a if a < b else b


def prod(a, b):
    check_tnorm_none(a, b, 'prod')
    return a * b


def lukasiewicz(a, b):
    check_tnorm_none(a, b, 'lukasiewicz')
    return max(a + b - 1, 0.0)


def drastic(a, b):
    check_tnorm_none(a, b, 'drastic')
    result = None
    if a == 1:
        result = b
    elif b == 1:
        result = a
    else:
        result = 0
    return result


def nilpotent(a, b):
    check_tnorm_none(a, b, 'nilpotent')
    result = None
    if a + b > 1:
        result = fmin(a, b)
    else:
        result = 0
    return result


def hamacher(a, b):
    check_tnorm_none(a, b, 'hamacher')
    result = None
    if a == b and b == 0:
        result = 0
    else:
        prod = a * b
        result = prod / (a + b - prod)
    return result
