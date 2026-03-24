def quad_darkening(x, a, b, c):
    return a * (1.0 - b * (1.0 - x) - c * (1.0 - x)**2)

def quad_darkening_two(x, b, c):
    return 1.0 - b * (1.0 - x) - c * (1.0 - x)**2
