from math import exp


def surface_function(x, y):
    """
    Surface area function.
    This function is equivalent to octave's peaks() function.
    https://octave.sourceforge.io/octave/function/peaks.html
    """
    result = (
        3 * (1 - x) ** 2 * exp(-(x**2) - (y + 1) ** 2)
        - 10 * (x / 5 - x**3 - y**5) * exp(-(x**2) - y**2)
        - 1 / 3 * exp(-((x + 1) ** 2) - y**2)
    )

    return result
