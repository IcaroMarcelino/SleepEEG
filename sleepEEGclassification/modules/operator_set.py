import math

def pdiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def plog(x):
    try:
        return math.log(abs(x))
    except:
        return 1

def psqrt(x):
    return abs(x)**(.5)

def F(x):
    return(x)
