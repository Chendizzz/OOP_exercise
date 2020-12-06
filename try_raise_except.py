"""
ex1: raise with parameters, it should be caught by except
"""

try:
    x = input('please input a number')
    if not x.isdigit():
        raise ValueError('x must be digit')

except ValueError as e:
    print('wrong input: ', repr(e))

