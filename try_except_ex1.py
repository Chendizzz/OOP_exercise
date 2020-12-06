import os
'''Exercise 3-a: ZeroDivisionError Exception w/Try Except Statements
'''

a = 5
b = 0
try:
    result = a/b
except ZeroDivisionError:
    result = 'you cannot divide with zero'

print(result)

"""Exercise 3-b: pass statement inside Try Except Statements
.get() is not a list method. Place pass keyword to the right line so that program doesn't throw an error."""

a = [1, 2, 3]
try:
    a.get()
except AttributeError:
    a = a

print(a)

# solution:
a = [1, 2, 3]
try:
    a.get()

except:
    pass
print(a)

"""Exercise 3-c: except Exception w/ Try Except Statements
Place msg="You can't add int to string" to the right place so that program avoids BaseExceptionError.
"""

a = 'hallo world'
# a = 5
try:
    result = a+10
except Exception:
    raise 
    result = 'you cannot add int to string'
print(result)

# solution uses exceptException: ...

"""Exercise 3-d: IndexError w/ Try Except Statements
Place msg="You're out of list range" to avoid IndexError.
"""

lst = [5, 10, 20]

try:
    print(lst[5])
except Exception:
    raise IndexError('out of list range')
    # msg = 'you are out of list range'

# print(msg)
