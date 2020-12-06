import os
"""
Exercise 1-a: New Class Which Inherits
A new class named F14 is initated for you which inherits from the parent class Jets.
Instead of taking parameters other than self such as name and country, 
initiate the new class so that name is always fixed to "F14" and origin is always fixed to "USA"

Make sure the new class has its own initation method (constructor or __init__) 
ehich takes only one parameter:self and overrides name and origin attributes as those don't change for an F14 fighter jet.
"""


class Jets:

    def __init__(self, name, country):
        self.type = "Jet"
        self.area = "Air"
        self.name = name
        self.origin = country


# Type your code below inside the new class

class F14(Jets):
    def __init__(self):
        self.name = 'F14'
        self.origin = 'USA'


a = F14()
print(a.origin)
print(a.name)

"""Exercise 1-b: __init__ Constructor and New Attributes
A new class named F14 is initated for you which inherits from the parent class Jets.
Under this new class, define 4 methods regarding engine, seat, tail and speed.
Make sure the new class has its own initation method (constructor or __init__) 
ehich takes only one parameter:self and overrides name and origin attributes to "F14" and "USA" always as those don't change for an F14 fighter jet.

Also add 3 more attributes:
engine, seat, tail which are all 2 by default.
"""


class Jets:
    model = None
    country = 0

    def __init__(self, name, country):
        self.type = "Jet"
        self.area = "Air"
        self.name = name
        self.origin = country


# Type your code here

class F14(Jets):
    def __init__(self):
        self.engine = 2
        self.seat = 2
        self.name = 'F14'
        self.origin = 'USA'

a = F14()
print(a.engine)
print(a.seat)