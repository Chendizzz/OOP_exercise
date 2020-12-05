import numpy as np

'Exercise Question 1: Create a Vehicle class with max_speed and mileage instance attributes'


class Vehicleq1:
    def __init__(self):
        self.max_speed = None
        self.mileage = None


'Exercise Question 2: Create a Vehicle class without any variables and methods'


class Vehicleq2:
    def __init__(self):
        pass


# answer
class Verhicleq2a:
    pass


'Exercise Question 3: Create child class Bus that will inherit all of the variables and methods of the Vehicle class'


class Vehicleq3:
    def __init__(self, max_speed, mileage):
        self.max_speed = max_speed
        self.mileage = mileage


"""class Bus(Vehicleq3):
    def __init__(self, max_speed, mileage):
        self.max_speed = max_speed
        self.mileage = mileage"""


# answer
class Bus(Vehicleq3):
    pass


bus = Bus(50, 20000)
print(bus.max_speed, bus.mileage)

"""Exercise Question 5: Define property that should have the same value for every class instance, 
Define a class attribute”color” with a default value white. I.e., Every Vehicle should be whit'"""


class Vehicleq5:
    def __init__(self, name, max_speed, mileage):
        self.name = name
        self.max_speed = max_speed
        self.mileage = mileage
        self.color = 'white'


class Bus(Vehicleq5):
    pass


class Car(Vehicleq5):
    pass


car = Car('AudiQ5', 280, 1000)
bus = Bus('Volvo', 180, 10000)
print('car color', car.color, 'name: ', car.name, 'max speed: ', car.max_speed, 'mileage: ', car.mileage,
      'bus color', bus.color, 'name:', bus.name, 'max speed: ', bus.max_speed, 'mileage: ', bus.mileage,)

# solution
"""Variables created in .__init__() are called instance variables. 
An instance variable’s value is specific to a particular instance of the class. For example, in the solution, 
All Vehicle objects have a name and a max_speed, but the name and max_speed variables’ values will vary depending on the Vehicle instance.

On the other hand, class attributes are attributes that have the same value for all class instances. 
You can define a class attribute by assigning a value to a variable name outside of .__init__()."""


class Vehicleq5a:
    color = 'white'

    def __init__(self, name, max_speed, mileage):
        self.name = name
        self.max_speed = max_speed
        self.mileage = mileage


class Bus(Vehicleq5):
    pass


class Car(Vehicleq5):
    pass


School_bus = Bus("School Volvo", 180, 12)
print(School_bus.color, School_bus.name, "Speed:", School_bus.max_speed, "Mileage:", School_bus.mileage)

car = Car("Audi Q5", 240, 18)
print(car.color, car.name, "Speed:", car.max_speed, "Mileage:", car.mileage)

"""Exercise Question 6: Class Inheritance, Create a Bus child class that inherits from the Vehicle class. 
The default fare charge of any vehicle is seating capacity * 100. If Vehicle is Bus instance, 
we need to add an extra 10% on full fare as a maintenance charge. 
So total fare for bus instance will become the final amount = total fare + 10% of the total fare."""


class Vehicleq6:
    def __init__(self, name, mileage, capacity):
        self.name = name
        self.mileage = mileage
        self.capacity = capacity

    def fare(self):
        return self.capacity * 100


# override
class Bus(Vehicleq6):
    def fare(self):
        return self.capacity*100+self.capacity*10


School_bus = Bus("School Volvo", 12, 50)
print("Total Bus fare is:", School_bus.fare())

'''Exercise Question 7: Determine which class a given Bus object belongs to (Check type of a object)
'''


class Vehicleq7:
    def __init__(self, name, mileage, capacity):
        self.name = name
        self.mileage = mileage
        self.capacity = capacity

    def determine_type(self):
        return self.__class__


class Bus(Vehicleq7):
    pass


School_bus = Bus("School Volvo", 12, 50)
print(School_bus.determine_type())

# solution
''' use the predefined type() function
'''


class Vehicleq7a:
    def __init__(self, name, mileage, capacity):
        self.name = name
        self.mileage = mileage
        self.capacity = capacity


class Bus(Vehicleq7a):
    pass


School_bus = Bus("School Volvo", 12, 50)
print(type(School_bus))

# actually it shows the same result as my answer

'''Exercise Question 8: Determine if School_bus is also an instance of the Vehicle class
'''


class Vehicleq8:
    def __init__(self, name, mileage, capacity):
        self.name = name
        self.mileage = mileage
        self.capacity = capacity


class Bus(Vehicleq8):
    pass


School_bus = Bus("School Volvo", 12, 50)

# solution
# use Python's built-in isinstance() function
print(isinstance(School_bus, Vehicleq8))