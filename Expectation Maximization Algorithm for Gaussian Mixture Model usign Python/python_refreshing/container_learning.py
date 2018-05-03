xs = [3, 1, 2]
print(xs, xs[2])

print( xs[-1] )

xs[2] = 'foo'
print(xs)

xs.append('bar')
print( xs )


x = xs.pop()
print(x, xs)


nums = list( range(5) )
print( nums )
print( nums[2:4] )
print( nums[2:] )
print( nums[:4] )
print(nums[:])
print(nums[:-1])

nums[2:4] = [8, 9]
print(nums)



animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)


for idx, animal in enumerate(animals):
    print( '#%d: %s' % (idx+1, animal) )


nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)


# using list comprehension
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(squares, even_squares)


#dictionary
d = {'cat' : 'cute', 'dog' : 'funny'}
print( d['cat'] )
print( 'cat' in d )
d['fish'] = 'wet'
print( d['fish'] )
print( d.get('monkey', 'N/A') )
print( d.get('fish', 'N/A') )
del d['fish']
print( d.get('fish', 'N/A') )



# Loops
d = {'person' : 2, 'cat' : 4, 'spider' : 8}
for animal in d:
    legs = d[animal]
    print( ' A %s has %s legs' % (animal, legs) )


# Dictionay comprehension
nums = {0, 1, 2, 3, 4, 5}
even_nums_to_square = {x : x ** 2 for x in nums if x % 2 == 0}
print(even_nums_to_square)


# Set
animals = {'cat', 'dog'}
animals.remove("cat")


# Set comprehension
from math import sqrt
nums = { int( sqrt(x) )  for x in range(90) }
print( nums )



# tuples
d = { (x, x+1) : x for x in range(10) }
t = (5,6)
print( type(t) )
print( d[t] )
print( d[(1,2)] )



def sign(x):
    if x < 0:
        return 'negative'
    elif x == 0:
        return 'zero'
    else:
        return 'positive'


for x in [-1,0,1]:
    print(sign(x))


# Functions
def hello(name, loud=False):
    ret = "Hello, %s!" % name;
    if loud:
        ret = ret.upper()
    print(ret)

hello('Bob')
hello('Fred', True)



# classes
class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name;

    def greet(self, loud=False):
        toPrint = ( "Hello, %s!" % self.name );
        if loud:
            toPrint = toPrint.upper();
        print( toPrint )


greeter = Greeter('Fred')
greeter.greet()
greeter.greet(loud=True)