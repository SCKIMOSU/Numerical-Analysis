def maximum(x, y):
    if x > y:
        return x
    elif x == y:
        return 'equal'
    else:
        return y

print('max is = ', maximum(4, 5))

# save as maximum.py


def say_hi():
    print('hi, this is my1 module')

__version__='0.1'

# save as my.py 

import my

my1.say_hi()
print('version', my.__version__)

# save as demo.py
