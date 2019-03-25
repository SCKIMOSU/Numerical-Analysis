import numpy as np
points = [0.01*i for i in np.arange(1,11)]

for x in points:
    print (x, np.sin(x), '%.2f' % (abs(x-np.sin(x))/np.sin(x)*100))
    # Formatting Numbers as Strings
    # print('The order total comes to %.2f' % 123.444)
    # The order total comes to 123.44
    # a = "abcdefghijklmnopqrstuvwxyz"
    # print('%.3s' % a)
    # abc
