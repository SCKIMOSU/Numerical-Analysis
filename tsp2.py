# https://ericphanson.com/blog/2016/the-traveling-salesman-and-10-lines-of-python/
import random, numpy, math, copy, matplotlib.pyplot as plt
cities = [random.sample(range(100), 2) for x in range(15)]
tour = random.sample(range(15),15)
for temperature in numpy.logspace(0,5,num=100000)[::-1]:
    [i,j] = sorted(random.sample(range(15),2))
    newTour =  tour[:i] + tour[j:j+1] +  tour[i+1:j] + tour[i:i+1] + tour[j+1:]

    if math.exp( ( sum([ math.sqrt(sum([(cities[tour[(k+1) % 15]][d] - cities[tour[k % 15]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]]) - sum([math.sqrt(sum([(cities[newTour[(k+1) % 15]][d] - cities[newTour[k % 15]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]])) / temperature) > random.random():
        tour = copy.copy(newTour);

plt.plot([cities[tour[i % 15]][0] for i in range(16)], [cities[tour[i % 15]][1] for i in range(16)], 'xb-');
plt.show()

xmin = min(pair[0] for pair in cities)
xmax = max(pair[0] for pair in cities)

ymin = min(pair[1] for pair in cities)
ymax = max(pair[1] for pair in cities)

def transform(pair):
    x = pair[0]
    y = pair[1]
    return [(x-xmin)*100/(xmax - xmin), (y-ymin)*100/(ymax - ymin)]

rescaled_cities = [ transform(b) for b in cities]

cities = [(random.random()/10, random.random()/10) for x in range(15)];
