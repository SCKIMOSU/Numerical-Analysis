# https://codereview.stackexchange.com/questions/143643/traveling-salesman-problem-genetic-algorithm
from random import sample
from random import random
from random import uniform
from random import shuffle
from math import sqrt
from time import time
from itertools import permutations
import matplotlib.pyplot as plt
import cProfile
import pstats
import io


class NodesLeastDistanceGA:
    """ Traveling salesman problem genetic algorithm """

    def __init__(self, parent, side, verbose=False):
        """ Constructor """
        self._parent = parent
        self._side = side

        self._mutate_rate = 0.07
        self._population_size = 60 if len(parent) > 10 else 10
        self._new_generation_size = self._population_size*2
        self._rounds = 200
        self._genlen = len(parent)
        self._verbose = verbose
        self._cached_distances = {}
        self._cached_fitness = {}

    def algorithm(self):
        """
        Initialize population of lists and sets fittest.

        Yields a new population and mutate a small amount of the individuals to avoid getting stuck.

        Thorough elitism the most fitest individual is carried through the rounds.
        """

        population = self.generate_population()
        fitest = min(population, key=self.fitness)

        total_time = time()
        for r in range(self._rounds):
            new_pop = []
            while len(new_pop) < self._new_generation_size:
                father = self.select(population)
                mother = self.select(population)
                child = self.crossover(father, mother)
                if child not in new_pop:
                    new_pop += [child]
                    continue
                for i in range(len(new_pop)):
                    if random() < self._mutate_rate:
                        new_pop[i] = self.mutate(new_pop[i])

            new_fittest = min(population, key=self.fitness)
            if self.fitness(fitest) > self.fitness(new_fittest):
                fitest = new_fittest
            if r % 50 == 0:
                print(r, self.fitness(min(population, key=self.fitness)))

            population = self.selection(new_pop)
            if fitest not in population:
                population += [fitest]

        self.result(population, fitest, total_time)

    def result(self, population, fitest, total_time):
        if self._verbose:
            for ind in sorted(population, key=self.fitness):
                print("Path: {}, Fitness: {:.3f}".format(ind, self.fitness(ind)))

            print("Cached-> Fitness:{}, Distances: {}".format(len(self._cached_fitness), len(self._cached_distances)))
        print("Execution Time: {:.3f}s".format(time() - total_time))
        print("Best path found: {}, fitness: {:.3f}".format(fitest, self.fitness(fitest)))
        if self._verbose:
            self.plot(fitest)

    def selection(self, new_pop):
        """ Determines which individuals that survives. Shuffle to destroy symmetry selection over rounds. """

        shuffle(new_pop)
        pop = []
        for _ in range(self._population_size):
            survivor = self.select(new_pop)
            new_pop.remove(survivor)
            pop += [survivor]
        return pop

    def select(self, pop):
        """ Selects a individual that might have a low fitness. """

        pop_total_fit = sum(1.0 / self.fitness(p) for p in pop)
        limit = uniform(0.0, pop_total_fit)
        c = 0
        for p in pop:
            c += 1 / self._cached_fitness[hash(tuple(p))]
            if c > limit:
                return p

    def fitness(self, child):
        """
        Returns the fitness of a individual if it has been calculated.
        Else it calculates the distance between each node and sum it up, cache it,
        this is the fitness of current individual. In this case a low
        fitness is a good fitness.
        """

        h = hash(tuple(child))
        if h in self._cached_fitness.keys():
            return self._cached_fitness[h]

        distance = 0
        for i in range(len(child)-1):
            distance += self.point_distance(child[i], child[i+1])
        self._cached_fitness[h] = distance
        return distance

    @staticmethod
    def crossover(father, mother):
        """
        Cross two individual thorough a gen by gen approach.
        For readability and optimization this function is kept ugly.
        """
        child = [None]*len(father)
        rate = 0.5
        for gen in father:
            parent, other_parent = (father, mother) if random() > rate \
                else (mother, father)

            key = None
            for key, value in enumerate(parent):
                if value == gen:
                    break
            if not child[key]:
                child[key] = gen
                continue
            for key, value in enumerate(other_parent):
                if value == gen:
                    break
            if not child[key]:
                child[key] = gen
                continue

            for key, value in enumerate(child):
                if not value:
                    child[key] = gen
                    break
        return child

    @staticmethod
    def mutate(child):
        """ Swaps place of two gens. """

        i1, i2 = sample(range(1, len(child)-1), 2)
        child[i1], child[i2] = child[i2], child[i1]
        return child

    def point_distance(self, p1, p2):
        """ Calculates the distance between two points and cache it. """

        nodes = hash((p1, p2))
        if nodes in self._cached_distances.keys():
            return self._cached_distances[nodes]
        d = sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        self._cached_distances[nodes] = d
        return d

    def generate_population(self):
        """ Creates the initial populations. """

        pop = [self._parent[:1]+sample(
            self._parent[1:-1], len(self._parent)-2)+self._parent[-1:]
               for _ in range(self._population_size)]
        for p in pop:
            h = hash(tuple(p))
            self._cached_fitness[h] = self.fitness(p)
        return pop

    def plot(self, path):
        plt.axis([-1, self._side+1, -1, self._side+1])
        for i in range(0, len(path)-1):
            plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], color="brown", marker="o")
        plt.show()

    def correct_ans(self, nodes):
        """ Very bad bruteforce approach. """

        if len(nodes) > 11:
            print("Not a good idea.")
            raise Exception
        start = time()
        best = nodes
        for path in permutations(nodes[1:-1]):
            path = nodes[:1]+list(path)+nodes[-1:]
            if self.fitness(best) > self.fitness(path):
                best = path
        print("Correct ans should be: {}: fitness: {:.3f}, solutions: {}".format(
            str(best), self.fitness(best), len(list(permutations(nodes)))))
        print("Bruteforce approch: {:.3f}".format(time()-start))
        self.plot(best)

    def profile(self):
        pr = cProfile.Profile()
        pr.enable()
        self.algorithm()
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats = s.getvalue().split("\n")
        stats = "\n".join([x for x in stats if "GridGA.py" in x])
        print(stats)


def main():
    nodes = [(13, 2), (1, 12), (12, 5), (19, 6), (2, 10), (15, 15), (5, 11), (17, 9),
             (10, 18), (17, 5), #(13, 12), #(1, 17), (2, 6), (7, 16), (19, 2), (3, 7),
             #(10, 9), (5, 19), (1, 2), (9, 2)
             ]
    nodes += nodes[:1]
    ga = NodesLeastDistanceGA(nodes, 20)
    ga.profile()


if __name__ == '__main__':
    main()
