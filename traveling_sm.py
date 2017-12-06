import matplotlib.pyplot as plt
import random
import math
import numpy
from operator import attrgetter
import matplotlib.animation as animation
import time
#class
class City:
    
    def __init__(self, single, xval, yval):
        self.single = single
        self.xval = xval
        self.yval = yval
        

class Cities(list):
        
    journey=[]    
    journey.append(City('A', 2, 3))
    journey.append(City('B', 9, 9))
    journey.append(City('C', 6, 7))
    journey.append(City('D', 1, 9))
    journey.append(City('E', 3, 8))
    journey.append(City('F', 5, 8))
    journey.append(City('G', 4, 4))
    journey.append(City('H', 10, 6))
    journey.append(City('I', 7, 5))
    journey.append(City('J', 8, 2))
        
    def __init__(self):
        #self.travel = []
        self.fitness = float(-1)
        self.distance = float(0)
        self.journey = random.sample(self.journey, len(self.journey))
        
def meta_print(pop):
    for i, cities in enumerate(pop):
        print("Config no.: %d Fitness: %f Distance: %f"%(i, cities.fitness, cities.distance))
        for city in cities.journey:
            print(("City: %s X_coor: %f Y_coor: %f" %(city.single, city.xval, city.yval)))
        print()
    print() 

    
def find_max(pop):
    temp = []
    for item in pop:
        temp.append(item.distance)
    return(max(temp))

def find_min(pop):
    temp = []
    for item in pop:
        temp.append(item.distance)
    return(min(temp))

def find_avg(pop):
    temp = []
    for item in pop:
        temp.append(item.distance)
    return(sum(temp)/len(temp))


def graphs(max_array,min_array, avg_array, max_route, min_route):

    max_chart=plt.figure(1)
    plt.plot(max_array, 'o')
    plt.xlabel('Generacje')
    plt.ylabel('Wartosci maksimum')
    
    min_chart=plt.figure(2)
    plt.plot(min_array, 'o')
    plt.xlabel('Generacje')
    plt.ylabel('Wartosci minimum')
    
    avg_chart=plt.figure(3)
    plt.plot(avg_array, 'o')
    plt.xlabel('Generacje')
    plt.ylabel('Wartosci srednie')
    
    max_route = sorted(max_route, key=lambda cities: cities.distance)
    min_route = sorted(min_route, key=lambda cities: cities.distance)
    
    max_route = max_route[-1]
    min_route = min_route[0]
    
    xmax=[]
    ymax=[]
    xmin=[]
    ymin=[]
    
    for city in max_route.journey:
        xmax.append(city.xval)
        ymax.append(city.yval)
        
    max_road=plt.figure(4)
    plt.plot(xmax,ymax, 'o-', color="xkcd:light red")
    plt.title('Droga maksymalna')
    for city in max_route.journey:
        plt.annotate(city.single, xy=(city.xval, city.yval))
    
    for city in min_route.journey:
        xmin.append(city.xval)
        ymin.append(city.yval)
        
    min_road=plt.figure(5)
    plt.plot(xmin,ymin, 'o-', color="xkcd:bright green")
    plt.title('Droga minimalna')
    for city in min_route.journey:
        plt.annotate(city.single, xy=(city.xval, city.yval))    
    
    return(max_chart,min_chart,avg_chart, max_road, min_road)
    
    
def population_creator(size):
    population=[]

    for x in range (size):
        population.append(Cities())   
    return(population)   

def dist_calc(pop):
    tempx=0
    tempy=0
    for cities in pop:
        for i in range (len(cities.journey)-1):
            tempx = abs(cities.journey[i+1].xval - cities.journey[i].xval)
            tempy = abs(cities.journey[i+1].yval - cities.journey[i].yval)
            cities.distance += math.sqrt(tempx*tempx+tempy*tempy)  
    return(pop)
    
def fitness(pop):
    pop = dist_calc(pop)
    pop = sorted(pop, key=lambda cities: cities.distance)
    for cities in pop:
        cities.fitness = pop[-1].distance - cities.distance + 1
    
    return(pop)
    
def select(pop,weights):
    parents = []        
    while True:
        idx1 = numpy.random.choice(len(pop), p = weights)
        idx2 = numpy.random.choice(len(pop), p = weights)
        parent1 = pop[idx1]
        parent2 = pop[idx2]
        if idx1 != idx2:
            break
    parents.append(parent1)
    parents.append(parent2)
    
    return(parents)


 
def _repeated(element, collection):
    c = 0
    for e in collection:
        if e == element:
            c += 1
    return c > 1
 
def _swap(data_a, data_b, cross_points):
    c1, c2 = cross_points
    new_a = data_a[:c1] + data_b[c1:c2] + data_a[c2:]
    new_b = data_b[:c1] + data_a[c1:c2] + data_b[c2:]
    return new_a, new_b
 
 
def _map(swapped, cross_points):
    n = len(swapped[0])
    c1, c2 = cross_points
    s1, s2 = swapped
    map_ = s1[c1:c2], s2[c1:c2]
    for i_chromosome in range(n):
        if not c1 < i_chromosome < c2:
            for i_son in range(2):
                while _repeated(swapped[i_son][i_chromosome], swapped[i_son]):
                    map_index = map_[i_son].index(swapped[i_son][i_chromosome])
                    swapped[i_son][i_chromosome] = map_[1-i_son][map_index]
    return s1, s2
 
 
def pmx(parent_a, parent_b):
    assert(len(parent_a) == len(parent_b))
    n = len(parent_a)
    cross_points = sorted([random.randint(0, n) for _ in range(2)])
    swapped = _swap(parent_a, parent_b, cross_points)
    mapped = _map(swapped, cross_points)
 
    return mapped
    
def mutation(children,mut):
    for child in children:
        if random.uniform(0.0,1.0) <= mut:
            while True:
                idx1 = random.randint(0,len(child.journey)-1)
                idx2 = random.randint(0,len(child.journey)-1)
                if idx1 != idx2:
                    break
            child.journey[idx1], child.journey[idx2] = child.journey[idx2], child.journey[idx1]
    return(children)
    
def weights_calc(size):
    weights = []
    temp = 1.0/size
    los = 0
    for x in range(size):
        weights.append(temp)
    for indx in range(int(size/2)):
        los = random.uniform(0.0, temp)
        weights[indx] = weights[indx] + los
        weights[-indx] = weights[-indx] - los
    weights.sort(reverse=True)
    return(weights)

def crossover(pop, mut):
    wghts = weights_calc(len(pop))
    children = []
    for _  in range(int(len(pop)/2)):
        parents = select(pop, wghts)
        off=pmx(parents[0].journey, parents[1].journey)
        for of in off:
            child=Cities()
            child.journey = of
            children.append(child)
    children = mutation(children, mut)
    return children

#parameters
pop_size = 50
generations = 100
mut = 0.8


max_array = []
min_array = []
avg_array = []
max_route = []
min_route = []
popul = population_creator(pop_size)


for i in range(0,generations):
    print(i)
    popul = fitness(popul)
    max_array.append(find_max(popul))
    min_array.append(find_min(popul))
    avg_array.append(find_avg(popul))
    max_route.append(popul[-1])
    min_route.append(popul[0])
    popul = crossover(popul, mut)
    

graphs(max_array,min_array, avg_array, max_route, min_route)
    







