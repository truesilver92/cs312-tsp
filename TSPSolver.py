#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq

#########################
# start of stuff for fancy (ACO)

alpha = 1
beta = 1
rho = 0.5
ants_count = 2000  # number of ants to use
Q = 1
max_iterations = 100
cost_pos = 0 # the index in the inner list where the cost of the edge is stored
ph_pos = 1 # the index in the inner list where the pharamone value of the edge is stored

def init_adjacencyMatrix(cities):
    initial_cost_matrix = np.arange(float(len(cities))**2 * 2).reshape(len(cities), len(cities), 2)# creates the matrix time: O(n^2) space: O(n^2)
    for i in range(len(cities)): # fill the matrix with the correct initial values time: O(n^2) space: O(1)
        for j in range(len(cities)):
            initial_cost_matrix[i][j] = np.arange(float(2.0))
            initial_cost_matrix[i][j][cost_pos] = cities[i].costTo(cities[j])
            initial_cost_matrix[i][j][ph_pos] = 1.0
            #print(initial_cost_matrix[i][j])
    return initial_cost_matrix
            

def init_ants(ants, cities):
    for a in ants:
        a.visited = []
        a.not_visited = list(cities)
        r = np.random.random_integers(0, len(cities) - 1)
        a.visited.append(a.not_visited[r])
        a.not_visited.pop(r)
        a.cost = 0

def calcProbability(from_city, to_city, m):
    return m[from_city._index][to_city._index][cost_pos]**(-beta) * m[from_city._index][to_city._index][ph_pos]**alpha
            
def dewTour(m, ant):
    while len(ant.not_visited) != 0:
        p_list = [] # put edge probabilities here
        last_visited = ant.visited[len(ant.visited) - 1]
        numerator_sum = 0.0

        for i in ant.not_visited:
            probability = calcProbability(last_visited, i, m)
            p_list.append(probability)
            numerator_sum += probability
        #print(p_list/numerator_sum)
        next = np.random.choice(ant.not_visited, 1, map(lambda x:x/numerator_sum, p_list))[0]
        #print(next._index)
        ant.visited.append(next)
        ant.not_visited.remove(next)
        ant.cost = ant.cost + m[last_visited._index][next._index][cost_pos]
    #print("ant.cost: %s" %ant.cost)

def updatePharamones(m, ants):
    for i in range(len(m)):
        for j in range(len(m[i])):
            #print("before: %f" %m[i][j][ph_pos])
            m[i][j][ph_pos] *= rho
            #print("after: %f" %m[i][j][ph_pos])

    print(min(map(lambda x:x.cost,ants)))

    for ant in ants:
        if ant.cost == float('inf'):
            continue
        ph_add = Q / ant.cost
        for i in range(len(m) - 1):
            m[ant.visited[i]._index][ant.visited[i + 1]._index][ph_pos] += ph_add
        m[ant.visited[len(ant.visited) - 1]._index][ant.visited[0]._index][ph_pos] += ph_add

def minTour(ants):
    results = None
    c = float('inf')
    for a in ants:
        #print(list(map(lambda x:x._index, a.visited)))
        sol = TSPSolution(a.visited)
        cost = sol.costOfRoute()
        #print("sol: %s" %list(map(lambda x:x._index, a.visited)))
        #print("cost: %f" %cost)
        if cost < c:
            #print("hi")
            results = sol
            c = cost
    return results

#########################
# end of fancy crap

def setColumn(cost_matrix, column, value):# sets the entire column to value time: O(n) space: O(1)
    for i in range(len(cost_matrix)):# time: O(n) space: O(1)
        cost_matrix[i][column] = value

def setRow(cost_matrix, row, value):# sets the entire row to a value, time: O(n) space: O(1)
    for i in range(len(cost_matrix)):# time: O(n) space: O(1)
        cost_matrix[row][i] = value

def reduceRow(cost_matrix, row_number):# reduces a row and returns the low_bound increase it causes time: O(n) space: O(1)
    m = min(cost_matrix[row_number])# get the amount to reduce by time: O(n) space: O(1)
    if m == float('inf'): # if the row is already cleared then just return
        return 0
    for i in range(len(cost_matrix[row_number])):# subtract the reduce amount time: O(n) space: O(1)
        cost_matrix[row_number][i] -= m
    return m

def reduceColumn(cost_matrix, column_number): # same as reduceRow except for the column time: O(n) space: O(1)
    column = []
    for i in range(len(cost_matrix)):
        column.append(cost_matrix[i][column_number])
    m = min(column)
    if m == float('inf'):
        return 0
    for i in range(len(cost_matrix)):
        cost_matrix[i][column_number] -= m
    return m

def reduceCostMatrix(cost_matrix):# does the reduceRow and reduceColumn on each row and then each column time: O(n^2) space: O(1)
    low_bound = 0
    for i in range(len(cost_matrix)):
        low_bound += reduceRow(cost_matrix, i)
    for i in range(len(cost_matrix)):
        low_bound += reduceColumn(cost_matrix, i)
    return low_bound


class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None

    def setupWithScenario( self, scenario ):
        self._scenario = scenario


    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour
        </summary>
        <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (
not counting initial BSSF estimate)</returns> '''
    def defaultRandomTour( self, start_time, time_allowance=60.0 ):

        results = {}


        start_time = time.time()

        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        while not foundTour:
            # create a random permutation
            perm = np.random.permutation( ncities )

            #for i in range( ncities ):
                #swap = i
                #while swap == i:
                    #swap = np.random.randint(ncities)
                #temp = perm[i]
                #perm[i] = perm[swap]
                #perm[swap] = temp

            route = []

            # Now build the route using the random permutation
            for i in range( ncities ):
                route.append( cities[ perm[i] ] )

            bssf = TSPSolution(route)
            #bssf_cost = bssf.cost()
            #count++;
            count += 1

            #if costOfBssf() < float('inf'):
            if bssf.costOfRoute() < np.inf:
                # Found a valid route
                foundTour = True
        #} while (costOfBssf() == double.PositiveInfinity);                // until a valid route is found
        #timer.Stop();

        results['cost'] = bssf.costOfRoute() #costOfBssf().ToString();                          // load results array
        results['time'] = time.time() - start_time
        results['count'] = count
        results['soln'] = bssf

       # return results;
        return results

    def initialize_matrix(self, cities):
        matrix = np.empty(shape=(len(cities), len(cities)))
        for i in range(0, len(cities)):
            for j in range(0, len(cities)):
                if i == j:
                    # puts infinity on the diagonals
                    matrix[i, j] = float("inf")
                else:
                    matrix[i, j] = cities[i].costTo(cities[j])
        return matrix
    def set_vals_to_infinity(self, matrix, ind_a, ind_b, length):
        matrix[ind_a,:] = float("inf") # time O(n)
        matrix[:,ind_b] = float("inf") # time O(n)
        matrix[ind_b, ind_a] = float("inf") # time O(1)

    def greedy( self, start_time, time_allowance=60.0 ):
        start_time = time.time()
        cities = self._scenario.getCities()
        solution = self.prims_alg(self.initialize_matrix(cities), cities, 0)
        for i in range(1, len(cities)):
            temp_solution = self.prims_alg(self.initialize_matrix(cities), cities, i)
            total = solution.costOfRoute()
            amount = temp_solution.costOfRoute()
            length = len(temp_solution.route)
            if temp_solution.costOfRoute() < solution.costOfRoute() and len(temp_solution.route) >= len(cities):
                solution = temp_solution
    
        results = {}
        results['cost'] = solution.costOfRoute()
        results['time'] = time.time() - start_time
        results['count'] = 1
        results['soln'] = solution
        return results


    def prims_alg(self, matrix, cities, city_index):
        # time O(n^2) space O(n)
        min_vals = np.argmin(matrix, axis=1)
        route = [] # array of indecies of the correct route
        route.append(city_index)
        index = min_vals[city_index]
        route.append(index)
        self.set_vals_to_infinity(matrix, 0, index, len(cities))
        self.set_vals_to_infinity(matrix, city_index, city_index, len(cities))
        # time O(n)
        while True:
            low = float("inf")
            temp_ind = index
            for i in range(0, len(cities)):
                if matrix[index, i] < low:
                    low = matrix[index, i]
                    temp_ind = i
            if low == float("inf"):
                break
            else:
                self.set_vals_to_infinity(matrix, index, temp_ind, len(cities))
                index = temp_ind
                route.append(index)
        # route.append(0)
        # print(matrix)
        city_route = [] # take the indecies of the route, and make an array of cities at those indecies
        # time O(n) space O(n)
        for r in route:
            city_route.append(cities[r])
    
        return TSPSolution(city_route)


    def branchAndBound( self, start_time, time_allowance=60.0 ):# the branchAndBound algorithm
        start_time = time.time()
        cities = self._scenario.getCities()
        initial_cost_matrix = np.arange(float(len(cities))**2).reshape(len(cities), len(cities))# creates the matrix time: O(n^2) space: O(n^2)
        for i in range(len(cities)): # fill the matrix with the correct initial values time: O(n^2) space: O(1)
            for j in range(len(cities)):
                initial_cost_matrix[i][j] = cities[i].costTo(cities[j])

        low_bound = reduceCostMatrix(initial_cost_matrix)# reduce the initial matrix time: O(n^2) space: O(1)
        
        pq = [] # this is where the priority queue is set up
        heapq.heappush(pq, (len(cities) - 1, low_bound, [0], initial_cost_matrix)) # push the first state, it is a tuple starting with the "inverse" of the depth of the state, next is the lower bound, next is the list of visited nodes represented, next is the cost matrix of the state time: O(nlogn) space: O(n^2)
        pelim_results = self.defaultRandomTour(time.time())# get the bssf to start working off of
        bssf = {}
        bssf['cost'] = pelim_results['cost']
        #bssf['cost'] = float('inf')
        bssf['soln'] = pelim_results['soln']
        bssf['count'] = 1

        states_created = 1
        states_pruned = 0
        states_max = 0

        while len(pq) != 0 and time.time() - start_time < 60:# keep working until we exhaust states or run out of time 
            state = heapq.heappop(pq)# pop from the heap time: O(logn) space: O(1)
            depth = len(cities) - state[0]# convert the stored depth (which is lower the deeper the state) to actual depth time: O(1) space: O(1)
            low_bound = state[1]
            visited = state[2]
            cost_matrix = state[3]
            if depth == len(cities): # this means that the next node is the last node
                if low_bound < bssf['cost']: # see if this is a better path
                    #bssf['cost'] = low_bound
                    bssf['soln'] = []
                    for i in visited:# convert the city indexes into the city objects time: O(n) space: O(1)
                        bssf['soln'].append(cities[i])
                    #bssf['soln'].append(cities[0])
                    bssf['soln'] = TSPSolution(bssf['soln'])
                    bssf['cost'] = bssf['soln'].costOfRoute()
                    bssf['count'] += 1# keep track of how many times bssf is updated
                continue

            for i in range(1, len(cities)):# make a new state for every possible path time: O(n^3) space: O(n^3)
                nlow_bound = low_bound
                if cost_matrix[visited[len(visited) - 1]][i] != float('inf'):# if not an invalid location
                    new_cm = np.array(cost_matrix)# make a copy of the cost matrix time: O(n^2) space: O(n^2)
                    nlow_bound += new_cm[visited[len(visited) - 1]][i] # add the cost to go to the node
                    setRow(new_cm, visited[len(visited) - 1], float('inf'))# clear out the from node costs
                    setColumn(new_cm, i, float('inf'))# clear out any going to this new node again
                    new_cm[i][visited[len(visited) - 1]] = float('inf')
                    nlow_bound += reduceCostMatrix(new_cm)# reduce the matrix again time: O(n^2) space: O(1)
                    states_created += 1
                    if nlow_bound < bssf['cost']:
                        nvisited = list(visited)
                        nvisited.append(i)
                        heapq.heappush(pq, (len(cities) - depth - 1, nlow_bound, nvisited, new_cm)) # push the new state, it has relative data to the current one as far as depth goes
                        states_max = max(states_max, len(pq))
                        continue
                    states_pruned += 1
                    # if not better it was just pruned
                # we can't go to this (i) verticy. It is us or already visited

        bssf['time'] = time.time() - start_time
        print("states_created: " + str(states_created))
        print("states_pruned: " + str(states_pruned))
        print("states_max: " + str(states_max))
        return bssf

    def fancy( self, start_time, time_allowance=60.0 ):
        # set up of local variables
        start_time = time.time()
        cities = self._scenario.getCities()
        min_cost = float('inf')
        m = init_adjacencyMatrix(cities) # setup the edge matrix
        ants = []
        
        # setup ant objects
        for a in range(ants_count):
            ants.append(Ant())

        bssf = None
        
        for i in range(max_iterations):
            print("iterations left:")
            print(max_iterations - i)
            init_ants(ants, cities)
            for j in ants:
                dewTour(m, j)
            updatePharamones(m, ants)
            potential_bssf = minTour(ants)
            if bssf == None:
                bssf = potential_bssf
                continue
            if potential_bssf.costOfRoute() < bssf.costOfRoute():
                bssf = potential_bssf
            
        #print(m)
        #bssf = minTour(ants)
        results = {}
        results['time'] = time.time() - start_time
        results['soln'] = bssf
        results['cost'] = bssf.costOfRoute()
        results['count'] = 1337
        return results
