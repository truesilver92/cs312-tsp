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

def setColumn(cost_matrix, column, value):
    for i in range(len(cost_matrix)):
        cost_matrix[i][column] = value

def setRow(cost_matrix, row, value):
    for i in range(len(cost_matrix)):
        cost_matrix[row][i] = value

def reduceRow(cost_matrix, row_number):
    m = min(cost_matrix[row_number])
    if m == float('inf'):
        return 0
    for i in range(len(cost_matrix[row_number])):
        cost_matrix[row_number][i] -= m
    return m

def reduceColumn(cost_matrix, column_number):
    column = []
    for i in range(len(cost_matrix)):
        column.append(cost_matrix[i][column_number])
    m = min(column)
    if m == float('inf'):
        return 0
    for i in range(len(cost_matrix)):
        cost_matrix[i][column_number] -= m
    return m

def reduceCostMatrix(cost_matrix):
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



    def greedy( self, start_time, time_allowance=60.0 ):
        pass


    def branchAndBound( self, start_time, time_allowance=60.0 ):
        start_time = time.time()
        cities = self._scenario.getCities()
        initial_cost_matrix = np.arange(float(len(cities))**2).reshape(len(cities), len(cities))
        for i in range(len(cities)):
            for j in range(len(cities)):
                initial_cost_matrix[i][j] = cities[i].costTo(cities[j])

        low_bound = reduceCostMatrix(initial_cost_matrix)
        
        pq = []
        heapq.heappush(pq, (len(cities) - 1, low_bound, [0], initial_cost_matrix))
        pelim_results = self.defaultRandomTour(time.time())
        bssf = {}
        bssf['cost'] = pelim_results['cost']
        bssf['cost'] = float('inf')
        #bssf['soln'] = pelim_results['soln']
        bssf['soln'] = [0,1,2,3]

        while len(pq) != 0 and time.time() - start_time < 60:
            state = heapq.heappop(pq)
            depth = len(cities) - state[0]
            low_bound = state[1]
            visited = state[2]
            print(state)
            cost_matrix = state[3]
            if depth == len(cities) - 1: # this means that the next node is the last node
                if low_bound < bssf['cost']: # see if this is a better path
                    bssf['cost'] = low_bound
                    bssf['soln'] = visited
                continue

            for i in range(1, len(cities)):
                nlow_bound = low_bound
                if cost_matrix[visited[len(visited) - 1]][i] != float('inf'):
                    new_cm = np.array(cost_matrix)
                    nlow_bound += new_cm[visited[len(visited) - 1]][i]
                    setRow(new_cm, visited[len(visited) - 1], float('inf'))
                    setColumn(new_cm, i, float('inf'))
                    new_cm[i][visited[len(visited) - 1]] = float('inf')
                    nlow_bound += reduceCostMatrix(new_cm)
                    if nlow_bound < bssf['cost']:
                        nvisited = list(visited)
                        nvisited.append(i)
                        heapq.heappush(pq, (len(cities) - depth - 1, nlow_bound, nvisited, new_cm))
                    # if not better it was just pruned
                # we can't go to this (i) verticy. It is us or already visited

        csoln = []
        for i in bssf['soln']:
            csoln.append(cities[i])
            
        route = TSPSolution(csoln)
        results = {}
        results['soln'] = route
        results['cost'] = route.costOfRoute()
        results['time'] = time.time() - start_time
        results['count'] = 7
        return results

    def fancy( self, start_time, time_allowance=60.0 ):
        pass


