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

    def reduceRow(cost_matrix, row_number):
        m = min(cost_matrix[row_number])
        for i in len(cost_matrix[row_number]):
            cost_matrix[row_number][i] -= m
        return m

    def reduceColumn(cost_matrix, column_number):
        colunm = []
        for i in range(len(cost_matrix)):
            column.append(cost_matrix[i][column_number])
        m = min(column)
        for i in range(len(cost_matrix)):
            cost_matrix[i][column_number] -= m
        return m

    def reduceCostMatrix(cost_matrix):
        low_bound = 0
        for i in range(len(cost_matrix)):
            low_bound += reduceRow(cost_matrix)
        for i in range(len(cost_matrix)):
            low_bound += reduceColumn(cost_matrix)
        return low_bound

    def branchAndBound( self, start_time, time_allowance=60.0 ):
        cities = self._scenario.getCities()
        initial_cost_matrix = np.arange(len(cities)**2).reshape(len(cities), len(cities))
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i == j:
                    initial_cost_matrix[i][j] = float('inf')
                    continue
                initial_cost_matrix[i][j] = cities[i].costTo(cities[j])

        low_bound = reduceCostMatrix(initial_cost_matrix)

    def fancy( self, start_time, time_allowance=60.0 ):
        pass


