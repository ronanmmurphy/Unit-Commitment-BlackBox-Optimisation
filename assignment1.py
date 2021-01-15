"""
Ronan murphy 15397831
Assignment 1 - Optimisation 
"""
import numpy as np
import pandas as pd
import random
import statistics as st
import matplotlib.pyplot as plt
import math
plant_info = pd.read_csv("plant_info.csv", delimiter=",")
demand = np.loadtxt("demand.csv")
solar_curve = np.loadtxt("solar_curve.csv")
nplants = len(plant_info["type"])
nhours = 24

def production_cost(x):
        # just cost per MW (arranged as (24 x 10)) times x
        cost_per_MW = plant_info["cost"].values # extract the np array
        return np.sum(cost_per_MW.reshape((-1, 1)) * x)

def supply(x):
        return np.sum(x, axis=0)

def supply_demand_penalty(x):
        # get the summed supply
        sup = supply(x)

        # where sup - demand > 0 -> square it
        excess_supply = np.sum(np.maximum(sup - demand, 0) ** 2)
        # where demand - sup > 0 -> cube it
        excess_demand = np.sum(np.maximum(demand - sup, 0) ** 3)

        return excess_supply + excess_demand

def change(x):
        # make y as a copy of x, shifted right (circularly)
        y = np.zeros_like(x)
        y[:-1] = x[1:]
        y[-1] = x[0]
        return np.sum(np.abs(x - y))

def output_change_penalty(x):
        # sum of output change penalties for solid fuel generators
        pen = 0
        for plant in range(nplants):
                if plant_info["type"][plant] == "solid":
                        pen += change(x[plant])
        return pen

def f(x):

        # x is a 1D array, so we reshape
        x = x.reshape((nplants, nhours))

        # ensure x doesn't try to exceed capacity as follows

        # first tile capacity from one value per plant
        # to one value per plant per hour
        capacity = np.tile(plant_info["capacity"].values.reshape((-1, 1)), (1, nhours))
        capacity = capacity.astype(float)

        # apply the solar curve
        for plant in range(nplants):
                if plant_info["type"][plant] == "solar":
                        capacity[plant, :] *= solar_curve

        # now just clamp x
        x = np.minimum(capacity, x)

        return production_cost(x) + supply_demand_penalty(x) + output_change_penalty(x)

#initaliser funtion to assign 240 random int values as x values to function 
def init():
    return (100*np.random.random(nhours*nplants)).astype(int)
#the function used to change the step size of each iteration, step is delta variable which is whole number
def nbr(x):
    delta = 5
    x = x.copy()
    i = random.randrange(nhours*nplants)
    x[i] = x[i] + np.random.normal(0, delta)
    return x
#generic hill climb algorithm with 50,000 iterations
def hill_climb(init, nbr, f, its=50000):
    stats = []#stats variable for the graphing
    x = init() # initalise x
    fx = f(x)  # avoid re-calculating f 
    for i in range(its):#iterate 50,000
        xnew = nbr(x) 
        fxnew = f(xnew) # avoid re-calculating f 
        if fxnew < fx: # accept only positive improvements
            x = xnew
            fx = fxnew
        stats.append((i, fx)) #add the y values to array for graph
    return x, np.array(stats)
HC_x, HC_stats= (hill_climb(init, nbr, f))
#iterate though for 5 runs, this is to prevent random init values affecting reults with bias
y_values_hc = []
for i in range(5):
    y = f(hill_climb(init,nbr,f)[0])
    
    print("For run ",i, " in HC the cost value is: ", y)
    
    y_values_hc.append(y)
#mean, standard deviation and penalty are calculated
print("Average Cost using HC: ", st.mean(y_values_hc))
print("Standard Deviation of Cost using HC: ", st.stdev(y_values_hc))

print("HC the Supply and Demand Penalty is: ", supply_demand_penalty(HC_x))
#plot the f(x) vs iterations to see how it improves over time
plt.plot(HC_stats[:, 0], HC_stats[:, 1])
plt.xlabel("Iteration"); plt.ylabel("Objective")

#simulated annealing function
def anneal(f, nbr, init, its=50000):
    # assume we are minimising
    stats =[]
    x = init() # initial random solution
    fx = f(x)
    T = 1 # initial temperature
    alpha = 0.99 # temperature decay per iteration
    for i in range(its):
        xnew = nbr(x) # generate a neighbour of x
        fxnew = f(xnew)
        if (fxnew < fx or random.random() < math.exp((fx - fxnew) / T)):# accept improvment or disimprove probability T
            x = xnew
            fx = fxnew
            T *= alpha
        stats.append((i,fx))
    return x, np.array(stats)

anneal_x, ann_stat = anneal(f, nbr, init)
# same method as above for 5 runs find average cost, standard dev and penalty
y_values_sa = []
for i in range(5):
    y = f(anneal(f,nbr, init)[0])
    print("For run ",i, " in SA the cost value is ", y)
    y_values_sa.append(y)
print("Average Cost using SA: ", st.mean(y_values_sa))
print("Standard Deviation of Cost using SA: ", st.stdev(y_values_sa))

print("SA the Supply and Demand Penalty is: ", supply_demand_penalty(anneal_x))
#graph the function vs iterations
plt.plot(ann_stat[:, 0], ann_stat[:, 1])
plt.xlabel("Iteration"); plt.ylabel("Objective")

#lahc function initalise L and iteraitons
def LAHC(f, init, nbr, L=50, its=50000):
    stats = []
    x = init() # initial solution
    fx = f(x) # cost of current solution
    best = x # best-ever solution
    Cbest = fx # cost of best-ever
    his = [fx] * L # initial history
    for i in range(its): # number of iterations
        xnew = nbr(x) # candidate solution
        fxnew = f(xnew) # cost of candidate
        if fxnew < Cbest: # minimising
            best = xnew # update best-ever
            Cbest = fxnew
        v = i % L # v indexes f circularly
        if fxnew <= his[v] or fxnew <= fx:
            x = xnew # accept candidateCs = Cs_ # (otherwise reject)
            fx=fxnew
        his[v] = fx # update circular history
        stats.append((i, fx)) 
    return best, np.array(stats)
    

#same as above plot the the results and run it 5 times comparing average, standard dev and penalty
Lahc_x, lahc_stats = LAHC(f, init, nbr)
plt.plot(lahc_stats[:, 0], lahc_stats[:, 1])
plt.xlabel("Iteration"); plt.ylabel("Objective")

y_values_lahc = []
for i in range(5):
    y = f(LAHC(f,init,nbr)[0])
    print("For run ",i, " in LHAC the cost value is ", y)
    
    y_values_lahc.append(y)
print("Average Cost using LAHC: ", st.mean(y_values_lahc))
print("Standard Deviation of Cost using LAHC: ", st.stdev(y_values_lahc))

print("LAHC the Supply and Demand Penalty is: ", supply_demand_penalty(Lahc_x))
