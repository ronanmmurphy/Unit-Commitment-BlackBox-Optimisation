# Unit-Commitment-BlackBox-Optimisation
Implemented three black box optimisation algorithms: Hill-Climb, Simulated Annealing, and Late Acceptance Hill Climb to solve unit commitment problem. The CSV files contain the data which the problem defines.

Problem:
In a small country there are 10 generators of four types: Hydroelectric, Solid fuel, Gas, Solar. Each
generator can create an integer number of MW (megawatts), up to a maximum capacity. Each generator has
its own maximum capacity. Each generator also has a cost for producing each MW. The generators, types,
capacities, and costs are given in plant_info.csv. The maximum supply from any Solar plant depends on
the time of day. Relative to the plant’s maximum, it can achieve 50% from 6am to 10am, 100% from 11am to
3pm, 50% from 4pm to 6pm, and 0% otherwise. This is given in solar_curve.csv.

The demand per hour of the day is given by demand.csv.

The objective function equals cost of production for one day plus supply and demand penalties for that day
plus output change penalties.
Supply and demand penalties: Our aim is to match demand closely. Even over-supply is bad, because it can
damage our infrastructure. So we add a penalty for failure to match demand, as follows:
• For every hour in which the total supply is greater than the demand, the penalty is (supply - demand)
squared, where supply and demand are measured in MW.
• For every hour in which the total supply is less than the demand, the penalty is (demand - supply)
cubed.
Output change penalties: Solid fuel plants can change their output very slowly – by at most 1 MW per hour.
Thus for each solid-fuel plant, there is a penalty associated with changing its output from hour to hour. This
penalty is equal to the absolute change. Eg if the output for a plant is 500 in one hour and 400 in the next,
then the penalty for this is 100.

Comparison of the Three Algorithms:



![objective function comparison](https://github.com/ronanmmurphy/Unit-Commitment-BlackBox-Optimisation/blob/main/Images/Objectivefunction.png?raw=true)

The green line is LAHC, the blue line HC is and the orange line is SA. We see that although LAHC is the
slowest to reach the optimum therefore the cost is high it produces the best results. Similarly the SA
produces almost as good of results but gets to solution with much less iterations. Finally Hill climb has
a step descent and is less accurate but this is because its initial values are more costly and must
change a lot to get to global minimum.

The fitness landscape of the problem is smooth, unimodal, convex and informative and all the
optimising algorithms are effective in finding good solutions. To select the best one it would have to
consider both the mean, average and penalties as we don’t want under/oversupplying to occur. LAHC
has the best accuracy at reaching a solution but performs badly with penalties. HC is the opposite
where it performs worse but is the least wasteful. I would choose the balance of the three the
simulated annealing algorithm as it performs well and has the least penalties, although a lot more
testing is needed to find the best hyperparameters.
