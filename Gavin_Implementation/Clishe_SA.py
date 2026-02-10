from Ptest_Tests.Ptest_10000 import data
#from Place_Benchmarks.Place_100 import data
import random
from copy import deepcopy
import numpy as np
import math
import time
from SA_funcs import *

#matplotlib is used for visualization/debugging. I do not expect to need it in the final result
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#print(data)

state = deepcopy(data) #without the deepcopy, each time data is modified in this file, the information in the memory location holding `data` is modified. Then if you rerun the "import data" line, it imports the still-modified value held in the cache, NOT the variable actually in the Place_5 file. Therefore, a copy must be made to ensure that the original data variable that we import is not modified.
#print(state)

def cool(T: int) -> float: 
    # defines the cooling schedule for the temperature T
    return 0.98*T

start_time = time.perf_counter()
print("Starting...")

MASTER_SEED = 708677375                          # use this line if you want to specify a specific seed
#MASTER_SEED = random.randint(1000,1000000000)

master = random.Random(MASTER_SEED) 

# random.Random(MASTER_SEED) constructs an RNG object whose output depends (deterministically) on MASTER_SEED.
# master.getrandbits(32) will be used to generate four random (but deterministic) seeds from master on each iteration. 
# successive getrandbits(32) calls results in different numbers being called each time. For example, 
# s1 = master.getrandbits(32) --> 1789368711
# s2 = master.getrandbits(32) --> 3146859322
# s3 = master.getrandbits(32) --> 43676229
# s4 = master.getrandbits(32) --> 3522623596
# These four seeds will be generated (and will be different) on each iteration. The perturb() function will use three of them and the 
# accept_move() function will use the other one. But since everything is derived from this random.Random object, the end result of the whole
# algorithm will be the the same as long as MASTER_SEED remains unchanged. 

T_min = 0.1
T = 40000

REHEAT_TEMPS = [20000, 10000]                               # adding reheats specified in this list
REHEAT_MOVES_PER_T_STEP = [150, 100]                        # successive cooling schedules will be faster
REHEAT_RANDOM_MOVE_CHANCE = [0.5, 0.7]                      # ... and more random
num_reheats_done = 0                                        # counts num of reheats

curr_moves_per_t_step = 250                                 # initializing current moves per t step and random move chance
curr_random_move_chance = 0.3  

curr_solution = annotate_net_lengths_and_weights(state)     # we start by adding length and weight fields to each net.
#plot_placement(curr_solution)
#print("First 10 nets of the current solution are: ", *state['nets'][:10], sep='\n')                # prints the first 10 nets. unpacks them and separates by new line for readability
current_cost = cost(curr_solution)
initial_cost = deepcopy(current_cost)

best_solution = deepcopy(curr_solution)
best_cost = deepcopy(current_cost)
lookups = build_fast_lookups(curr_solution)     # Only needs to be run once. lookups[pos_to_cell] will need to be updated every time a move is accepted, but the apply_proposed_move() function handles this.  

while True:
    while T > T_min:
        print(f"Current temperature: {T}.")
        for i in range(curr_moves_per_t_step):
            # setting deterministic seeds that differ on each iteration. See comments above. 
            s1 = master.getrandbits(32)
            s2 = master.getrandbits(32)
            s3 = master.getrandbits(32)
            s4 = master.getrandbits(32)
            s5 = master.getrandbits(32)
            s6 = master.getrandbits(32)

            # create a dictionary that contains information about the proposed move. Does not actually modify `state`. Propose_move uses three random numbers. 
            proposal_info = propose_move(state=curr_solution, seeds=[s1,s2,s3, s5, s6],random_move_chance=curr_random_move_chance, lookups=lookups)                                 

            # compute the cost of making this change. save the potential new cost to new_cost, the change in cost to delta_cost, and the potential net updates to net_updates. Still no change has actually been made
            new_cost, delta_cost, net_updates = compute_move_cost_update(state=curr_solution, proposal=proposal_info, current_cost=current_cost, lookups=lookups)

            if accept_move(d_cost=delta_cost, T=T, k=1, seed=s4):

                # if the move is accepted, we need to actually apply this move, which consists of modifying curr_solution using the net_updates that has already been computed. Uses proposal_info to learn about the move that is being made.
                curr_solution = apply_proposed_move(state=curr_solution, proposal=proposal_info, net_updates=net_updates, lookups=lookups)   
                current_cost = new_cost     # uses the already computed new_cost to update the current cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = curr_solution
        
        T = cool(T)
    
    if num_reheats_done < len(REHEAT_TEMPS):
        T = REHEAT_TEMPS[num_reheats_done]
        curr_moves_per_t_step = REHEAT_MOVES_PER_T_STEP[num_reheats_done]
        curr_random_move_chance = REHEAT_RANDOM_MOVE_CHANCE[num_reheats_done]
        curr_time = time.perf_counter()
        print(f"Cooling done. Reheating to {T}. {curr_time - start_time} seconds have elapsed.")
        print(f"Best solution so far has cost of {best_cost}.")
        num_reheats_done += 1
    else:
        break

print(f"Initial cost was {initial_cost}; best cost is {best_cost}")

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"End. Execution time: {execution_time} seconds")
print(f"Master seed is {MASTER_SEED}")
#plot_placement(best_solution)



#100 benchmark best solution so far: 777. Cooling schedule: 0.98T, k=1, T = 40000, Tmin = 0.1, moves per T = 2500, random move chance 100%, master seed = 617171369, "moved cooling schedule ... " commit version, "Execution time: 48.5148046000395 seconds"
#100 benchmark best solution so far: 762. Cooling schedule: 0.98T, k=1, T = 40000, Tmin = 0.1, moves per T = 2500, random move chance 30%, master seed = 163934370, "moved cooling schedule ... " commit version, "Execution time: 115 seconds"
#10,000 benchmark has best solution of 1,192,898. Cooling schedule is 0.98, T=40k, Tmin=0.1, moves per T = 250, random move chance 100%, master seed = 226433316, same commit version, execution time: 324 seconds.
#ptest_10000 has best soln of 380,269 compared to initial 1,247,867. same params as above. master seed is 711008386. took 126 seconds
#ptest_25000 has best soln of 1,846,142 compared to to initial cost of 4,921,479. Same parameters. 323 seconds. Seed of 505162987
#ptest_10000 has new best soln of 200,285 when random move is 50%. 400 seconds. seed of 708677375
#ptest_10000 has new best soln of 175,570 with 30% random move, same params above. 347 seconds. Seed of 648228016
#ptest_10000 has new best soln of 111,891 with 10% random move and 500 moves per step. Otherwise everything same. Seed of 220475152. Took 1306 seconds (21 mins)
