from Gavin_Implementation.Place_Benchmarks.Place_100 import data
import random
from copy import deepcopy
import numpy as np
import math
import time
from Gavin_Implementation.SA_funcs import *

#matplotlib is used for visualization/debugging. I do not expect to need it in the final result
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#print(data)

state = deepcopy(data) #without the deepcopy, each time data is modified in this file, the information in the memory location holding `data` is modified. Then if you rerun the "import data" line, it imports the still-modified value held in the cache, NOT the variable actually in the Place_5 file. Therefore, a copy must be made to ensure that the original data variable that we import is not modified.
#print(state)

##################################### Example placement state ############################
# data = {
#     'grid_size': 3,
# 
#     'cells': {
#         'CELL_0': {'type': 'MOVABLE', 'fixed': False, 'position': (1, 2)},
#         'CELL_1': {'type': 'MOVABLE', 'fixed': False, 'position': (1, 0)},
#         'CELL_2': {'type': 'MOVABLE', 'fixed': False, 'position': (0, 0)},
#         'CELL_3': {'type': 'MOVABLE', 'fixed': False, 'position': (0, 1)},
#         'IO_0': {'type': 'IO', 'fixed': True, 'position': (2, 1)},
#     },
# 
#     'nets': [
#         {'name': 'NET_0', 'cells': ('IO_0', 'CELL_2')},
#         {'name': 'NET_1', 'cells': ('CELL_0', 'CELL_1')},
#         {'name': 'NET_10', 'cells': ('CELL_1', 'IO_0')},
#         {'name': 'NET_11', 'cells': ('CELL_1', 'CELL_2')},
#         {'name': 'NET_12', 'cells': ('CELL_3', 'CELL_2')},
#         {'name': 'NET_2', 'cells': ('CELL_0', 'CELL_1')},
#         {'name': 'NET_3', 'cells': ('CELL_2', 'CELL_3')},
#         {'name': 'NET_4', 'cells': ('CELL_3', 'CELL_2')},
#         {'name': 'NET_5', 'cells': ('CELL_2', 'CELL_0')},
#         {'name': 'NET_6', 'cells': ('CELL_0', 'CELL_3')},
#         {'name': 'NET_7', 'cells': ('CELL_3', 'CELL_2')},
#         {'name': 'NET_8', 'cells': ('IO_0', 'CELL_3')},
#         {'name': 'NET_9', 'cells': ('CELL_0', 'IO_0')},
#     ]
# }
###########################################################################################
def cool(T: int) -> float: 
    # defines the cooling schedule for the temperature T
    return 0.95*T

start_time = time.perf_counter()
print("Starting...")

MASTER_SEED = random.randint(1000,1000000000)
print(MASTER_SEED)
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
NUM_MOVES_PER_T_STEP = 250
T = 40000

curr_solution = annotate_net_lengths_and_weights(state)     # we start by adding length and weight fields to each net.
#plot_placement(curr_solution)
#print("First 10 nets of the current solution are: ", *state['nets'][:10], sep='\n')                # prints the first 10 nets. unpacks them and separates by new line for readability
current_cost = cost(curr_solution)

best_solution = deepcopy(curr_solution)
best_cost = deepcopy(current_cost)
lookups = build_fast_lookups(curr_solution)     # Only needs to be run once. lookups[pos_to_cell] will need to be updated every time a move is accepted, but the apply_proposed_move() function handles this.  

while T > T_min:
    for i in range(NUM_MOVES_PER_T_STEP):
        # setting deterministic seeds that differ on each iteration. See comments above. 
        s1 = master.getrandbits(32)
        s2 = master.getrandbits(32)
        s3 = master.getrandbits(32)
        s4 = master.getrandbits(32)
        s5 = master.getrandbits(32)
        s6 = master.getrandbits(32)

        # create a dictionary that contains information about the proposed move. Does not actually modify `state`. Propose_move uses three random numbers. 
        proposal_info = propose_move(state=curr_solution, seeds=[s1,s2,s3, s5, s6],random_move_chance=1, lookups=lookups)                                 

        # compute the cost of making this change. save the potential new cost to new_cost, the change in cost to delta_cost, and the potential net updates to net_updates. Still no change has actually been made
        new_cost, delta_cost, net_updates = compute_move_cost_update(state=curr_solution, proposal=proposal_info, current_cost=current_cost)

        if accept_move(d_cost=delta_cost, T=T, k=1, seed=s4):

            # if the move is accepted, we need to actually apply this move, which consists of modifying curr_solution using the net_updates that has already been computed. Uses proposal_info to learn about the move that is being made.
            curr_solution = apply_proposed_move(state=curr_solution, proposal=proposal_info, net_updates=net_updates, lookups=lookups)   
            current_cost = new_cost     # uses the already computed new_cost to update the current cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = curr_solution
    
    T = cool(T)


print(best_cost)


end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"End. Execution time: {execution_time} seconds")
#plot_placement(best_solution)