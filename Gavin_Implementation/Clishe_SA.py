from Ptest_Tests.Ptest_50 import data
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
    return 0.99*T

start_time = time.perf_counter()
print("Starting...")

#MASTER_SEED = 708677375                          # use this line if you want to specify a specific seed
MASTER_SEED = random.randint(1000,1000000000)

master = random.Random(MASTER_SEED) 

# Creates random number generators (not random numbers) seeded by master.getrandbits(32) which is different each time it is called. As long as MASTER_SEED is the same, these RNG objects will also be the same per iteration (but each are different from each other)
rngs = {
    "net": random.Random(master.getrandbits(32)),       # used in propose_move()
    "cell": random.Random(master.getrandbits(32)),      # used in propose_move()
    "ring": random.Random(master.getrandbits(32)),      # used in search_ring() inside propose_move()
    "branch": random.Random(master.getrandbits(32)),    # used in propose_move()
    "rand": random.Random(master.getrandbits(32)),      # used in propose_move()
    "accept": random.Random(master.getrandbits(32)),    # used in accept_move()
} 

T_min = 0.1
T = 40000
MOVES_PER_T_STEP = 5000
curr_random_move_chance = 0.1
K_BOLTZ = 1

curr_solution = annotate_net_lengths_and_weights(state)     # we start by adding length and weight fields to each net.
#plot_placement(curr_solution)
#print("First 10 nets of the current solution are: ", *state['nets'][:10], sep='\n')                # prints the first 10 nets. unpacks them and separates by new line for readability
current_cost = cost(curr_solution)
initial_cost = deepcopy(current_cost)

best_solution = deepcopy(curr_solution)
best_cost = deepcopy(current_cost)
lookups = build_fast_lookups(curr_solution)     # Only needs to be run once. lookups[pos_to_cell] will need to be updated every time a move is accepted, but the apply_proposed_move() function handles this.  

# these quantities are used for plotting the SA graphs
step_idx = 0
anneal_steps = []
temps = []
mean_boltz = []
accepted_moves = []
end_costs = []

while T > T_min:
    step_idx += 1
    accept_count = 0
    boltz_samples = []      # when we plot boltzmann exponential vs temp step, we take the average of all computed boltzmann exponentials at that temp step
    #print(f"Current temperature: {T}.")

    for i in range(MOVES_PER_T_STEP):

        # create a dictionary that contains information about the proposed move. Does not actually modify `state`. Propose_move uses three random numbers. 
        proposal_info = propose_move(
            state=curr_solution, 
            rngs=rngs, 
            random_move_chance=curr_random_move_chance, 
            lookups=lookups)                                 

        # compute the cost of making this change. save the potential new cost to new_cost, the change in cost to delta_cost, and the potential net updates to net_updates. Still no change has actually been made
        new_cost, delta_cost, net_updates = compute_move_cost_update(
            state=curr_solution, 
            proposal=proposal_info, 
            current_cost=current_cost, 
            lookups=lookups
            )
        
        if delta_cost > 0:
            boltz_samples.append(math.exp(-delta_cost / (K_BOLTZ * T))) # for plotting. we only compute the boltzmann exponential when the delta cost is positive

        if accept_move(d_cost=delta_cost, T=T, k=K_BOLTZ, rng = rngs['accept']):
            accept_count += 1
            # if the move is accepted, we need to actually apply this move, which consists of modifying curr_solution using the net_updates that has already been computed. Uses proposal_info to learn about the move that is being made.
            curr_solution = apply_proposed_move(
                state=curr_solution, 
                proposal=proposal_info, 
                net_updates=net_updates, 
                lookups=lookups
                )   
            
            current_cost = new_cost     # uses the already computed new_cost to update the current cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = curr_solution
    
    # below lines are for plotting
    anneal_steps.append(step_idx)
    temps.append(T)
    mean_boltz.append(float(np.mean(boltz_samples)) if boltz_samples else 1.0)      # appending the mean of the boltzmann exponentials at this temp step. If no boltz samples occur, append 1. 
    accepted_moves.append(accept_count)
    end_costs.append(current_cost)

    T = cool(T)

print(f"Initial cost was {initial_cost}; best cost is {best_cost}")

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"End. Execution time: {execution_time} seconds")
print(f"Master seed is {MASTER_SEED}")
#plot_placement(best_solution)

best_solution = strip_net_length_weight(best_solution)

verify_solution_integrity(original_state=state, best_solution=best_solution)            # returns None if all the checks pass. raises valueError if any fail. 

################################################# PLOTS ###################################################
fig, axes = plt.subplots(4, 1, figsize=(10, 10))

# temperature schedule
axes[0].set_title("Temperature vs Annealing Step")
axes[0].plot(anneal_steps, temps)
axes[0].set_ylabel("Temperature")
axes[0].set_xlabel("Annealing Step")
axes[0].grid(True)

# average boltzmann exponential per temp step
axes[1].set_title("Average Boltzmann Exponential vs Annealing Step")
axes[1].plot(anneal_steps, mean_boltz)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("Boltzmann Exponential")
axes[1].set_xlabel("Annealing Step")
axes[1].grid(True)

# accepted moves per temp step
axes[2].set_title("Number of Accepted Moves per Annealing Step")
axes[2].plot(anneal_steps, accepted_moves)
axes[2].set_ylabel("Num Moves Acc")
axes[2].set_xlabel("Annealing Step")
axes[2].grid(True)

# cost per temp step with moving average

axes[3].set_title("Final Cost per Annealing Step")
axes[3].plot(anneal_steps, end_costs, label="Final cost")

window = 10

moving_avg = np.convolve(end_costs, np.ones(window) / window, mode="valid")
ma_steps = anneal_steps[window - 1:]  # align x-axis with valid convolution
axes[3].plot(ma_steps, moving_avg, linestyle="--", linewidth=2,
                label=f"Moving avg (window size={window} iterations)", color='r')
    
axes[3].set_ylabel("Cost Function")
axes[3].set_xlabel("Annealing Step")
axes[3].grid(True)
axes[3].legend()

plt.tight_layout()
plt.show()
