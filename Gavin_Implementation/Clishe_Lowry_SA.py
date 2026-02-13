#This script optimizes standard cell placement using a form of simulated annealing
#To operate, make sure you place the sample data in the appropriate folders with the appropriate file names
#This would be ./Ptest_Tests/Ptest_XXXXX and ./Place_Benchmarks/Place_XXXXX
#Result folders should also be prepared ./Clishe_Lowry_Ptest_Tests and ./Clishe_Lowry_Place_Benchmarks
#Make sure the file extention .py is added to all data set files
#Once ready, simply run this file by opening a terminal window in this directory and running: python3 Clishe_Lowry_SA.py

#At the top of this file are tunable parameters. The first is the file name of the desired test (without the .py)
#Make sure it is placed in the correct folder, and the result will appear in the corresponding separate folder as outlined above
#In some cases we abstracted away from the standard SA parameters in favor of desired probabilities and number of temperature steps
#This allows the algorithm to scale with the problem size to some degree, as well as better control the execution time
#This algorithm also has an advanced preturb function where good proposed moves are calculated by moving cells in a net closer together
#You can tune the ratio of random vs caculated moves below, set it to 1 for completly random operation

#Tunable parameters
dataName = 'Ptest_500'    #Name of netlist file. Need to add .py to the end of provided files. Make sure original folder names are used and that result folders exist
T = 0                       #Initial temp determines probability of accepting a bad solution at the start. Leave at 0 to use calculated value based on grid size
probEnd = 0.0000454         #Probability of accepting any bad move at the end
tempCount = 25000           #Number of temperature steps to cycle through between initial and final temp during the geometric cooling cycle
MOVES_PER_T_STEP = 250      #Number of moves to attempt at each temperature step
K_BOLTZ = 1                 #Constant to change how the acceptance rate of bad moves is calculated. Leave this at 1 and change temperatures
curr_random_move_chance = 1 #Percent of proposed moves that are randomly generated. Non random moves pick a random net and tries to move one of its cells as close as possible to the other
MASTER_SEED = 708677375     #Set seed to make RND reproducable. Comment this line and uncomment the line right after the import block use a random seed

#Set import / export folder names based on data set
if dataName[:2] == 'Pt':        
    folderName = 'Ptest_Tests'
else:
    folderName = 'Place_Benchmarks'

#Import net list data and functions
data = getattr(__import__(f'{folderName}.{dataName}', fromlist = ['data']), 'data')
import random
from copy import deepcopy
import numpy as np
import time
from SA_funcs import *
import pprint

#MASTER_SEED = random.randint(1000,1000000000)          # use this line to generate a random seed

state = deepcopy(data) #without the deepcopy, each time data is modified in this file, the information in the memory location holding `data` is modified. Then if you rerun the "import data" line, it imports the still-modified value held in the cache, NOT the variable actually in the Place_5 file. Therefore, a copy must be made to ensure that the original data variable that we import is not modified.

start_time = time.perf_counter()
print(f"\nStarting placement for {dataName} using master seed {MASTER_SEED}...")

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

#Compute intial temp, final temp, and cooling rate based on desired number of temp steps, final bad move acceptance rate, and grid size
if T == 0:                                      #If T is 0 use calculated value, if not use given value
    T = 5.9729*(data['grid_size'])**1.1874      #Calculate optimal starting temp for grid size
T_min = -1/np.log(probEnd)
coolRate = (T_min/T)**(1/tempCount)

curr_solution = annotate_net_lengths_and_weights(state)     # we start by adding length and weight fields to each net.
#plot_placement(curr_solution)
#print("First 10 nets of the current solution are: ", *state['nets'][:10], sep='\n')                # prints the first 10 nets. unpacks them and separates by new line for readability
current_cost = cost(curr_solution)
initial_cost = deepcopy(current_cost)
print(f'\nInitial cost is {initial_cost}...')

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

    T = cool(coolRate, T)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"\nFound best cost of {best_cost} in {execution_time:.3f} sec...\n")

#plot_placement(best_solution)

best_solution = strip_net_length_weight(best_solution)

verify_solution_integrity(original_state=state, best_solution=best_solution)            # returns None if all the checks pass. raises valueError if any fail. 

print(f"\nExporting reults to ./Clishe_Lowry_{folderName}/{dataName}.py ...")
output_content = "data = " + pprint.pformat(best_solution)                              # Format output data
with open(f'Clishe_Lowry_{folderName}/{dataName}.py', 'w') as f:                        # Open export location
    f.write(output_content)                                                             # Export output data

################################################# PLOTS ###################################################
print("\nGraphing...")
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