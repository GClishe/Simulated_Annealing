from Gavin_Implementation.Place_Benchmarks.Place_5 import data

##################################### Example placement data ############################
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

def cool(T: int) -> int: 
    # defines the cooling schedule for the temperature T
    return 0.95*T


def cost(data: dict) -> int:
    """
    Return cost associated with a particular state. Cost is the total manhattan distance 
    separating connected cells (those on the same net). Distance between connected cells
    is |x_i - x_j| + |y_i-y_j|. cost(state) returns sum of all distances.  
    
    :param state: Placement data describing grid size, cell locations, and net connectivity.
    :type state: dict
    :return: Total Manhattan wirelength summed over all nets.
    :rtype: int
    """
    # a potential optimization once the rest of the algorithm is written might be to instead only recompute the length of nets connected to a moved cell. 

    total_length = 0                                     # initializes length (cost) to 0
    for net in data['nets']:
        cell_i, cell_j = net['cells']                    # grabs the two connected cells on each net

        x_i, y_i = data['cells'][cell_i]['position']     # grabs the coordinates associated with cell_i in the data['cells'] dictionary
        x_j, y_j = data['cells'][cell_j]['position']     # same but for cell_j

        wire_length = abs(x_i - x_j) + abs(y_i - y_j)    # computes manhattan length between cell_i and cell_j
        total_length += wire_length                      # adds length of this net to the total length
    
    return total_length

def perturb(curr_solution: dict) -> dict:
    # function that takes the current solution and makes a single move (however that is defined), returning the resulting solution. 
    # my current approach (might be quite intensive though) is to find the net with the highest cost (assuming at least one of the cells
    # is movable), then move one of the cells to the nearest available cell (movable or not) to the other one. I do not think this move function
    # should be deterministic. Because if it is, then you run the risk of every single iteration in num_moves_per_step attempting the exact same
    # move. At lower temperatures, this move might be rejected every single time, which is a waste. So maybe instead of choosing THE highest cost net,
    # it instead proababalistically chooses nets, with higher weights applied to nets of larger cost. Perhaps a first approach might assign weights as 
    # net_cost/total_cost. It may not matter, but this approach does normalize the weights such that they sum to 1. This is what i did in my undergrad research

    

    



