from Gavin_Implementation.Place_Benchmarks.Place_5 import data
import random
import numpy as np

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

def perturb(data: dict) -> dict:
    # function that takes the current solution and makes a single move (however that is defined), returning the resulting solution. 
    # my current approach (might be quite intensive though) is to find the net with the highest cost (assuming at least one of the cells
    # is movable), then move one of the cells to the nearest available cell (movable or not) to the other one. I do not think this move function
    # should be deterministic. Because if it is, then you run the risk of every single iteration in num_moves_per_step attempting the exact same
    # move. At lower temperatures, this move might be rejected every single time, which is a waste. So maybe instead of choosing THE highest cost net,
    # it instead probablistically chooses nets, with higher weights applied to nets of larger cost. Perhaps a first approach might assign weights as 
    # net_cost/total_cost (or something else)

    
    # the code below reuses the same code in the cost function. Once I have an implementation for each part of the SA algorithm, I will rewrite to remove repeated code.
    nets = []
    weights=[]
    max_length = 2*(data['grid_size'] + 1)               # nets will be weighted based on their length compared to the max length. There is no need for the weights to be normalized, so we do not need to weight them based on their length compared to total cost. 
    for net in data['nets']:                   
        cell_i, cell_j = net['cells']                    # grabs the two connected cells on each net

        x_i, y_i = data['cells'][cell_i]['position']     # grabs the coordinates associated with cell_i in the data['cells'] dictionary
        x_j, y_j = data['cells'][cell_j]['position']     # same but for cell_j

        wire_length = abs(x_i - x_j) + abs(y_i - y_j)

        # adds length and weight as parameters attached to each net
        net['length'] = wire_length
        net['weight'] = wire_length/max_length 

        if any(data['cells'][cell_name]['type'] == 'MOVABLE' for cell_name in net['cells']): # checks if either of the cells on this net are movable
            # if either of the cells are movable, add the net to the `nets` list (and its weight to the `weights` list)
            weights.append(net['weight'])         
            nets.append(net)
    
    # choose a net probabilistically based on its weight relative to the other nets. 
    # because `nets` and `weights` contains only nets with movable cells, chosen_net is also guaranteed
    # to have at least one movable cell
    chosen_net = random.choices(nets, weights=weights) 

    # Now we need to randomly choose either one of the movable cells on that net. If only one cell is movable, we must choose that one. We do this with a mask of movable cells        
    movable_cell_mask = [data['cells'][cell_name]['type'] == 'MOVABLE' for cell_name in chosen_net['cells']] # creates a [True/False, True/False] mask describing cells that are movable or not
    movable_cells = np.array(chosen_net['cells'])[movable_cell_mask] # uses the mask to create an array that contains the movable cell(s) on that net. len(movable_cells) is either 1 or 2.
    random.choices(movable_cells)[0]                                         # randomly chooses one of the cells in movable_cells. choices(movable_cells) has type list, so we choose index 0 to extract the string.
    

chosen_net = data['nets'][1]

# Now we need to randomly choose either one of the movable cells on that net. If only one cell is movable, we must choose that one. We do this with a mask of movable cells        
movable_cell_mask = [data['cells'][cell_name]['type'] == 'MOVABLE' for cell_name in chosen_net['cells']] # creates a [True/False, True/False] mask describing cells that are movable or not
movable_cells = np.array(chosen_net['cells'])[movable_cell_mask] # uses the mask to create an array that contains the movable cell(s) on that net. len(movable_cells) is either 1 or 2.
cell_to_move = random.choices(movable_cells)[0]                         # randomly selects one of the cells in movable_cells. choices(movable_cells) has type list, so we choose index 0 to extract the string.   

#Now I want to move that cell to be close to its net-neighbor (target_cell). 
for cell in chosen_net['cells']:
    if cell != cell_to_move:
        target_cell = cell          # the cell on chosen_net that will not be moved is the target cell.


def search_ring(data: dict, target_coordinates: tuple[int, int], grid_size: int) -> tuple[int, int]:
    """
    Searches outward from a target coordinate in Manhattan distance "rings" and returns
    a randomly selected available (not locked, in-bounds) from the nearest ring. 

    First searches ring of Manhattan distance of 1 from the target, and proceeds searching
    until the ring size equals twice the grid_size parameter. If multiple cells exist at the
    same minimum distance, one is chosen randomly. 
    
    :param data: Placement data dictionary.
    :type data: dict
    :param target_coordinates: (x, y) coordinates of the target cell.
    :type target_coordinates: tuple[int, int]
    :param grid_size: Size of the (square) grid. Valid coordinates satisfy
                    0 <= x < grid_size and 0 <= y < grid_size.
    :type grid_size: int
    :return: Coordinates (x, y) of a nearest available cell.
    :rtype: tuple[int, int]
    :raises ValueError: If no available cell exists within the grid.
    """
    
    X, Y = target_coordinates                                                       # unpacks the target_coordinates tuple to X and Y variables

    locked_coordinates = {
        cell["position"] for cell in data["cells"].values() if cell["fixed"]        # creates a set containing all locked coordinates. Set is chosen because it is easy to search and order is not needed
    }

    def in_bounds(coord: tuple[int, int], grid_size: int) -> bool:
        # helper function that decides if a coordinate is is in the grid. 
        return (0 <= coord[0] < grid_size) and (0 <= coord[1] < grid_size)

    for ring_size in range(1, 2 * grid_size + 1):                                   # ring_size (manhattan distance to target) can be at most 2*grid_size
        candidate_moves = set()                                                     # the goal is to randomly select one cell that is ring_size away from the target. Set is chosen over list to remove duplicates

        for i in range(ring_size + 1):
            j = ring_size - i                                                       # with this setup, we will iterate through all i,j pairs such that i + j = ring_size. This will allow us to generate all coordinates on the ring
            coords = [(X+i, Y+j), (X+i, Y-j), (X-i, Y+j), (X-i, Y-j)]               # this actually creates the coordinates. However, it does allow duplicates (for example, (X+i, Y+j) and (X+i, Y-j) are the same when j==0). This is why candidate_moves is a set.

            for coord in coords:                                                    # checking each of the four coordinates generated by an i,j pair
                if in_bounds(coord, grid_size) and coord not in locked_coordinates:
                    candidate_moves.add(coord)                                      # if it is both in the boundary and not locked, then we add it to the candidate moves. 

        if candidate_moves:  
            return random.choice(tuple(candidate_moves))                            # set objects dont work with random.choice, so instead we convert to a tuple first. It makes no difference if we convert to list or tuple, so i just chose tuple.

    raise ValueError("There are no available cells. Either all cells are locked or grid_size is invalid.")
