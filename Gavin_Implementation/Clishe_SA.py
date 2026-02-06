from Gavin_Implementation.Place_Benchmarks.Place_5 import data
import random
import numpy as np
print(data)
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


def search_ring(data: dict, target_coordinates: tuple[int, int], grid_size: int, seed: int = None) -> tuple[int, int]:
    """
    Searches outward from a target coordinate in Manhattan distance "rings" and returns
    a randomly selected available (not locked, in-bounds) from the nearest ring. 

    First searches ring of Manhattan distance of 1 from the target, and proceeds searching
    until the ring size equals twice the grid_size parameter. If multiple cells exist at the
    same minimum distance, one is chosen randomly. Seed parameter ensures reproducability. 
    
    :param data: Placement data dictionary.
    :type data: dict
    :param target_coordinates: (x, y) coordinates of the target cell.
    :type target_coordinates: tuple[int, int]
    :param grid_size: Size of the (square) grid. Valid coordinates satisfy
                    0 <= x < grid_size and 0 <= y < grid_size.
    :type grid_size: int
    :param seed: Seed for random number generator.
    :type: int
    :return: Coordinates (x, y) of a nearest available cell.
    :rtype: tuple[int, int]
    :raises ValueError: If no available cell exists within the grid.
    """
    
    rng = random.Random(seed)                                                       # creates an object that generates random sequence of numbers according to a seed. 

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
            return rng.choice(tuple(candidate_moves))                               # choose a random value in candidate_moves (converted to tuple first, as .choice doesnt work with sets) according to the seed. 

    raise ValueError("There are no available cells. Either all cells are locked or grid_size is invalid.")


def perturb(data: dict, seeds: list[int, int, int] = [None, None, None]) -> dict:
    """
    Proposes a signle perturbation (move) to the current placement solution. The move is generated in three randomized steps:
    First, you choose a net that contains at least one moveable cell, with probability proportional to the net's current
    wire length. Next, choose one movable cell on that net to move. Third, move that cell to a nearby coordinate close to the 
    other cell on that net, using the search_ring() function. If the desination coordinate is occupied, swap the two cells. 

    To ensure reproducibility despite randomness, we seed each random number generator with a different seed. seeds[0] seeds
    the net selection, seed[1] seeds the movable-cell selection, and seeds[2] is forwarded to search_ring() for destination 
    selection.
    
    :type data: dict
    :param seeds: Three seeds controlling the random choices made by this function.
    :type seeds: list[int, int, int]
    :return: The modified placement dictionary (cells may be moved / swapped).
    :rtype: dict
    :raises ValueError: If no nets contain a MOVABLE cell (cannot generate a move).
    """
    
    # the code below reuses the same code in the cost function. Once I have an implementation for each part of the SA algorithm, I will rewrite to remove repeated code.
    nets = []
    weights = []
    max_length = 2 * (data['grid_size'] + 1)               # nets will be weighted based on their length compared to the max length. There is no need for the weights to be normalized.

    for net in data['nets']:
        cell_i, cell_j = net['cells']                      # grabs the two connected cells on each net

        x_i, y_i = data['cells'][cell_i]['position']       # grabs the coordinates associated with cell_i in the data['cells'] dictionary
        x_j, y_j = data['cells'][cell_j]['position']       # same but for cell_j

        wire_length = abs(x_i - x_j) + abs(y_i - y_j)       # manhattan distance

        # adds length and weight as parameters attached to each net
        net['length'] = wire_length
        net['weight'] = wire_length / max_length

        # if either of the cells are movable, add the net to the `nets` list (and its weight to the `weights` list)
        if any(data['cells'][cell_name]['type'] == 'MOVABLE' for cell_name in net['cells']): 
            weights.append(net['weight'])
            nets.append(net)

    # check if nets is empty. If so, raise a ValueError
    if not nets:   
        raise ValueError("No nets contain a MOVABLE cell, so no perturbation can be generated.")

    # choose a net probabilistically based on its weight relative to the other nets.
    # because `nets` and `weights` contains only nets with movable cells, chosen_net is also guaranteed
    # to have at least one movable cell
    rng_net = random.Random(seeds[0])
    chosen_net = rng_net.choices(nets, weights=weights, k=1)[0]

    # now we need to randomly choose either one of the movable cells on that net. If only one cell is movable, we must choose that one.
    movable_cell_mask = [data['cells'][cell_name]['type'] == 'MOVABLE' for cell_name in chosen_net['cells']]  # creates a [True/False, True/False] mask describing cells that are movable or not
    movable_cells = np.array(chosen_net['cells'])[movable_cell_mask]                                          # uses the mask to create an array that contains the movable cell(s) on that net. len(movable_cells) is either 1 or 2.

    rng_cell = random.Random(seeds[1])
    cell_to_move = rng_cell.choices(list(movable_cells), k=1)[0]                                              # randomly chooses one of the cells in movable_cells. choices(...) returns a list, so [0] extracts the string.
    cell_to_move_original_coords = data['cells'][cell_to_move]['position']

    # Now I want to move that cell to be close to its net-neighbor (target_cell).
    for cell in chosen_net['cells']:
        if cell != cell_to_move:
            target_cell = cell                    # the cell on chosen_net that will not be moved is the target cell.

    new_coords = search_ring(
        data=data,
        target_coordinates=data['cells'][target_cell]['position'],
        grid_size=data['grid_size'],
        seed=seeds[2]
    )  # finds the coordinates of the new position of cell_to_move

    # Now we need to actually move this cell to the new coords. First step will be to check if the new coords are occupied. If so, swap them.
    for cell in data['cells'].values():
        if new_coords == cell['position']:
            cell['position'] = cell_to_move_original_coords     # moves the cell occupying the new coordinates to the old position of the cell we are moving

    data['cells'][cell_to_move]['position'] = new_coords        # updates the cell we are moving to be in the new position.

    return data

