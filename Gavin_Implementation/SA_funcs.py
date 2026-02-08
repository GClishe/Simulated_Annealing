import random
from copy import deepcopy
import numpy as np
import math
import time

#matplotlib is used for visualization/debugging. I do not expect to need it in the final result
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def cool(T: int) -> float: 
    # defines the cooling schedule for the temperature T
    return 0.95*T

#def cost(state: dict) -> int:
#    """
#    Return cost associated with a particular state. Cost is the total manhattan distance 
#    separating connected cells (those on the same net). Distance between connected cells
#    is |x_i - x_j| + |y_i-y_j|. cost(state) returns sum of all distances.  
#    
#    :param state: Placement state describing grid size, cell locations, and net connectivity.
#    :type state: dict
#    :return: Total Manhattan wirelength summed over all nets.
#    :rtype: int
#    """
#    # a potential optimization once the rest of the algorithm is written might be to instead only recompute the length of nets connected to a moved cell. 
#
#    total_length = 0                                     # initializes length (cost) to 0
#    for net in state['nets']:
#        cell_i, cell_j = net['cells']                    # grabs the two connected cells on each net
#
#        x_i, y_i = state['cells'][cell_i]['position']     # grabs the coordinates associated with cell_i in the state['cells'] dictionary
#        x_j, y_j = state['cells'][cell_j]['position']     # same but for cell_j
#
#        wire_length = abs(x_i - x_j) + abs(y_i - y_j)    # computes manhattan length between cell_i and cell_j
#        total_length += wire_length                      # adds length of this net to the total length
#    
#    return total_length

def cost(state: dict) -> int:
    """
    Computes the total cost as the sum of all individual net lengths.
    Requires that state['nets'] dicts have fields 'length'. Thus, run 
    annotate_net_lengths_and_weights() before calling this function. 
    """

    total_cost = 0
    for net in state['nets']:
        total_cost += net['length']
    
    return total_cost

def annotate_net_lengths_and_weights(state: dict) -> dict:
    """
    Modifies the state dictionary, adding length and weight parameters to each net. 
    Computing the total cost later on will be easier, as it will only need to add
    up all of the wire_length values. Additionally, when a move is accepted, only 
    affected nets need be updated rather than re-computing total cost. 
    """
    grid_size = state["grid_size"]

    max_length = 2 * (grid_size - 1)        # used to weight the random number generator responsible for choosing nets in perturb().

    for net in state["nets"]:
        cell_i, cell_j = net["cells"]

        x_i, y_i = state["cells"][cell_i]["position"]
        x_j, y_j = state["cells"][cell_j]["position"]

        wire_length = abs(x_i - x_j) + abs(y_i - y_j)

        net["length"] = wire_length
        net["weight"] = wire_length / max_length

    return state

# search_ring is essentially a helper function designed for use in the perturb() function. 
def search_ring(state: dict, target_coordinates: tuple[int, int], grid_size: int, seed: int = None) -> tuple[int, int]:
    """
    Searches outward from a target coordinate in Manhattan distance "rings" and returns
    a randomly selected available (not locked, in-bounds) from the nearest ring. 

    First searches ring of Manhattan distance of 1 from the target, and proceeds searching
    until the ring size equals twice the grid_size parameter. If multiple cells exist at the
    same minimum distance, one is chosen randomly. Seed parameter ensures reproducability. 
    
    :param state: Placement state dictionary.
    :type state: dict
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
        cell["position"] for cell in state["cells"].values() if cell["fixed"]        # creates a set containing all locked coordinates. Set is chosen because it is easy to search and order is not needed
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

def build_fast_lookups(state: dict) -> dict:
    """
    Function that builds immutable tables that are useful for searching in propose_move(). 
    Calling this function once at the start means we wont have to build these lists at each 
    iteration. 
    """
    
    cell_names = list(state["cells"].keys())                        # list containing all the cell names

    g = state["grid_size"]
    all_coords = [(x, y) for x in range(g) for y in range(g)]       # list containing all coordinate pairs on the grid

    fixed_positions = {cell["position"] for cell in state["cells"].values() if cell["fixed"]}   # set containing all the coordinates of fixed cells on the grid

    pos_to_cell = {cell["position"]: name for name, cell in state["cells"].items()}             # dict containing coordinate: cell name pairs 

    return {
        "cell_names": cell_names,
        "all_coords": all_coords,
        "fixed_positions": fixed_positions,
        "pos_to_cell": pos_to_cell
    }

def propose_move(state: dict,
                 seeds: list[int, int, int, int, int] = [None, None, None, None, None],
                 random_move_chance: float = 0.2,
                 lookups: dict | None = None
                    ) -> dict:
    """
    Proposes a single move for simulated annealing. Assumes that annotate_net_lengths_and_weights() has 
    already been called, since each net must have 'length' and 'weight' fields. Does not modify `state`, 
    so it is no longer necessary to create a deepcopy of `state` each time a new move is proposed (unlike in
    previous iteration). 

    There is a chance (depending on random_move_chance) that the move proposed is completely random. In this case,
    the selected cell is random and the selected destination is also random. 

    Lookups is a lookup table that should be generated once before the SA algorithm starts iterating.

    Returns a dict describing the proposal:
      {
        'net_name': str,
        'cell_to_move': str,
        'src': (x, y),
        'dst': (x, y),
        'swap_with': str | None,   # if dst occupied by an unfixed cell
        'target_cell': str
      }
    """
 
    if not (0.0 <= random_move_chance <= 1.0):
        raise ValueError("random_move_chance must be in [0, 1].")
    
    if lookups is None:
        lookups = build_fast_lookups(state)         # builds lookups if none are provided. This would be costly though. 
    
    cell_names = lookups["cell_names"]              # grabs the cell names
    all_coords = lookups['all_coords']              # grabs all the coordinates on the grid
    fixed_positions = lookups['fixed_positions']    # grabs all the fixed positions on the grid
    pos_to_cell = lookups['pos_to_cell']            # grabs all coordinate: cell name pairs

    # decide if a random move will be made
    rng_branch = random.Random(seeds[3])            # seeds a random number generator used to decide if random mode is used
    if rng_branch.random() < random_move_chance:
        rng_rand = random.Random(seeds[4])          # seeds a random number generator used to pick a random cell

        while True:
            cell_to_move = rng_rand.choice(cell_names)          # randomly chooses one of the cells          
            if state['cells'][cell_to_move]['fixed'] is False:  # makes sure that the cell chosen is unfixed.
                break
        src = state['cells'][cell_to_move]['position']          # coordinates of the cell being moved

        while True:
            dst = rng_rand.choice(all_coords)                   # randomly selects a destination coordinates
            if dst not in fixed_positions:                      # makes sure that dst is not in fixed_positions
                break

        g = state["grid_size"]

        swap_with = None                                        # name of the cell that we are swapping with 
        occupant = pos_to_cell.get(dst)                         # gets the name of the cell occupying dst. occupant == None if no cell occupies those coords. 
        if occupant is not None and occupant != cell_to_move:
            # dst cannot be fixed (we excluded fixed_positions), so swap is ok
            swap_with = occupant

        return {
            "net_name": "__RANDOM__",           # no net was chosen in this random mode
            "cell_to_move": cell_to_move,       # cell that was randomly selected to move
            "src": src,                         # original coords of that cell
            "dst": dst,                         # destination coords of that cell
            "swap_with": swap_with,             # name of the cell that we are swapping with
            "target_cell": None,                # no target cell in this random choice mode (target cell is the one that cell_to_move is trying to be close to)
        }       

    nets = []
    weights = []

    for net in state["nets"]:
        if "weight" not in net or "length" not in net:
            raise ValueError(
                "Net is missing 'length'/'weight'. Call annotate_net_lengths_and_weights(state) before propose_move()."
            )

        if any(state["cells"][cell_name]["fixed"] is False for cell_name in net["cells"]):          # checks to see if either of the cells in the net are fixed or not
            nets.append(net)                                                                        # if so, add this net to the nets list. Same wigh weights list below. 
            weights.append(net["weight"])

    if not nets:    # checks if nets is populated. 
        raise ValueError("No nets contain an unlocked cell, so no move can be proposed.")

    rng_net = random.Random(seeds[0])
    chosen_net = rng_net.choices(nets, weights=weights, k=1)[0]                                     # randomly selects one of the nets with unfixed cell(s). More weight is given to the cells that have a longer length. 

    unlocked_cell_mask = [state["cells"][cell_name]["fixed"] is False for cell_name in chosen_net["cells"]] # creates a mask that identifies which of the cells in the selected net are fixed/unfixed
    unlocked_cells = np.array(chosen_net["cells"])[unlocked_cell_mask]                                      # applies the mask to select only the unfixed cells

    rng_cell = random.Random(seeds[1])
    cell_to_move = rng_cell.choices(list(unlocked_cells), k=1)[0]                                           # randomly selects one of the unfixed cells on that net. If only one cell is unfixed, obviously that one will be chosen. Otherwise, it is a 50/50 shot.

    src = state["cells"][cell_to_move]["position"]                                                          # src encodes the coordinates of the cell_to_move

    c0, c1 = chosen_net["cells"]                                                                            # c0, c1 are the names of the cells on the selected net
    target_cell = c1 if cell_to_move == c0 else c0                                                          # target_cell is the cell on the net that is NOT the cell_to_move

    dst = search_ring(                                                                                      # grabs the coordinates of the destination cell, computed according to search_ring
        state=state,    
        target_coordinates=state["cells"][target_cell]["position"],
        grid_size=state["grid_size"],
        seed=seeds[2],
    )

    swap_with = None                                                                                        # swap_with initialized to None if dst is not occupied. 
    for cell_name, cell in state["cells"].items():
        if cell["position"] == dst:
            if cell["fixed"] is True:
                raise ValueError("Destination occupied by fixed cell (unexpected).")                        # throws an error if dst is already occupied by a fixed cell. This should never happen, but this is just to be safe
            swap_with = cell_name                                                                           # swap_with contains the name of the cell that is currently occupying dst
            break

    return {
        "net_name": chosen_net.get("name", ""), # name of the net that is being chosen 
        "cell_to_move": cell_to_move,           # name of the cell on net_name that is being moved (or at least the one for which a move is proposed)
        "src": src,                             # original coordinates of the cell that we are proposing to move
        "dst": dst,                             # coordinates of the destination for the cell we are moving
        "swap_with": swap_with,                 # name of the cell that occupies dst, if one exists (otherwise None)
        "target_cell": target_cell,             # name of the cell on net_name that is *not* being chosen to move. cell_to_move is trying to be close as possible to target_cell.
    }


def compute_move_cost_update(state: dict, proposal: dict, current_cost: float) -> tuple[float, float, list[dict]]:
    """
    Computes the change in total cost resulting from a proposed move without modifying `state` (in 
    other words, without actually making the move). This function evaluates the effect of moving a
    single cell (and swapping with another, if a swap is proposed) by recomputing the lengths and
    weights of only the nets touched by the move.

    This function assumes that annotate_net_lengths_and_weights(state) has already been called, so
    that state['nets'] contain the required 'length' and 'weight' fields. The proposed move must be 
    generated by proposed_move(), since this function relies on the `proposal` dictionary that is 
    returned there. 

    This function also returns a list of dictionaries containing all required information to update 
    state['nets'] and state['cells'] if the proposed move ends up being accepted. Each dictionary in 
    this list contains the index of the net, the name of the net, the old length of the net, the new 
    length of the net, the old weight of the net, and the new weight of the net. 
    
    :param state: Current placement state.
    :type state: dict
    :param proposal: Dictionary describing a proposed move, including the cell to move, source and
                     destination coordinates, and optional swap information.
    :type proposal: dict
    :param current_cost: Current total placement cost prior to applying the proposed move.
    :type current_cost: float
    :return: A tuple containing:
             (1) the new total cost after applying the proposed move,
             (2) how much the total cost changed (positive or negative),
             (3) a list of dictionaries describing per-net updates for all affected nets.
    :rtype: tuple[float, float, list[dict]]
    """
    
    if "cell_to_move" not in proposal or "src" not in proposal or "dst" not in proposal:            # raise an error if the required fields are not present.
        raise ValueError("proposal dict missing required keys (cell_to_move/src/dst).")

    moved = proposal["cell_to_move"]                                    # grabs the name of the cell that we are proposing to move               
    swap_with = proposal.get("swap_with", None)                         # grabs the name of the cell that is being swapped. If there is no swap, grab None. Technically proposal['swap_with'] already contains None if there is no swap, but this is added safety. 

    virtual_pos = {moved: proposal["dst"]}                              # assigns virtual_pos = {cell_to_move: coordinates of destination}
    if swap_with is not None:
        virtual_pos[swap_with] = proposal["src"]                        # if a cell is being swapped, virtual_pos = {cell_to_move: coordinates of destination, name_of_swapped_cell: original coords of cell_to_move}

    touched_cells = {moved}                                             # touched_cells is a set containing the cell that is being moved and the cell that is swapped (if exists). This is necessary to identify the nets that must be updated. 
    if swap_with is not None:
        touched_cells.add(swap_with)

    max_length = 2 * (state["grid_size"] - 1)

    delta = 0                                                           # delta will be used to track how much the total cost changes as nets are updated with new positions
    net_updates: list[dict] = []                                        # creates new nets in the same format as in the state variable

    for idx, net in enumerate(state["nets"]):                           # idx indexes each net in state['nets'] to keep track of ones that contain `moved` and `swap_with`
        if "length" not in net or "weight" not in net:
            raise ValueError("Net missing 'length'/'weight'. Run annotate_net_lengths_and_weights(state) first.")   # this error will be thrown if annotate_net_lengths_and_weights() has not been called

        a, b = net["cells"]                                             # a,b are the names of the cells on the net
        if (a not in touched_cells) and (b not in touched_cells):       # if moved or swap_with are not on this net, skip to the next idx, net iteration. 
            continue  
        
        # everything below here executes when the net contains one of the cells being moved
        old_len = net["length"]                                         # grab the old net length 
        old_wt = net["weight"]                                          # grab the old net weight

        ax, ay = virtual_pos.get(a, state["cells"][a]["position"])      # if a is the cell that we are proposing to move or is the cell being swapped, (ax,ay) contains the coordinates of its destination. If a is an unaffected cell on the net, (ax,ay) are the coordinates specified for that cell in `state`
        bx, by = virtual_pos.get(b, state["cells"][b]["position"])      # same as above but for b. 
        new_len = abs(ax - bx) + abs(ay - by)                           # computes the new length of the net
        new_wt = new_len / max_length                                   # computes the new weight of the net
 
        delta += (new_len - old_len)                                    # finds out how much the length of that net changes; adds to the total. No abs here, because a negative number should result in the total cost decreasing
 
        net_updates.append(                                             # adds information of the updated net to net_updates. Once again, note that `state` has thusfar remained unchanged. 
            {
                "net_index": idx,                   # index of the updated net
                "net_name": net.get("name", ""),    # name of the updated net
                "old_length": old_len,              # old length of the updated net
                "new_length": new_len,              # new length of the updated net
                "old_weight": old_wt,               # old weight of the updated net
                "new_weight": new_wt,               # new weight of the udpated net
            }
        )

    new_cost = current_cost + delta                 # if delta is negative, the new cost is lower. Otherwise it is larger. 
    return new_cost, delta, net_updates             # returns a tuple containing the new TOTAL cost after the proposed net, the change in the total cost, and all of the nets being updated. 

def apply_proposed_move(
    state: dict,
    proposal: dict,
    net_updates: list[dict] | None = None,
    lookups: dict | None = None,
) -> dict:
    """
    Applies the previously proposed placement move to the current placement state by updating
    cell coordinates and updating affected net data (such as lengths and weights). 

    This function updates the position of the moved cell, swaps it with another cell if the 
    destination is already occupied by a movable cell, and then updates the lengths and weights
    of each net that is affected by the coordinate swaps. If a precomputed list of net updates
    (from the compute_move_cost_update() function) is given, then this function does not recompute
    anything. If no list of net updates is provided, this function will compute the updates. 
    
    If `lookups` is provided, it is updated here.
      - lookups["pos_to_cell"] is updated for the moved (and swapped) cell(s)
      - lookups["fixed_positions"] is unchanged here (since this function does not change fixed flags)

    :param state: Current placement state.
    :type state: dict
    :param proposal: Dictionary describing the proposed move. Must contain the keys
                     "cell_to_move", "src", and "dst". May optionally contain "swap_with"
                     indicating a cell to be swapped with the moved cell.
    :type proposal: dict
    :param net_updates: Optional list of dictionaries encoding precomputed net updates 
                        produced by compute_move_cost_update(). Each dictionary must 
                        specify the affected net index along with its new length and weight.
    :type net_updates: list[dict] | None
    :param lookups: Optional cached lookups from build_fast_lookups(state):
                    {"pos_to_cell": ..., "fixed_positions": ..., "cell_names": ..., "all_coords": ...}
    :return: The updated placement state. The returned value is the same object as the input state.
    :rtype: dict
    """
    moved = proposal["cell_to_move"]                     # name of the cell being moved
    src = proposal["src"]                                # original coordinates of the cell being moved         
    dst = proposal["dst"]                                # destination coords of the cell being moved
    swap_with = proposal.get("swap_with", None)          # name of the cell being swapped, if one exists. 

    # next two lines store references to the global lookups[...] object. So when pos_to_cell is modified, lookups[...] will be as well.
    pos_to_cell = lookups.get("pos_to_cell") if lookups is not None else None               # grabs the dicitonary containing current coordinate: cell name pairs. Once the move is accepted, we need to modify this.
    fixed_positions = lookups.get("fixed_positions") if lookups is not None else None       # grabs the set containing the fixed positions. this one will not need to be updated

    # safety check to make sure that the destination is valid
    if fixed_positions is not None and dst in fixed_positions:
        raise ValueError("Destination is a fixed position; refusing to apply move.")

    # additional safety checks. Makes sure that the original coordinates for the cell we are moving is correct. Does same thing with the cell being swapped. These two lines might save some time if we comment out, but probably not much.
    if state["cells"][moved]["position"] != src:
        raise ValueError("Proposal 'src' does not match current state for moved cell (stale proposal?).")
    if swap_with is not None and state["cells"][swap_with]["position"] != dst:
        raise ValueError("Proposal 'dst' does not match current state for swap_with cell (stale proposal?).")

    
    state["cells"][moved]["position"] = dst             # updates the coordinates of the cell being moved to the destination coords. 
    if swap_with is not None:               
        state["cells"][swap_with]["position"] = src     # also updates the coordinates of the cell that we are swapping. 

    # now we also need to update the coordinates stored in pos_to_cell. We are modifying pos_to_cell, which points to the lookups['pos_to_cell'] object. Therefore, when we modify pos_to_cell, the lookups dict will change globally.
    if pos_to_cell is not None:
        if pos_to_cell.get(src) == moved:           # if the name of the cell corresponding to the src coordinates is the cell being moved (this should be the case)
            pos_to_cell.pop(src, None)              # then, get rid of it. If src does not exist, then None is returned instead of throwing an error. 
        
        pos_to_cell[dst] = moved                    # after the move, the coordinates dst now correspond to the name of the cell that we just placed there

        if swap_with is not None:                   # if we are making a swap
            pos_to_cell[src] = swap_with            # then, the coordinates src now corresopnd with the name of the cell we are swapping with 

    # updates net data
    if net_updates is not None:                     # use the net_updates list that was output in compute_move_cost_update to replace affected nets with the already computed metadata
        for upd in net_updates:                     # each upd dictionary contains the changes on a particular net
            net = state["nets"][upd["net_index"]]   # grabs the net dictionary in the state['nets'] list at index upd['net_index']. Assigns to net variable
            net["length"] = upd["new_length"]       # updates the length of the new net
            net["weight"] = upd["new_weight"]       # updates the weight of the new net
        return state                                # returning state (rather than nothing at all) is not necessary. However, in the SA alogorithm `state = apply_proposed_state(...)` makes it slightly more obvious that state mutation is occuring
    
    # if a net_updates list is not provided, we need to compute the new length and weight of each net. The code below is the same as in the compute_move_cost_update() function. 
    touched = {moved}
    if swap_with is not None:
        touched.add(swap_with)

    max_length = 2 * (state["grid_size"] - 1)

    for net in state["nets"]:
        a, b = net["cells"]
        if (a not in touched) and (b not in touched):
            continue
        ax, ay = state["cells"][a]["position"]
        bx, by = state["cells"][b]["position"]
        new_len = abs(ax - bx) + abs(ay - by)
        net["length"] = new_len
        net["weight"] = new_len / max_length

    return state            # Again, it is not necessary that we actually return state. the state parameter is passed by reference, so any mutations we already did inside the function have already happened, regardless of what we return. 

def accept_move(d_cost: int, T: int, k: int, seed: int) -> bool:
    """
    Decides whether or not to accept a proposed move. A move that does not increase cost (d_cost <= 0) is always
    accepted. A move that increases cost is accepted probabilistically according to the boltzmann factor
    exp(-d_cost / (k * T))
    
    :param d_cost: Change in cost resulting from the proposed move
                   (new_cost - current_cost).
    :type d_cost: int
    :param T: Current annealing temperature. Higher values increase the
              probability of accepting worse moves.
    :type T: int
    :param k: Scaling factor that controls sensitivity to cost increases.
    :type k: int
    :param seed: Seed for the random number generator to ensure reproducible
                 decisions.
    :type seed: int
    :return: True if the move is accepted, False otherwise.
    :rtype: bool
    """
    if d_cost <= 0:
        return True
    boltz = math.exp(-1*d_cost / (k*T))
    r = random.Random(seed).random()        # chooses random number between 0 and 1 with a seed
    if r < boltz:
        return True
    else: 
        return False

#below is a purely chatgpt generated code specifically designed to test my perturb() function. I will be removing this function in the final version. 
def plot_placement(state, *, show_nets=True, label_cells=True, title=None):
    """
    Plot an NxN placement grid with shaded occupied cells and optional net lines.
    Coordinates are assumed to be integer cell indices with (0,0) at bottom-left.
    """
    N = state["grid_size"]

    fig, ax = plt.subplots(figsize=(6, 6))

    # --- grid lines (cell boundaries) ---
    for k in range(N + 1):
        ax.plot([k, k], [0, N], linewidth=1)
        ax.plot([0, N], [k, k], linewidth=1)

    # --- occupied cells ---
    for name, info in state["cells"].items():
        x, y = info["position"]
        ax.add_patch(Rectangle((x, y), 1, 1, alpha=0.35))  # no explicit color
        if label_cells:
            ax.text(x + 0.5, y + 0.5, name, ha="center", va="center", fontsize=8)

    # --- nets (center-to-center) ---
    if show_nets and "nets" in state:
        for net in state["nets"]:
            a, b = net["cells"]
            xa, ya = state["cells"][a]["position"]
            xb, yb = state["cells"][b]["position"]
            ax.plot([xa + 0.5, xb + 0.5], [ya + 0.5, yb + 0.5],
                    linestyle="--", linewidth=1, alpha=0.7)

    # --- axes formatting ---
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Placement (occupied cells + nets)")
    plt.show()

def unfix_all(state: dict) -> None:
    # All movable cells are set to the fixed = False state. 
    for cell in state['cells'].values():
        if cell['type'] == 'MOVABLE':
            cell['fixed'] = False