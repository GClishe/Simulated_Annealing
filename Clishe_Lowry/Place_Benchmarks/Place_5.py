# Auto-generated placement data
# Valid Python structure

data = {
    'grid_size': 3,

    'cells': {
        'CELL_0': {'type': 'MOVABLE', 'fixed': False, 'position': (1, 2)},
        'CELL_1': {'type': 'MOVABLE', 'fixed': False, 'position': (1, 0)},
        'CELL_2': {'type': 'MOVABLE', 'fixed': False, 'position': (0, 0)},
        'CELL_3': {'type': 'MOVABLE', 'fixed': False, 'position': (0, 1)},
        'IO_0': {'type': 'IO', 'fixed': True, 'position': (2, 1)},
    },

    'nets': [
        {'name': 'NET_0', 'cells': ('IO_0', 'CELL_2')},
        {'name': 'NET_1', 'cells': ('CELL_0', 'CELL_1')},
        {'name': 'NET_10', 'cells': ('CELL_1', 'IO_0')},
        {'name': 'NET_11', 'cells': ('CELL_1', 'CELL_2')},
        {'name': 'NET_12', 'cells': ('CELL_3', 'CELL_2')},
        {'name': 'NET_2', 'cells': ('CELL_0', 'CELL_1')},
        {'name': 'NET_3', 'cells': ('CELL_2', 'CELL_3')},
        {'name': 'NET_4', 'cells': ('CELL_3', 'CELL_2')},
        {'name': 'NET_5', 'cells': ('CELL_2', 'CELL_0')},
        {'name': 'NET_6', 'cells': ('CELL_0', 'CELL_3')},
        {'name': 'NET_7', 'cells': ('CELL_3', 'CELL_2')},
        {'name': 'NET_8', 'cells': ('IO_0', 'CELL_3')},
        {'name': 'NET_9', 'cells': ('CELL_0', 'IO_0')},
    ]
}
