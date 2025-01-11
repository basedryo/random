'''
    This Class allows for a simulation of 3x3x3 of the Rubick's cube
    It only applies rotation to the front face, hence for a generic face the cube orientation is changed first.
    The move is then applied to the front and the cube orientation is reset to the initial one.
    
    Code author: alessandro1.barro@mail.polimi.it
'''

import random
import numpy as np

class RubiksCube:
    def __init__(self):
        '''
        Initialize the cube object, contains every index and mapping necessary for object manipulation.
        Some of them may be editable, but be cautious for consistency.
        '''

        # Cube keys
        self.face_keys = ['F', 'B', 'U', 'D', 'R', 'L'] # Face keys (orientation may be edited, beware of color_keys consistency)
        self.color_keys = [1, 2, 3, 4, 5, 6] # e.g. [White, Yellow, Blue, Green, Red, Orange]
        self.direction_keys = ['+', '-'] # Clockwise and counterclockwise
        self.orientation_key = ['h', 'v'] # Horizontal and vertical (for orientation manipulation)
        self.move_keys = [('F', '+'), ('F', '-'), ('B', '+'), ('B', '-'), ('U', '+'), ('U', '-'), ('D', '+'), ('D', '-'), ('R', '+'), ('R', '-'), ('L', '+'), ('L', '-')] # Set of possible moves

        # Other indices
        self.piece_positions = {f'{face_key}{i}{j}': (face_key, i, j) for face_key in self.face_keys for i in range(3) for j in range(3)}
        self.adjacency = {'F': [], 'B': [('h', 2)], 'U': [('v', 1)], 'D': [('v', -1)], 'L': [('h', -1)], 'R': [('h', 1)]}
        self.counter_adjacency = {'F': [], 'B': [('h', 2)], 'U': [('v', -1)], 'D': [('v', 1)], 'L': [('h', 1)], 'R': [('h', -1)]}
        
        # State initialization
        self.state = {face_key: color_key * np.ones((3, 3), dtype=object) for face_key, color_key in zip(self.face_keys, self.color_keys)}
        self.solved_state = self.state

    def change_orientation(self, orientation_key):
        '''
        Changes the orientation of the cube. While it can be a helpful standalone function, it mainly
        serves for applying moves (in this model, only F,F' are applied, but orientation change allow
        for every face to be transposed into F).
        '''
        if orientation_key == 'h': # Front becomes the old left
            temp = self.state['F'].copy()
            self.state['F'] = self.state['R']
            self.state['R'] = self.state['B']
            self.state['B'] = self.state['L']
            self.state['L'] = temp
            self.state['U'] = np.rot90(self.state['U'], -1)
            self.state['D'] = np.rot90(self.state['D'], 1)
        elif orientation_key == 'v': # Front becomes the old up
            temp = self.state['F'].copy()
            self.state['F'] = self.state['U']
            self.state['U'] = np.rot90(self.state['B'], 2)
            self.state['B'] = np.rot90(self.state['D'], 2)
            self.state['D'] = temp
            self.state['L'] = np.rot90(self.state['L'], -1)
            self.state['R'] = np.rot90(self.state['R'], 1)
    
    def perform_rotation(self, direction_key):
        '''
        Actually applies the move of choice. Once the selected face is brought to F with fw_set_front(),
        then F' or F is applied accordingly. Then the orientation is reset with bw_set_front().
        '''

        if direction_key == '+':
            self.state['F'] = np.rot90(self.state['F'], -1)
            temp = self.state['U'][2, :].copy()
            self.state['U'][2, :] = np.flip(self.state['L'][:, 2])
            self.state['L'][:, 2] = self.state['D'][0, :]
            self.state['D'][0, :] = np.flip(self.state['R'][:, 0])
            self.state['R'][:, 0] = temp
        elif direction_key == '-':
            self.state['F'] = np.rot90(self.state['F'], 1)
            temp = self.state['U'][2, :].copy()
            self.state['U'][2, :] = self.state['R'][:, 0]
            self.state['R'][:, 0] = np.flip(self.state['D'][0, :])
            self.state['D'][0, :] = self.state['L'][:, 2]
            self.state['L'][:, 2] = np.flip(temp)

    def forward_set_front(self, face_key):
        for d, n in self.adjacency[face_key]:
            for _ in range(n % 4):
                self.change_orientation(d)

    def backward_set_front(self, face_key):
        for d, n in self.counter_adjacency[face_key]:
            for _ in range(n % 4):
                self.change_orientation(d)

    def move(self, face_key, direction_key):
        '''
        Collects the necessary elements to perform some move. Notation can be found in the __init__()
        method.
        '''

        self.forward_set_front(face_key)
        self.perform_rotation(direction_key)
        self.backward_set_front(face_key)

    def apply_random_move(self):
        '''
        Allows some random move for the current object state. Helpful for random walk methods and state-action
        transition exploration.
        '''

        face_key = random.choice(self.face_keys)
        direction_key = random.choice(self.direction_keys)
        self.action(face_key, direction_key)

    def copy_state(self):
        '''
        Copies the current state object of the cube.
        '''

        return {face: np.copy(self.state[face]) for face in self.face_keys}

    def restore_state(self, state_copy):
        '''
        Sets the current state object of the cube as the input one. Beware of dtype=object and overall consistency.
        '''

        for face in self.face_keys:
            self.state[face] = np.copy(state_copy[face])

    def reset_state(self, scramble_moves=0):
        '''
        Resets the state to the initial one.
        '''

        self.state = {face_key: np.full((3, 3), color_key, dtype='object') for face_key, color_key in zip(self.face_keys, self.color_keys)}

    def shuffle_state(self, t):
        for _ in range(t):
            face_key = random.choice(self.face_keys)
            direction_key = random.choice(self.direction_keys)
            self.move(face_key, direction_key)

    def is_solved(self):
        for face_key in self.state.values():
            if not np.all(face_key == face_key[0, 0]):
                return False
        return True

'''
#*** EXAMPLE USAGE ***
cube = RubiksCube() # Init cube instance

print(cube.state)

cube.move('F','+')

# Perform standard RUR'U' algorithm
#cube.move('R', '+')
#cube.move('U', '+')
#cube.move('R', '-')
#cube.move('U', '-')

print('*---*')
print(cube.state)

  # Check new state
'''

