import numpy as np
import cv2

# Colors in BGR for OpenCV
COLORS = {
    'W': (255, 255, 255),  # White
    'R': (0, 0, 255),      # Red
    'B': (255, 0, 0),      # Blue
    'O': (0, 165, 255),    # Orange
    'G': (0, 255, 0),      # Green
    'Y': (0, 255, 255)     # Yellow
}

class Cube2D:
    def __init__(self):
        # 6 faces: U(p), R(ight), F(ront), D(own), L(eft), B(ack)
        self.faces = {
            'U': np.full((3, 3), 'W'),  # White top
            'R': np.full((3, 3), 'R'),  # Red right
            'F': np.full((3, 3), 'B'),  # Blue front
            'D': np.full((3, 3), 'Y'),  # Yellow bottom
            'L': np.full((3, 3), 'O'),  # Orange left
            'B': np.full((3, 3), 'G')   # Green back
        }

    def rotate_face_cw(self, face):
        """Rotate a face clockwise (90 degrees)"""
        self.faces[face] = np.rot90(self.faces[face], -1)
        self._rotate_adjacent_edges_cw(face)

    def rotate_face_ccw(self, face):
        """Rotate a face counter-clockwise (90 degrees)"""
        self.faces[face] = np.rot90(self.faces[face], 1)
        self._rotate_adjacent_edges_ccw(face)

    def _rotate_adjacent_edges_cw(self, face):
        """Rotate the adjacent edges when a face is rotated clockwise"""
        if face == 'F':
            # Front face rotation affects U bottom row, R left col, D top row, L right col
            temp = self.faces['U'][2, :].copy()
            self.faces['U'][2, :] = self.faces['L'][:, 2][::-1]
            self.faces['L'][:, 2] = self.faces['D'][0, :]
            self.faces['D'][0, :] = self.faces['R'][:, 0][::-1]
            self.faces['R'][:, 0] = temp
        
        elif face == 'U':
            # Up face rotation affects F top row, L top row, B top row, R top row
            temp = self.faces['F'][0, :].copy()
            self.faces['F'][0, :] = self.faces['R'][0, :]
            self.faces['R'][0, :] = self.faces['B'][0, :]
            self.faces['B'][0, :] = self.faces['L'][0, :]
            self.faces['L'][0, :] = temp
        
        elif face == 'R':
            # Right face rotation affects U right col, B left col, D right col, F right col
            temp = self.faces['U'][:, 2].copy()
            self.faces['U'][:, 2] = self.faces['F'][:, 2]
            self.faces['F'][:, 2] = self.faces['D'][:, 2]
            self.faces['D'][:, 2] = self.faces['B'][:, 0][::-1]
            self.faces['B'][:, 0] = temp[::-1]

    def _rotate_adjacent_edges_ccw(self, face):
        """Rotate the adjacent edges when a face is rotated counter-clockwise"""
        if face == 'F':
            # Front face rotation affects U bottom row, R left col, D top row, L right col
            temp = self.faces['U'][2, :].copy()
            self.faces['U'][2, :] = self.faces['R'][:, 0]
            self.faces['R'][:, 0] = self.faces['D'][0, :][::-1]
            self.faces['D'][0, :] = self.faces['L'][:, 2]
            self.faces['L'][:, 2] = temp[::-1]
        
        elif face == 'U':
            # Up face rotation affects F top row, L top row, B top row, R top row
            temp = self.faces['F'][0, :].copy()
            self.faces['F'][0, :] = self.faces['L'][0, :]
            self.faces['L'][0, :] = self.faces['B'][0, :]
            self.faces['B'][0, :] = self.faces['R'][0, :]
            self.faces['R'][0, :] = temp
        
        elif face == 'R':
            # Right face rotation affects U right col, B left col, D right col, F right col
            temp = self.faces['U'][:, 2].copy()
            self.faces['U'][:, 2] = self.faces['B'][:, 0][::-1]
            self.faces['B'][:, 0] = self.faces['D'][:, 2][::-1]
            self.faces['D'][:, 2] = self.faces['F'][:, 2]
            self.faces['F'][:, 2] = temp

    def rotate_row_left(self, row):
        """Rotate a horizontal row to the left"""
        temp = self.faces['U'][row].copy()
        self.faces['U'][row] = self.faces['R'][row]
        self.faces['R'][row] = self.faces['D'][row]
        self.faces['D'][row] = self.faces['L'][row]
        self.faces['L'][row] = temp
        
        # If rotating top or bottom row, also rotate the corresponding face
        if row == 0:
            self.rotate_face_ccw('U')
        elif row == 2:
            self.rotate_face_cw('D')

    def rotate_row_right(self, row):
        """Rotate a horizontal row to the right"""
        temp = self.faces['U'][row].copy()
        self.faces['U'][row] = self.faces['L'][row]
        self.faces['L'][row] = self.faces['D'][row]
        self.faces['D'][row] = self.faces['R'][row]
        self.faces['R'][row] = temp
        
        # If rotating top or bottom row, also rotate the corresponding face
        if row == 0:
            self.rotate_face_cw('U')
        elif row == 2:
            self.rotate_face_ccw('D')

    def rotate_col_up(self, col):
        """Rotate a vertical column up"""
        temp = self.faces['U'][:, col].copy()
        self.faces['U'][:, col] = self.faces['F'][:, col]
        self.faces['F'][:, col] = self.faces['D'][:, col]
        self.faces['D'][:, col] = self.faces['B'][:, 2-col][::-1]  # Back face is mirrored
        self.faces['B'][:, 2-col] = temp[::-1]
        
        # If rotating left or right column, also rotate the corresponding face
        if col == 0:
            self.rotate_face_ccw('L')
        elif col == 2:
            self.rotate_face_cw('R')

    def rotate_col_down(self, col):
        """Rotate a vertical column down"""
        temp = self.faces['U'][:, col].copy()
        self.faces['U'][:, col] = self.faces['B'][:, 2-col][::-1]  # Back face is mirrored
        self.faces['B'][:, 2-col] = self.faces['D'][:, col][::-1]
        self.faces['D'][:, col] = self.faces['F'][:, col]
        self.faces['F'][:, col] = temp
        
        # If rotating left or right column, also rotate the corresponding face
        if col == 0:
            self.rotate_face_cw('L')
        elif col == 2:
            self.rotate_face_ccw('R')

    def draw(self, size=50):
        """
        Returns an OpenCV image showing the cube in 2D
        Layout:
              U
            L F R B
              D
        """
        # Create blank image 
        img_height = 9 * size
        img_width = 12 * size
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        def draw_face(face_array, top_left):
            x0, y0 = top_left
            for i in range(3):
                for j in range(3):
                    color = COLORS[face_array[i, j]]
                    # Draw filled rectangle
                    cv2.rectangle(img, (x0 + j*size, y0 + i*size),
                                  (x0 + (j+1)*size, y0 + (i+1)*size),
                                  color, -1)
                    # Draw black border
                    cv2.rectangle(img, (x0 + j*size, y0 + i*size),
                                  (x0 + (j+1)*size, y0 + (i+1)*size),
                                  (0, 0, 0), 2)

        # Position faces in the layout
        positions = {
            'U': (3*size, 0),           # Top
            'L': (0, 3*size),           # Left
            'F': (3*size, 3*size),      # Front 
            'R': (6*size, 3*size),      # Right
            'B': (9*size, 3*size),      # Back
            'D': (3*size, 6*size)       # Bottom
        }

        # Draw all faces
        for face_name, pos in positions.items():
            draw_face(self.faces[face_name], pos)

        # Add face labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        thickness = 1

        for face_name, pos in positions.items():
            x, y = pos
            cv2.putText(img, face_name, (x + size//3, y - 10), 
                       font, font_scale, font_color, thickness)

        return img

    def reset(self):
        """Reset cube to solved state"""
        self.__init__()

    def scramble(self, moves=20):
        """Randomly scramble the cube"""
        import random
        
        possible_moves = [
            ('rotate_face_cw', ['U', 'R', 'F', 'D', 'L', 'B']),
            ('rotate_face_ccw', ['U', 'R', 'F', 'D', 'L', 'B']),
            ('rotate_row_left', [0, 1, 2]),
            ('rotate_row_right', [0, 1, 2]),
            ('rotate_col_up', [0, 1, 2]),
            ('rotate_col_down', [0, 1, 2])
        ]
        
        for _ in range(moves):
            move_type, options = random.choice(possible_moves)
            option = random.choice(options)
            
            if move_type == 'rotate_face_cw':
                self.rotate_face_cw(option)
            elif move_type == 'rotate_face_ccw':
                self.rotate_face_ccw(option)
            elif move_type == 'rotate_row_left':
                self.rotate_row_left(option)
            elif move_type == 'rotate_row_right':
                self.rotate_row_right(option)
            elif move_type == 'rotate_col_up':
                self.rotate_col_up(option)
            elif move_type == 'rotate_col_down':
                self.rotate_col_down(option)

    def is_solved(self):
        """Check if the cube is in a solved state"""
        for face_name, face_array in self.faces.items():
            # Check if all squares in each face are the same color
            first_color = face_array[0, 0]
            if not np.all(face_array == first_color):
                return False
        return True

    def get_state_string(self):
        """Get a string representation of the cube state"""
        state = ""
        for face_name in ['U', 'R', 'F', 'D', 'L', 'B']:
            face = self.faces[face_name]
            for i in range(3):
                for j in range(3):
                    state += face[i, j]
        return state