# Gesture Controlled Rubik's Cube Simulator

This is a Python based Rubik's Cube simulator that can be controlled entirely via hand gestures using MediaPipe and OpenCV. This project demonstrates the power of computer vision and gesture detection.

## Features

- Real-time hand gesture detection (swipes, pinch, fist, palm)
- Interactive 2D Rubik's Cube simulation with rotations of rows, columns, and faces
- Selection indicators for rows, columns, and faces
- Palm hold reset to reset the cube
- Scramble and solve checking functionality
- Gesture smoothing and cooldown to prevent accidental moves

## Technologies Used

- Python 3.x
- OpenCV – 2D cube visualization
- MediaPipe – Hand detection and landmark tracking
- NumPy – Cube state representation and matrix operations

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/gesture-rubiks-cube.git
cd gesture-rubiks-cube
```

2. Install required packages:

```bash
pip install requirements.txt
```

3. Run the simulator:

```bash
python main.py
```

## Usage

1. Make sure your webcam is connected.

2. Run the program.

3. Controls via Gestures:

- Single Finger Swipe: Rotate rows or columns
- Two Fingers Up/Down: Rotate selected face clockwise/counter-clockwise
- Pinch: Cycle through selected row, column, and face
- Fist: No action (can be extended for other features)
- Palm (Hold for 2.5s): Reset the cube to the solved state
- ESC key to exit the simulator.

## Future Enhancements

- Support for both hands and more complex gestures
- 3D cube visualization using OpenGL and Improved UI
- Undo/Redo moves
- Extend gesture recognition for VR/AR interfaces or robotic control

## References

(to be added...)
