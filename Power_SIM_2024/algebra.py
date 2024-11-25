import numpy as np

def rotationX(angle):
    """
    Returns the rotation matrix around the x-axis with a given angle.
    - angle: rotation angle around x-axis, following right-hand rule. [º]
    """
    angle = angle * np.pi/180
    return np.array(
        [[1, 0, 0],
         [0, np.cos(angle), np.sin(angle)],
         [0, -np.sin(angle), np.cos(angle)]]
    )

def rotationY(angle):
    """
    Returns the rotation matrix around the y-axis with a given angle.
    - angle: rotation angle around y-axis, following right-hand rule. [º]
    """
    angle = angle * np.pi/180
    return np.array(
        [[np.cos(angle), 0, -np.sin(angle)],
         [0, 1, 0],
         [np.sin(angle), 0, np.cos(angle)]]
    )

def rotationZ(angle):
    """
    Returns the rotation matrix around the z-axis with a given angle.
    - angle: rotation angle around z-axis, following right-hand rule. [º]
    """
    angle = angle * np.pi/180
    return np.array(
        [[np.cos(angle), np.sin(angle), 0],
         [-np.sin(angle), np.cos(angle), 0],
         [0, 0, 1]]
    )
