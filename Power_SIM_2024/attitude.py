import numpy as np

from numpy import sin, cos

def get_quaternion(rot_vec, angle):
    """
    Returns the quaternion for a known rotation axis and angle of rotation.
    Inputs:
    - rot_vec: 3-element rotational unit vector. 
    - angle: Angle of rotation 
    """
    q_0 = cos(angle/2)
    q_1 = rot_vec[0] * sin(angle/2)
    q_2 = rot_vec[1] * sin(angle/2)
    q_3 = rot_vec[2] * sin(angle/2)
    

    return np.array([q_0, q_1, q_2, q_3])

def quaternion_rotation(point, quaternion):
    """
    Returns the result of applying a rotation to the given point. Rotation 
    given by a quaternion.
    Inputs:
    - point: 3-element array of rotated point coords.
    - quaternion: 4-element array of quaternion elements. As defined in 
    get_quaternion().
    """
    p_1 = np.append(0, point)
    print(p_1)

    q_inv = np.append(quaternion[0], -quaternion[1:])
    print(quaternion, q_inv)

    return q_inv * p_1 * quaternion

p = np.array([1,0,0])
rot_vec = np.array([0,0,1])
rot_ang = 90 * (np.pi/180)

q = get_quaternion(rot_vec, rot_ang)

p_2 = quaternion_rotation(p, q)

print(p_2)