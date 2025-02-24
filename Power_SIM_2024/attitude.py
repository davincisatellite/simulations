import numpy as np
import matplotlib.pyplot as plt

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


def quaternion_multiplication(quat_1, quat_2):
    """
    Multiplies two quaternions with each other, in the order quat_1 * quat_2.
    Inputs:
    - quat_1: 4-element array (Quaternion) as defined by get_quaternion.
    - quat_2: See quat_1.
    Outputs:
    - quat_res: 4-element array quaternion. 
    """
    quat_res0 = quat_1[0]*quat_2[0] - quat_1[1]*quat_2[1] -\
                quat_1[2]*quat_2[2] - quat_1[3]*quat_2[3]
    quat_res1 = quat_1[0]*quat_2[1] + quat_1[1]*quat_2[0] -\
                quat_1[2]*quat_2[3] + quat_1[3]*quat_2[2]
    quat_res2 = quat_1[0]*quat_2[2] + quat_1[1]*quat_2[3] +\
                quat_1[2]*quat_2[0] - quat_1[3]*quat_2[1]
    quat_res3 = quat_1[0]*quat_2[3] - quat_1[1]*quat_2[2] +\
                quat_1[2]*quat_2[1] + quat_1[3]*quat_2[0]
    
    quat_res = np.array([quat_res0, quat_res1, quat_res2, quat_res3])

    return quat_res
    

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
    print(f"Point 1 in quaternion form: {p_1}")

    q_inv = np.append(quaternion[0], -quaternion[1:])
    print(quaternion, q_inv)

    s_1 = quaternion_multiplication(q_inv, p_1)
    rotated_pt = quaternion_multiplication(s_1, quaternion)

    return rotated_pt


def fibonacci_sphere(points=100):
    """
    Creates evenly spaced out points in a radius 1 sphere. 
    Copied from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    Inputs:
    - points: Number of evenly spaced out points in the sphere
    Outputs:
    - points: 
    """
    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(points):
        y = 1 - (i / float(points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return points

p = np.array([1,0,0])
rot_vec = np.array([0,0,1])
rot_ang = 45 * (np.pi/180)

q = get_quaternion(rot_vec, rot_ang)

p_2 = quaternion_rotation(p, q)

print(f"Point rotated:{p_2}")

points = np.array(fibonacci_sphere())
