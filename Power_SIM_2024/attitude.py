import numpy as np
import math

from numpy import sin, cos, max, min

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
    Output:
    - rotated_pt: Returns the cartesian position of the rotated point.
    """
    p_1 = np.append(0, point)

    # Check if no rotation is applied.
    if quaternion[0] == 1.0:
        rotated_pt = p_1
    else:
        # Creates inverted rotation quaternion
        q_inv = np.append(quaternion[0], -quaternion[1:])

        # Rotates the point quaternion
        s_1 = quaternion_multiplication(q_inv, p_1)
        rotated_pt = quaternion_multiplication(s_1, quaternion)

    return rotated_pt[1:]


def fibonacci_sphere(samples=1000):
    """
    Creates evenly spaced out points in a radius 1 sphere. 
    Copied from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    Inputs:
    - points: Number of evenly spaced out points in the sphere
    Outputs:
    - points: 
    """
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def power_output(orient, solar_arr):
    """
    Retrieves power production depending on satellite orientation. Considers
    an external reference frame where X-axis is sun-pointing. Assumes direct
    correlation between cosine of incidence and power production.
    Inputs:
    - orient: 3x3 array of satellite orientation within the out-of-vehicle
    frame.
    - solar_arr: 6-element array of the solar panel power production.
    """
    # Extracts X-component (Sun-facing) of each satellite attitude axis.
    Dx_x = orient[0,0]
    Dy_x = orient[1,0]
    Dz_x = orient[2,0]

    # Calculates power production in each of satellite's axes.
    Px = solar_arr[0]*max([Dx_x, 0]) + solar_arr[1]*np.abs(min([Dx_x, 0]))
    Py = solar_arr[2]*max([Dy_x, 0]) + solar_arr[3]*np.abs(min([Dy_x, 0]))
    Pz = solar_arr[4]*max([Dz_x, 0]) + solar_arr[5]*np.abs(min([Dz_x, 0]))

    return Px+Py+Pz




