# --- coding: utf-8 ---
# @Time    : 12/10/24 11:12 PM        # 文件创建时间
# @Author  : htLiang
# @Email   : ryzeliang@163.com
import numpy as np
from ur_rtde import UR_rtde
import math

def rpy2rotating_vector(rpy):
    # rpy to R
    R = rpy2R(rpy)
    # R to rotating_vector
    return R2rotating_vector(R)


def rpy2R(rpy):  # [r,p,y] 单位rad
    rot_x = np.array([[1, 0, 0],
                      [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                      [0, math.sin(rpy[0]), math.cos(rpy[0])]])
    rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                      [0, 1, 0],
                      [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
    rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                      [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                      [0, 0, 1]])
    R = np.dot(rot_z, np.dot(rot_y, rot_x))
    return R


def R2rotating_vector(R):
    theta = math.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
    print(f"theta:{theta}")
    rx = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
    return np.array([rx, ry, rz]) * theta


def R2rpy(R):
    # assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

if __name__ == '__main__':
    tool_configuration = [0, 0.7, 0.2, -np.pi/2, -np.pi/2, 0]
    array_rpy = [tool_configuration[3], tool_configuration[4], tool_configuration[5]]
    print(array_rpy)
    array_ratating_vec = rpy2rotating_vector(array_rpy)
    target_tcp = [tool_configuration[0], tool_configuration[1], tool_configuration[2],
                  array_ratating_vec[0], array_ratating_vec[1], array_ratating_vec[2]]
    print(target_tcp)