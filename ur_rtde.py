# --- coding: utf-8 ---
# @Time    : 12/10/24 9:22 PM        # 文件创建时间
# @Author  : htLiang
# @Email   : ryzeliang@163.com
import time
import copy
import socket
import struct
import numpy as np
import math
from robotiq_gripper import RobotiqGripper
from realsenseD435i import Camera
import rtde_control
import rtde_receive


class UR_rtde:
    def __init__(self, tcp_host_ip="192.168.243.101", workspace_limits=None,
                 is_use_robotiq=True, is_use_camera=True):
        # Init varibles
        if workspace_limits is None:
            workspace_limits = [[-0.7, 0.7], [-0.7, 0.7], [0.00, 0.6]]
        self.workspace_limits = workspace_limits
        self.tcp_host_ip = tcp_host_ip
        self.is_use_robotiq = is_use_robotiq
        self.is_use_camera = is_use_camera
        self.rtde_receive = rtde_receive.RTDEReceiveInterface(tcp_host_ip)
        self.rtde_control = rtde_control.RTDEControlInterface(tcp_host_ip)

        # UR5 robot configuration
        # Default joint/tool speed configuration
        self.joint_acc = 1.4  # Safe: 1.4   8
        self.joint_vel = 1.05  # Safe: 1.05  3

        # Joint tolerance for blocking calls
        self.joint_tolerance = 0.01

        # Default tool speed configuration
        self.tool_acc = 0.5  # Safe: 0.5
        self.tool_vel = 0.2  # Safe: 0.2

        # Tool pose tolerance for blocking calls
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

        # robotiq85 gripper configuration
        if (self.is_use_robotiq):
            # reference https://gitlab.com/sdurobotics/ur_rtde
            # Gripper activate
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.tcp_host_ip, 63352)  # don't change the 63352 port
            self.gripper._reset()
            # print("Activating gripper...")
            # self.gripper.activate()
            time.sleep(1.5)

        # realsense configuration
        if (self.is_use_camera):
            # Fetch RGB-D data from RealSense camera
            self.camera = Camera()
            self.cam_intrinsics = self.camera.intrinsics  # get camera intrinsics
        # self.cam_intrinsics = np.array([615.284,0,309.623,0,614.557,247.967,0,0,1]).reshape(3,3)
        # # Load camera pose (from running calibrate.py), intrinsics and depth scale
        self.cam_pose = np.loadtxt('cam_pose/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('cam_pose/camera_depth_scale.txt', delimiter=' ')

        # Default robot home joint configuration (the robot is up to air)
        self.home_joint_config = [-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                                  (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                                  -(0 / 360.0) * 2 * np.pi, 0.0]

        # test
        # self.testRobot()

    # Test for robot controlmove_and_wait_for_pos
    def testRobot(self):
        try:
            print("Test for robot...")
    #

        except:
            print("Test fail! ")

    # joint control
    '''
    input:joint_configuration = 6 joint angle
    '''
    def move_j(self, joint_configuration, k_acc=1, k_vel=1, t=0, r=0):
        # adjust velocity and acceleration
        adjusted_acc = k_acc * self.joint_acc
        adjusted_vel = k_vel * self.joint_vel

        # send moveJ command
        self.rtde_control.moveJ(joint_configuration, adjusted_acc, adjusted_vel)

        # wait tcp reaching target pose
        while True:
            actual_joint_positions = self.rtde_receive.getActualQ()
            if all([abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
                break
            time.sleep(0.01)
        self.rtde_close()

    # joint control
    '''
    move_j_p(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0)
    input:tool_configuration=[x y z r p y]
    x y z is the target tcp position, units in meters
    rpy ,units in radians
    '''
    def move_j_p(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0):
        """
         move to target pose（tool_configuration is target pose）

        :param tool_configuration: target position [x, y, z, r, p, y] (units in meters and radians)
        :param k_acc: ratio of acc
        :param k_vel: ration of velocity
        :param t: time(optional parameters)
        :param r: radius（optional parameters）
        """
        # adjust velocity and acceleration
        adjusted_acc = k_acc * self.joint_acc
        adjusted_vel = k_vel * self.joint_vel

        # change rpy to rx ry rz      rotation vectors, Rodriguez form
        array_rpy = [tool_configuration[3], tool_configuration[4], tool_configuration[5]]
        array_ratating_vec = self.rpy2rotating_vector(array_rpy)
        target_tcp = [tool_configuration[0], tool_configuration[1], tool_configuration[2],
                      array_ratating_vec[0],array_ratating_vec[1],array_ratating_vec[2]]


        # use ik to solve joints angle
        joint_target = self.rtde_control.getInverseKinematics(target_tcp)  # rotation vectors, Rodriguez form

        # you can also use diy ik solver, but should notice the rotation vectors or not rpy
        # the rpy of universal robot is zyx

        if joint_target is None:
            raise ValueError("The target tool position could not be solved to the joint space configuration\
                             check if the position is reachable.")  # Tolerance > 1e-10

        # Send moveJ command
        self.rtde_control.moveJ(joint_target, adjusted_acc, adjusted_vel, t, r)

        # Waiting for the TCP to reach the target pose
        while True:
            actual_tool_positions = self.rtde_receive.getActualTCPPose()

            if all([abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            # if you wanna ensure rotation < tolerance, you can use it:
            # if all([abs(actual_tool_positions[j] - target_tcp[j]) < self.tool_pose_tolerance[j] for j in range(6)]):
            #     break
            time.sleep(0.01)
        time.sleep(1.5)  # Ensure robot stabilization


    # move_l is mean that the robot keep running in a straight line
    def move_l(self, tool_configuration, k_acc=1, k_vel=1, t=0, r=0):
        """
        :param tool_configuration: target position [x, y, z, r, p, y] (units in meters and radians)
        :param k_acc: ratio of acc
        :param k_vel: ration of velocity
        :param t: time(optional parameters)
        :param r: radius（optional parameters）
        """
        # adjust
        adjusted_acc = k_acc * self.joint_acc
        adjusted_vel = k_vel * self.joint_vel

        # change rpy to rx ry rz      rotation vectors, Rodriguez form
        array_rpy = [tool_configuration[3], tool_configuration[4], tool_configuration[5]]
        array_ratating_vec = self.rpy2rotating_vector(array_rpy)
        target_tcp = [tool_configuration[0], tool_configuration[1], tool_configuration[2],
                      array_ratating_vec[0],array_ratating_vec[1],array_ratating_vec[2]]

        # Send movel command
        self.rtde_control.moveL(target_tcp, adjusted_acc, adjusted_vel, t, r)

        # Waiting tcp to reach the target pose
        while True:
            actual_tool_positions = self.rtde_receive.getActualTCPPose()
            if all([abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            time.sleep(0.01)
        time.sleep(1.5)  #Ensure robot stabilization
        self.rtde_close()

    # move_c is mean that the robot move circle
    def move_c(self, pose_via, tool_configuration, k_acc=1, k_vel=1, r=0, mode=0):
        """
        :param tool_configuration: target position [x, y, z, r, p, y] (units in meters and radians)
        :param k_acc: ratio of acc
        :param k_vel: ration of velocity
        :param t: time(optional parameters)
        :param r: radius（optional parameters）
        """
        # adjust
        adjusted_acc = k_acc * self.tool_acc
        adjusted_vel = k_vel * self.tool_vel

        # change rpy to rx ry rz      rotation vectors, Rodriguez form
        array_rpy = [tool_configuration[3], tool_configuration[4], tool_configuration[5]]
        array_ratating_vec = self.rpy2rotating_vector(array_rpy)
        target_tcp = [tool_configuration[0], tool_configuration[1], tool_configuration[2],
                      array_ratating_vec[0],array_ratating_vec[1],array_ratating_vec[2]]

        # send movec command
        self.rtde_control.moveC(pose_via, target_tcp, adjusted_acc, adjusted_vel, mode)

        # Waiting tcp to reach the target pose
        while True:
            actual_tool_positions = self.rtde_receive.getActualTCPPose()
            if all([abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            time.sleep(0.01)
        time.sleep(1.5)  # ensure robot stabilization
        self.rtde_close()
    def go_home(self):
        self.move_j(self.home_joint_config)

    def restartReal(self):
        self.go_home()
        # robotiq gripper configuration
        if (self.is_use_robotiq):
            # reference https://gitlab.com/sdurobotics/ur_rtde
            # Gripper activate
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.tcp_host_ip, 63352)  # don't change the 63352 port
            self.gripper._reset()
            print("Activating gripper...")
            self.gripper.activate()
            time.sleep(1.5)

        # realsense configuration
        if (self.is_use_camera):
            # Fetch RGB-D data from RealSense camera
            self.camera = Camera()
            # self.cam_intrinsics = self.camera.intrinsics  # get camera intrinsics
            self.cam_intrinsics = self.camera.intrinsics
            # # Load camera pose (from running calibrate.py), intrinsics and depth scale
            self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
            self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')

    def get_joint_data(self):
        """
        Get joint angle data（q_actual）
        :return: NumPy array containing the angles (in radians) of the 6 joints
        """
        q_actual = self.rtde_receive.getActualQ()
        return np.array(q_actual)

    def get_cartesian_info(self):
        """
        Get tool end position and pose (tool_vector_actual)
        :return: NumPy array with tool end positions and poses [x, y, z, rx, ry, rz]
        """
        tool_vector_actual = self.rtde_receive.getActualTCPPose()
        return np.array(tool_vector_actual)

    def rpy2rotating_vector(self, rpy):
        # rpy to R
        R = self.rpy2R(rpy)
        # R to rotating_vector
        return self.R2rotating_vector(R)

    def rpy2R(self, rpy):  # [r,p,y] 单位rad
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

    def R2rotating_vector(self, R):
        theta = math.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
        print(f"theta:{theta}")
        rx = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
        ry = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
        rz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
        return np.array([rx, ry, rz]) * theta

    def R2rpy(self, R):
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

    ## robotiq85 gripper
    # get gripper position [0-255]  open:0 ,close:255
    def get_current_tool_pos(self):
        return self.gripper.get_current_position()

    def log_gripper_info(self):
        print(f"Pos: {str(self.gripper.get_current_position())}")

    def close_gripper(self, speed=255, force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(255, speed, force)
        print("gripper had closed!")
        time.sleep(1.2)
        self.log_gripper_info()

    def open_gripper(self, speed=255, force=255):
        # position: int[0-255], speed: int[0-255], force: int[0-255]
        self.gripper.move_and_wait_for_pos(0, speed, force)
        print("gripper had opened!")
        time.sleep(1.2)
        self.log_gripper_info()

    ## get camera data
    def get_camera_data(self):
        color_img, depth_img = self.camera.get_data()
        return color_img, depth_img

    # Note: must be preceded by close_gripper()
    def check_grasp(self):
        # if the robot grasp unsuccessfully ,then the gripper close
        return self.get_current_tool_pos() > 220

    def plane_grasp(self, position, yaw=0, open_size=0.65, k_acc=0.8, k_vel=0.8, speed=255, force=125):
        rpy = [-np.pi, 0, 1.57 - yaw]
        # Determine if the grabbing position is in the workspace
        for i in range(3):
            position[i] = min(max(position[i], self.workspace_limits[i][0]), self.workspace_limits[i][1])
        # Determine if the angle RPY of the grab is within the specified range [-pi,pi]
        for i in range(3):
            if rpy[i] > np.pi:
                rpy[i] -= 2 * np.pi
            elif rpy[i] < -np.pi:
                rpy[i] += 2 * np.pi
        print('Executing: grasp at (%f, %f, %f) by the RPY angle (%f, %f, %f)' \
              % (position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]))

        # pre work
        grasp_home = [0.4, 0, 0.4, -np.pi, 0, 0]  # you can change me
        self.move_j_p(grasp_home, k_acc, k_vel)
        open_pos = int(-258 * open_size + 230)  # open size:0~0.85/140cm --> open pos:230~10
        self.gripper.move_and_wait_for_pos(open_pos, speed, force)
        print("gripper open size:")
        self.log_gripper_info()

        # Firstly, achieve pre-grasp position
        pre_position = copy.deepcopy(position)
        pre_position[2] = pre_position[2] + 0.1  # z axis
        # print(pre_position)
        self.move_j_p(pre_position + rpy, k_acc, k_vel)

        # Second，achieve grasp position
        self.move_l(position + rpy, 0.6 * k_acc, 0.6 * k_vel)
        self.close_gripper(speed, force)
        self.move_l(pre_position + rpy, 0.6 * k_acc, 0.6 * k_vel)
        if (self.check_grasp()):
            print("Check grasp fail! ")
            self.move_j_p(grasp_home)
            return False
        # Third,put the object into box
        box_position = [0.63, 0, 0.25, -np.pi, 0, 0]  # you can change me!
        self.move_j_p(box_position, k_acc, k_vel)
        box_position[2] = 0.1  # down to the 10cm
        self.move_j_p(box_position, k_acc, k_vel)
        self.open_gripper(speed, force)
        box_position[2] = 0.25
        self.move_j_p(box_position, k_acc, k_vel)
        self.move_j_p(grasp_home)
        print("grasp success!")
        return True

    def rtde_close(self):
         self.rtde_control.disconnect()
         self.rtde_receive.disconnect()


if __name__ == "__main__":
    ur_robot = UR_rtde()
    cur_tcp = ur_robot.get_current_tool_pos()
    print(cur_tcp)