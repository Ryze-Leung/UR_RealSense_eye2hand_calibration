# --- coding: utf-8 ---
# @Time    : 12/10/24 2:07 PM        # 文件创建时间
# @Author  : htLiang
# @Email   : ryzeliang@163.com
import struct
import numpy as np
import socket
from rtde_control import RTDEControlInterface as UR_control
import rtde_receive





# Example
if __name__ == "__main__":
    receive = rtde_receive.RTDEReceiveInterface('192.168.243.101')
    tcp = receive.getActualTCPPose()
    print(tcp)