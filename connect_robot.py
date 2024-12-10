# --- coding: utf-8 ---
# @Time    : 12/10/24 1:32 PM        # 文件创建时间
# @Author  : htLiang
# @Email   : ryzeliang@163.com
import socket
import struct
import numpy as np



def get_current_tcp():
    IP = '192.168.243.101'
    PORT = 30003
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP, PORT))
    data = s.recv(1116)
    # Parse the actual end coordinate system position at 56-61, the first three bits are x,y,z position information.
    # The last three bits are rx,ry,rz rotation vectors.
    position = struct.unpack('!6d',data[444:492])

    npPosition = np.asarray(position)
    s.close()
    return npPosition

if __name__ == '__main__':
    tcp_cur = get_current_tcp()
    print(tcp_cur)