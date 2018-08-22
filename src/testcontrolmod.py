#!/usr/bin/env python
# *-* coding:utf-8 *-*

import os
import sys
import time
import math
import thread
import threading
import rospy 
import serial
import cv2
import controlmodulecp as ct

from std_msgs.msg import Int32
from yidamsg.msg import Roi_cam
from sensor_msgs.msg import CompressedImage

cmd_settiltpos = [0xFF, 0x01, 0x00, 0x4D, 0x15, 0x68, 0xCB]

if __name__ == '__main__':
    #初始化相机
    ser_dev1, ipstream_dev1 = ct.init_camera('/dev/ttyUSB0', '192.168.0.222')
    print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    ct.set_tiltpos(ser_dev1, cmd_settiltpos)
#    ser_dev2, ipstream_dev2 = ct.init_camera('/dev/ttyUSB1', '192.168.1.60')
     
    #点号/任务号/相机号/预置位号
#    str_planeinfo = '01/01-1/1/1'
#    ct.get_planeimage(str_planeinfo, ser_dev1, ipstream_dev1)
#
#    #获取检查面
#    msg = Roi_cam()
#    msg.begin_x = [163, 1617, 746]
#    msg.begin_y = [625, 378, 712]
#    msg.end_x = [180, 1855, 1284]
#    msg.end_y = [649, 619, 722]
#    ct.get_singleimage(str_planeinfo, ser_dev1, ipstream_dev1, msg)
#
#    #控制相机 相机号/动作方向/动作步长 
#    #动作方向  1水平左转  2 水平右转  3 垂直向下 4 垂直向上 5 放大 6缩小
#     str_controlinfo = '1/2/1'
#     ct.control_camera(str_controlinfo)
#
#    #设置相机归位
    ct.control_camerareset(1)
#     ct.control_camerareset(2)
     #cmd_setpanpos=[0xFF,0x01,0x00,0x4B,0x00,0x00,0x4c]
     #ct.set_panpos(ser_dev1,cmd_setpanpos)
