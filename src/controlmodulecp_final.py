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

from std_msgs.msg import Int32
from yidamsg.msg import Roi_cam
from sensor_msgs.msg import CompressedImage
import numpy as np

#查询水平坐标 :FF 01 00 51 00 00 52        坐标反馈:  FF 01 00 59 XX YY CS
#设置水平坐标  FF 01 00 4B XX YY CS   0-35999
cmd_getpanpos = [0xFF, 0x01, 0x00, 0x51, 0x00, 0x00, 0x52]
cmd_setpanpos = [0xFF, 0x01, 0x00, 0x4B, 0x00, 0x00, 0x4C]
#查询垂直坐标 :FF 01 00 53 00 00 54        坐标反馈 ： FF 01 00 5B XX YY CS
#设置垂直坐标  FF 01 00 4D XX YY CS   0-35999
cmd_gettiltpos = [0xFF, 0x01, 0x00, 0x53, 0x00, 0x00, 0x54]
cmd_settiltpos = [0xFF, 0x01, 0x00, 0x4D, 0x00, 0x00, 0x4E]
#查询像机ZOOM坐标 :FF 01 00 55 00 00 56    坐标反馈 ： FF 01 00 5D XX YY CS
#设置像机ZOOM值 FF 01 00 4F XX YY CS  0 – 0x4000
cmd_getzoompos = [0xFF, 0x01, 0x00, 0x55, 0x00, 0x00, 0x56]
cmd_setzoompos = [0xFF, 0x01, 0x00, 0x4F, 0x00, 0x00, 0x50]
#转到预置点1 设置预置点1
cmd_gotopreset = [0xFF, 0x01, 0x00, 0x07, 0x00, 0x01, 0x09]
cmd_gotopreset200=[0xFF, 0x01, 0x00, 0x07, 0x00, 0xC8, 0xD1]

cmd_setpreset = [0xFF, 0x01, 0x00, 0x03, 0x00, 0x01, 0x05]

cmd_focus_value=[4.7,9.4,14.1,18.8,23.5,28.2,32.9,37.6,42.3,47,51.7,56.4,61.1,65.8,70.5, \
                    75.2,79.9,84.6,89.3,94]
cmd_zoom_location=[0x00,0x17C8,0x2253,0x289E,0x2CF7,0x3036,0x32DC,0x3502, \
                   0x36DB,0x3875,0x39DB,0x3B0E,0x3C1A,0x3CF4,0x3DB4, \
                   0x3E5A,0x3EDA,0x3F4D,0x3FA6,0x4000]
#200 preset
filenum1=0
filenum2=0
filenum=0
#设置预置位
def set_preset(ser, cmd_setpreset):
    ser.write(cmd_setpreset)
    time.sleep(0.1);
    try:
        data = '';
        n = ser.inWaiting();
        if n:
            data = data + ser.read(n);
            for l in xrange(len(data)):
                pass
                #print '%02X' % ord(data[l])
                #print(ord(data[l]))
    except IOError:
        print "Error!"
#转到预置点
def goto_preset(ser, cmd_gotopreset):
    ser.write(cmd_gotopreset)
    time.sleep(0.1);
    try:
        data = '';
        n = ser.inWaiting();
        if n:
            data = data + ser.read(n);
            for l in xrange(len(data)):
                pass
                #print '%02X' % ord(data[l])
                #print(ord(data[l]))
    except IOError:
        print "Error!"

#查询水平坐标
def get_panpos(ser, cmd_getpanpos):
    ser.write(cmd_getpanpos)
    time.sleep(0.1);
    try:
        data = '';
        n = ser.inWaiting();
        if n:
            data = data + ser.read(n);
            for l in xrange(len(data)):
                pass
                #print '%02X' % ord(data[l])
                #print(ord(data[l]))
    except IOError:
        print "Error!"
    return ord(data[4])*256 + ord(data[5])

#查询垂直坐标
def get_tiltpos(ser, cmd_gettiltpos):
    print(cmd_gettiltpos)
    ser.write(cmd_gettiltpos)
    time.sleep(0.1);
    try:
        data = '';
        n = ser.inWaiting();
        if n:
            data = data + ser.read(n);
	    print len(data)
            for l in xrange(len(data)):
                pass
                print '%02X' % ord(data[l])
                print(ord(data[l]))
    except IOError:
        print "Error!"
    print('##################data',data)
    return ord(data[4])*256 + ord(data[5])

#查询像机ZOOM坐标
def get_zoompos(ser, cmd_getzoompos):
    ser.write(cmd_getzoompos)
    time.sleep(0.1);
    try:
        data = '';
        n = ser.inWaiting();
        if n:
            data = data + ser.read(n);
            for l in xrange(len(data)):
                pass
                #print '%02X' % ord(data[l])
                #print(ord(data[l]))
    except IOError:
        print "Error!"
    return ord(data[4])*256 + ord(data[5])

#设置水平坐标
def set_panpos(ser, cmd_setpanpos):
    ser.write(cmd_setpanpos)
    time.sleep(0.1);
    try:
        data = '';
        n = ser.inWaiting();
        if n:
            data = data + ser.read(n);
            for l in xrange(len(data)):
                pass
                #print '%02X' % ord(data[l])
                #print(ord(data[l]))
    except IOError:
        print "Error!"  

#设置水平向左
def set_panleft(cameracontrol_id, current_ser, current_val_panpos, step_value):
    set_val_panpos = current_val_panpos + step_value
    if set_val_panpos > 35999:
        set_val_panpos -= 35999
    cmd_setpanpos[1] = cameracontrol_id
    cmd_setpanpos[4] = set_val_panpos / 256
    print cmd_setpanpos[4]
    cmd_setpanpos[5] = set_val_panpos % 256
    print cmd_setpanpos[5]
    cmd_setpanpos[6] = get_sumcs(cmd_setpanpos)
    print cmd_setpanpos[6]
    set_panpos(current_ser, cmd_setpanpos)

#设置水平向右
def set_panright(cameracontrol_id, current_ser, current_val_panpos, step_value):
    set_val_panpos = current_val_panpos - step_value
    if set_val_panpos < 0:
        set_val_panpos += 35999
    cmd_setpanpos[1] = cameracontrol_id
    cmd_setpanpos[4] = set_val_panpos / 256
    print cmd_setpanpos[4]
    cmd_setpanpos[5] = set_val_panpos % 256
    print cmd_setpanpos[5]
    cmd_setpanpos[6] = get_sumcs(cmd_setpanpos)
    print cmd_setpanpos[6]
    set_panpos(current_ser, cmd_setpanpos)

#设置垂直坐标
def set_tiltpos(ser, cmd_settiltpos):
    print("........cmd",cmd_settiltpos)
    ser.write(cmd_settiltpos)
    time.sleep(0.1);
    try:
        data = '';
        n = ser.inWaiting();
        if n:
            data = data + ser.read(n);
            for l in xrange(len(data)):
                pass
                #print '%02X' % ord(data[l])
                #print(ord(data[l]))
    except IOError:
        print "Error!"

#设置垂直向上
def set_tiltup(cameracontrol_id, current_ser, current_val_tiltpos, step_value):
    set_val_tiltpos = current_val_tiltpos + step_value
    print('set_val_tiltpos:',set_val_tiltpos)
    if set_val_tiltpos > 10961:
	set_val_tiltpos = set_val_tiltpos - 10961
    cmd_settiltpos[1] = cameracontrol_id
    cmd_settiltpos[4] = set_val_tiltpos / 256
    cmd_settiltpos[5] = set_val_tiltpos % 256
    cmd_settiltpos[6] = get_sumcs(cmd_settiltpos)
    set_tiltpos(current_ser, cmd_settiltpos)

#设置垂直向下
def set_tiltdown(cameracontrol_id, current_ser, current_val_tiltpos, step_value):
    set_val_tiltpos = current_val_tiltpos - step_value
    print('set_val_tiltpos',set_val_tiltpos)
    print('current_val_tiltpos',current_val_tiltpos)
    if set_val_tiltpos < 0:
	set_val_tiltpos = set_val_tiltpos + 10961 
    cmd_settiltpos[1] = cameracontrol_id
    cmd_settiltpos[4] = set_val_tiltpos / 256
    cmd_settiltpos[5] = set_val_tiltpos % 256
    cmd_settiltpos[6] = get_sumcs(cmd_settiltpos)
    set_tiltpos(current_ser, cmd_settiltpos)

#设置像机ZOOM坐标
def set_zoompos(ser, cmd_setzoompos):
    ser.write(cmd_setzoompos)
    time.sleep(0.1);
    try:
        data = '';
        n = ser.inWaiting();
        if n:
            data = data + ser.read(n);
            for l in xrange(len(data)):
                pass
                #print '%02X' % ord(data[l])
                #print(ord(data[l]))
    except IOError:
        print "Error!"

#设置放大
def set_zoominc(cameracontrol_id, current_ser, current_val_zoom, step_value):
    set_val_zoompos = current_val_zoom + step_value
    if set_val_zoompos >16384:
	set_val_zoompos=16384
    print('set_val_zoompos',set_val_zoompos)
    cmd_setzoompos[1] = cameracontrol_id
    cmd_setzoompos[4] = set_val_zoompos / 256
    cmd_setzoompos[5] = set_val_zoompos % 256
    cmd_setzoompos[6] = get_sumcs(cmd_setzoompos)
    set_zoompos(current_ser, cmd_setzoompos)

#设置缩小
def set_zoomdec(cameracontrol_id, current_ser, current_val_zoom, step_value):
    set_val_zoompos = current_val_zoom - step_value
    if set_val_zoompos < 0:
	set_val_zoompos = 0
    print('set_val_zoompos',set_val_zoompos)
    cmd_setzoompos[1] = cameracontrol_id
    cmd_setzoompos[4] = set_val_zoompos / 256
    cmd_setzoompos[5] = set_val_zoompos % 256
    cmd_setzoompos[6] = get_sumcs(cmd_setzoompos)
    set_zoompos(current_ser, cmd_setzoompos)

def get_sumcs(cmdin):
    result = (cmdin[1] + cmdin[2] + cmdin[3] + cmdin[4] + cmdin[5]) % 256;
    return result

def init_usbdev(dev, baudrate):
    try:
        ser_dev = serial.Serial(dev, baudrate)
        return ser_dev
    except serial.serialutil.SerialException as e:
        print('异常: 设备%s串口未连接，请检查！' % dev)
    finally:
        pass

def init_stream(ipadress):
     str_videocapture = 'rtspsrc location=rtsp://admin:123qweasd@' + str(ipadress) + ':554/h264/ch1/main/av_stream latency=0 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! videoconvert ! appsink'
     capture_stream = cv2.VideoCapture(str_videocapture)
     return capture_stream

def init_camera(dev, ipaddress):
     ser_dev = init_usbdev(dev, 2400)
     ipstream_dev = init_stream(ipaddress)
     return ser_dev, ipstream_dev
#获取开始保存图像的信号
def visitDir(path):
    if not os.path.isdir(path):
        print('Error: "', path, '" is not a directory or does not exist.')
    else:
	global filenum
	filenum=0
        try:
            for lists in os.listdir(path):
                sub_path = os.path.join(path, lists)
                filenum += 1
                print('No.', filenum, ' ', sub_path)
                if os.path.isdir(sub_path):
                    visitDir(sub_path)
	    return filenum
        except:
            pass

def get_planeimage(str_planeinfo, ser_dev,ipstream_dev):
    param_planeinfo = []
    param_planeinfo = str_planeinfo.split('/')

    cmd_gotopreset[1] = int(param_planeinfo[2])
    cmd_gotopreset[5] = int(param_planeinfo[3])
    cmd_gotopreset[6] = get_sumcs(cmd_gotopreset)
    goto_preset(ser_dev, cmd_gotopreset)
    time.sleep(5)

    if ipstream_dev.isOpened():
        _, frame_stream = ipstream_dev.read()
        return frame_stream
    return -1


def get_singleimage(str_planeinfo, ser_dev,ipstream_dev, data, single_image,singleid):
    
    print('#########get_singleimage#########')
    print str_planeinfo
    param_planeinfo = []
    param_planeinfo = str_planeinfo.split('/')
    camera_id_singleimage = int(param_planeinfo[2])
    camera_preset_singleimage = int(param_planeinfo[3])

    cmd_getzoompos[1] = camera_id_singleimage
    cmd_getzoompos[6] = get_sumcs(cmd_getzoompos)
    cmd_getpanpos[1] = camera_id_singleimage
    cmd_getpanpos[6] = get_sumcs(cmd_getpanpos)
    cmd_gettiltpos[1] = camera_id_singleimage
    cmd_gettiltpos[6] = get_sumcs(cmd_gettiltpos)
    cmd_setzoompos[1] = camera_id_singleimage
    cmd_setzoompos[6] = get_sumcs(cmd_setzoompos)
    cmd_setpanpos[1] = camera_id_singleimage
    cmd_setpanpos[6] = get_sumcs(cmd_setpanpos)
    cmd_settiltpos[1] = camera_id_singleimage
    cmd_settiltpos[6] = get_sumcs(cmd_settiltpos)
    cmd_gotopreset[1] = camera_id_singleimage
    cmd_gotopreset[5] = camera_preset_singleimage
    cmd_gotopreset[6] = get_sumcs(cmd_gotopreset)

    
    framecentral_x = []
    framecentral_y = []
    captureimage_width = 1920
    captureimage_height = 1080

    print("curlocation:",cmd_getpanpos)
    #获取设备检测框坐标，并计算出中心坐标，保存
    for i in range(len(data.begin_x)):
        framecentral_x.append(data.begin_x[i] + (data.end_x[i] - data.begin_x[i])/2)
        framecentral_y.append(data.begin_y[i] + (data.end_y[i] - data.begin_y[i])/2)
        print "framecentral_x: " + str(framecentral_x), "framecentral_y: " + str(framecentral_y)
    #根据中心坐标，计算云台、焦距参数
    for i in range(len(framecentral_x)):
        #(与中心坐标的)偏移量   
        offset_x = framecentral_x[i] - (captureimage_width / 2)  #960
        offset_y = framecentral_y[i] - (captureimage_height / 2) #540
        print "offset_x: " + str(offset_x), "offset_y: " + str(offset_y)
        #焦距处理(4.7-94,20倍) 2.6215, 131.38(0x4000/(94-4.7))
        #默认为1倍焦距
        ori_focus = get_zoompos(ser_dev, cmd_getzoompos)
        print "ori_focus: " + str(ori_focus)
        real_focus = 4.7
        focus_set = real_focus / 2.6215 * 1000

        #水平、垂直偏移量(换算成云台对应的度制)
        horizontal_x =  ((math.atan(offset_x / focus_set) / math.pi) * 180)
        vertical_y   = -((math.atan(  offset_y / focus_set) / math.pi) * 180)
        print "horizontal_x: " + str(horizontal_x), "vertical_y: " + str(vertical_y)
        if framecentral_x[i] > 960:
            offsetx_value = (horizontal_x - 1) * 100
        elif framecentral_x[i] == 960:
            offsetx_value = horizontal_x * 100
        elif framecentral_x[i] < 960:
            offsetx_value = (horizontal_x + 1) * 100
        else:
            pass
        offsety_value = (vertical_y - 1) * 100
        print "offsetx_value: " + str(int(offsetx_value)), "offsety_value: " + str(int(offsety_value))
        val_panpos = get_panpos(ser_dev, cmd_getpanpos)
        time.sleep(1.0)
        print('#######serdev',ser_dev)
        val_tiltpos = get_tiltpos(ser_dev, cmd_gettiltpos)
        print "val_panpos: " + str(val_panpos), "val_tiltpos: " + str(val_tiltpos)
        finalx_set = val_panpos + offsetx_value
        finaly_set = val_tiltpos + offsety_value
        print "finalx_set: " + str(int(finalx_set)), "finaly_set: " + str(int(finaly_set))
        if finalx_set > 35999:
            finalx_set = finalx_set -35999
        elif finalx_set < 0:
            finalx_set = finalx_set + 35999
        if finaly_set > 10961:
            finaly_set = finalx_set - 10961
        elif finaly_set < 0:
            finaly_set = finaly_set + 10961
        #最终焦距参数(根据设备检测框的高度计算)(0x4000/20)
        #if 目标宽>目标高  放大倍数根据宽决定
	#if 目标宽<目标高  放大倍数根据高决定
	frame_height = data.end_y[i] - data.begin_y[i];
	frame_width = data.end_x[i] - data.begin_y[i];
	if frame_height >= frame_width:
        	lagerfacter=int(math.floor((captureimage_height/2)/abs(frame_height)))
       		final_foucs=cmd_zoom_location[lagerfacter]
        else:
		lagerfacter=int(math.floor((captureimage_width/2)/abs(frame_width)))
                final_foucs=cmd_zoom_location[lagerfacter]
	#返回放大倍数
       
        if final_foucs > 16380:
            final_foucs = 16380
        print "final_foucs: " + str(int(final_foucs))
        
        cmd_setpanpos[4] = int(finalx_set) / 256
        cmd_setpanpos[5] = int(finalx_set) % 256
        cmd_setpanpos[6] = get_sumcs(cmd_setpanpos)
        cmd_settiltpos[4] = int(finaly_set) / 256
        cmd_settiltpos[5] = int(finaly_set) % 256
        cmd_settiltpos[6] = get_sumcs(cmd_settiltpos)
        cmd_setzoompos[4] = int(final_foucs) / 256
        cmd_setzoompos[5] = int(final_foucs) % 256
        cmd_setzoompos[6] = get_sumcs(cmd_setzoompos)
        print cmd_setpanpos
        print cmd_settiltpos
        print cmd_setzoompos

        #设置水平垂直
        print('cmd_settiltpos',cmd_settiltpos)
        set_panpos(ser_dev, cmd_setpanpos)
        set_tiltpos(ser_dev, cmd_settiltpos)
        set_zoompos(ser_dev, cmd_setzoompos)

        time.sleep(8)
        if ipstream_dev.isOpened():
        	_,frame_stream = ipstream_dev.read()
		#frame_stream_list.append(frame_stream)
                filenum=0
                filenum1=visitDir('/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/output')
                print('Total Permission Files: ', filenum1)
        #返回预置点
        	goto_preset(ser_dev, cmd_gotopreset)
        	time.sleep(5)
	#return locate ptz state
    return frame_stream,finalx_set,finaly_set,lagerfacter,filenum1


def get_deviceimage(str_planeinfo, ser_dev,ipstream_dev, data, single_image,singleid,prex_set,prey_set,lagerfacter):
    print('#########get_deviceimage#########')
    print str_planeinfo
    param_planeinfo = []
    param_planeinfo = str_planeinfo.split('/')
    camera_id_singleimage = int(param_planeinfo[2])
    camera_preset_singleimage = int(param_planeinfo[3])

    cmd_getzoompos[1] = camera_id_singleimage
    cmd_getzoompos[6] = get_sumcs(cmd_getzoompos)
    cmd_getpanpos[1] = camera_id_singleimage
    cmd_getpanpos[6] = get_sumcs(cmd_getpanpos)
    cmd_gettiltpos[1] = camera_id_singleimage
    cmd_gettiltpos[6] = get_sumcs(cmd_gettiltpos)
    cmd_setzoompos[1] = camera_id_singleimage
    cmd_setzoompos[6] = get_sumcs(cmd_setzoompos)
    cmd_setpanpos[1] = camera_id_singleimage
    cmd_setpanpos[6] = get_sumcs(cmd_setpanpos)
    cmd_settiltpos[1] = camera_id_singleimage
    cmd_settiltpos[6] = get_sumcs(cmd_settiltpos)
    cmd_gotopreset[1] = camera_id_singleimage
    cmd_gotopreset[5] = camera_preset_singleimage
    cmd_gotopreset[6] = get_sumcs(cmd_gotopreset)

    framecentral_x = []
    framecentral_y = []
    captureimage_width = 1920
    captureimage_height = 1080

    print("curlocation:",cmd_getpanpos)
    #获取设备检测框坐标，并计算出中心坐标，保存
    for i in range(len(data.begin_x)):
        framecentral_x.append(data.begin_x[i] + (data.end_x[i] - data.begin_x[i])/2)
        framecentral_y.append(data.begin_y[i] + (data.end_y[i] - data.begin_y[i])/2)
        print "framecentral_x: " + str(framecentral_x), "framecentral_y: " + str(framecentral_y)
    #根据中心坐标，计算云台、焦距参数
    for i in range(len(framecentral_x)):
        #(与中心坐标的)偏移量   
        offset_x = framecentral_x[i] - (captureimage_width / 2)  #960
        offset_y = framecentral_y[i] - (captureimage_height / 2) #540
        print "offset_x: " + str(offset_x), "offset_y: " + str(offset_y)
        #焦距处理(4.7-94,20倍) 2.6215, 131.38(0x4000/(94-4.7))
        #默认为1倍焦距
        ori_focus = cmd_focus_value[lagerfacter]
        print "ori_focus: " + str(ori_focus)
        #real_focus = (ori_focus/819.13)*4.65 + 4.7
        focus_set = ori_focus / 2.6215 * 1000

        #水平、垂直偏移量(换算成云台对应的度制)
        horizontal_x =  ((math.atan(offset_x / focus_set) / math.pi) * 180)
        vertical_y   = -((math.atan(  offset_y / focus_set) / math.pi) * 180)
        print "horizontal_x: " + str(horizontal_x), "vertical_y: " + str(vertical_y)
        if framecentral_x[i] > 960:
            offsetx_value = (horizontal_x - 1) * 100
        elif framecentral_x[i] == 960:
            offsetx_value = horizontal_x * 100
        elif framecentral_x[i] < 960:
            offsetx_value = (horizontal_x + 1) * 100
        else:
            pass
        if framecentral_y[i] > 540:
            offsety_value = (vertical_y + 1) * 100
        elif framecentral_y[i] == 540:
            offsety_value = vertical_y * 100
        elif framecentral_y[i] < 540:
            offsety_value = (vertical_y - 1) * 100
        else:
            pass
        offsety_value = (vertical_y - 1) * 100
        print "offsetx_value: " + str(int(offsetx_value)), "offsety_value: " + str(int(offsety_value))
        val_panpos = prex_set
        time.sleep(1.0)
        print('#######serdev',ser_dev)
        val_tiltpos = prey_set
        print "val_panpos: " + str(val_panpos), "val_tiltpos: " + str(val_tiltpos)
        finalx_set = val_panpos + offsetx_value
        finaly_set = val_tiltpos + offsety_value
        print "finalx_set: " + str(int(finalx_set)), "finaly_set: " + str(int(finaly_set))
        if finalx_set > 35999:
            finalx_set = finalx_set -35999
        elif finalx_set < 0:
            finalx_set = finalx_set + 35999
        if finaly_set > 10961:
            finaly_set = finalx_set - 10961
        elif finaly_set < 0:
            finaly_set = finaly_set + 10961
        #最终焦距参数(根据设备检测框的高度计算)(0x4000/20)
        frame_height = data.end_y[i] - data.begin_y[i];
	frame_width = data.end_x[i] - data.begin_x[i];
        #final_foucs = (captureimage_height / 2) * (ori_focus / 819.13 + 1) / (abs(frame_height)) * 4914;
        #final_foucs= cmd_zoom_location[int(math.floor((captureimage_height/2)/abs(frame_height)))]
        #放大因子
        if frame_height>=frame_width:
		m_lager=int(math.floor((captureimage_height/2)/abs(frame_height)))+lagerfacter
       		print('@@@@@@Device LagerFacter:',m_lager)
       		if m_lager>19:
            		m_lager=19
        	final_foucs=cmd_zoom_location[m_lager]
        else:
		m_lager=int(math.floor((captureimage_width/2)/abs(frame_width)))+lagerfacter
                print('@@@@@@Device LagerFacter:',m_lager)
                if m_lager>19:
                        m_lager=19
                final_foucs=cmd_zoom_location[m_lager]
        
        cmd_setpanpos[4] = int(finalx_set) / 256
        cmd_setpanpos[5] = int(finalx_set) % 256
        cmd_setpanpos[6] = get_sumcs(cmd_setpanpos)
        cmd_settiltpos[4] = int(finaly_set) / 256
        cmd_settiltpos[5] = int(finaly_set) % 256
        cmd_settiltpos[6] = get_sumcs(cmd_settiltpos)
        cmd_setzoompos[4] = int(final_foucs) / 256
        cmd_setzoompos[5] = int(final_foucs) % 256
        cmd_setzoompos[6] = get_sumcs(cmd_setzoompos)
        print cmd_setpanpos
        print cmd_settiltpos
        print cmd_setzoompos

        #设置水平垂直
        print('cmd_settiltpos',cmd_settiltpos)
        set_panpos(ser_dev, cmd_setpanpos)
        set_tiltpos(ser_dev, cmd_settiltpos)
        set_zoompos(ser_dev, cmd_setzoompos)

        time.sleep(8)
        if ipstream_dev.isOpened():
        	_,frame_stream = ipstream_dev.read()
		#frame_stream_list.append(frame_stream)
                filenum2=0
                filenum2=visitDir('/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/output')
                print('Total Permission Files: ', filenum2)
        #返回预置点
        	goto_preset(ser_dev, cmd_gotopreset)	
        #返回预置点
        	#frame_stream=cv2.imread("/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/Device_image/1074-1-1-white_watch.jpg")
		goto_preset(ser_dev, cmd_gotopreset)
        	time.sleep(5)
	#return locate ptz state
    return frame_stream,filenum2


def hmset(current_dev,str_planeinfo):
	
	cameraID=int(str_planeinfo.split('/')[0])
	HValue=int(str_planeinfo.split('/')[1])
	VValue=int(str_planeinfo.split('/')[2])
	FValue=int(str_planeinfo.split('/')[3])
	print('@@@@@@@@@@@@@@@23',cameraID,HValue,VValue,FValue)
	cmd_setpanpos[1] = int(cameraID)        
	cmd_setpanpos[4] = int(HValue*100) / 256
        cmd_setpanpos[5] = int(HValue*100) % 256
        cmd_setpanpos[6] = get_sumcs(cmd_setpanpos)
	cmd_settiltpos[1] = int(cameraID)
        cmd_settiltpos[4] = int(VValue*100) / 256
        cmd_settiltpos[5] = int(VValue*100) % 256
        cmd_settiltpos[6] = get_sumcs(cmd_settiltpos)
	cmd_setzoompos[1] = int(cameraID)
        cmd_setzoompos[4] = int(FValue*819) / 256
        cmd_setzoompos[5] = int(FValue*819) % 256
        cmd_setzoompos[6] = get_sumcs(cmd_setzoompos)
        print cmd_setpanpos
        print cmd_settiltpos
        print cmd_setzoompos
	set_panpos(current_dev,cmd_setpanpos)
	set_tiltpos(current_dev,cmd_settiltpos)
	set_zoompos(current_dev,cmd_setzoompos)
	

def control_camera_id(cameracontrol_id, current_ser, cameracontrol_type, cameracontrol_step, \
    current_val_panpos, current_val_tiltpos, current_val_zoom):
    #1水平左转
    print('@@@@@@@@@@@@@22222222222222222222')
    print('cameracontrol_type',cameracontrol_type)
    print('cameracontrol_step',cameracontrol_step)
    if 1 == cameracontrol_type:
        if 1 == cameracontrol_step:
            step_value = 500
            set_panleft(cameracontrol_id, current_ser, current_val_panpos, step_value)
        elif 2 == cameracontrol_step:
            step_value = 1000
            set_panleft(cameracontrol_id, current_ser, current_val_panpos, step_value)
            print('@@@@@@@@@@@@@3333333333333333333')
        elif 3 == cameracontrol_step:
            step_value = 1500
            set_panleft(cameracontrol_id, current_ser, current_val_panpos, step_value)
        else:
            pass
    #2水平右转
    elif 2 == cameracontrol_type:
	print('@@@@@@@@@@@@@22222222222222222222')
        if 1 == cameracontrol_step:
            step_value = 500
            set_panright(cameracontrol_id, current_ser, current_val_panpos, step_value)
        elif 2 == cameracontrol_step:
            step_value = 1000
            set_panright(cameracontrol_id, current_ser, current_val_panpos, step_value)
        elif 3 == cameracontrol_step:
            step_value = 1500
            set_panright(cameracontrol_id, current_ser, current_val_panpos, step_value)
        else:
            pass
    #3垂直向下
    elif 3 == cameracontrol_type:
        if 1 == cameracontrol_step:
            step_value = 500
            set_tiltdown(cameracontrol_id, current_ser, current_val_tiltpos, step_value)
        elif 2 == cameracontrol_step:
            step_value = 1000
            set_tiltdown(cameracontrol_id, current_ser, current_val_tiltpos, step_value)
        elif 3 == cameracontrol_step:
            step_value = 1500
            set_tiltdown(cameracontrol_id, current_ser, current_val_tiltpos, step_value)
        else:
            pass
    #4垂直向上
    elif 4 == cameracontrol_type:
        if 1 == cameracontrol_step:
            step_value = 500
            set_tiltup(cameracontrol_id, current_ser, current_val_tiltpos, step_value)
        elif 2 == cameracontrol_step:
            step_value = 1000
            set_tiltup(cameracontrol_id, current_ser, current_val_tiltpos, step_value)
        elif 3 == cameracontrol_step:
            step_value = 1500
            set_tiltup(cameracontrol_id, current_ser, current_val_tiltpos, step_value)
        else:
            pass
    #5放大
    elif 5 == cameracontrol_type:
        if 1 == cameracontrol_step:
            step_value = 819
            set_zoominc(cameracontrol_id, current_ser, current_val_zoom, step_value)
        elif 2 == cameracontrol_step:
            step_value = 1638
            set_zoominc(cameracontrol_id, current_ser, current_val_zoom, step_value)
        elif 3 == cameracontrol_step:
            step_value = 2457
            set_zoominc(cameracontrol_id, current_ser, current_val_zoom, step_value)
        else:
            pass
    #6缩小
    elif 6 == cameracontrol_type:
        if 1 == cameracontrol_step:
            step_value = 819
            set_zoomdec(cameracontrol_id, current_ser, current_val_zoom, step_value)
        elif 2 == cameracontrol_step:
            step_value = 1638
            set_zoomdec(cameracontrol_id, current_ser, current_val_zoom, step_value)
        elif 3 == cameracontrol_step:
            step_value = 2457
            set_zoomdec(cameracontrol_id, current_ser, current_val_zoom, step_value)
        else:
            pass
    else:
        pass

def control_camera(str_controlinfo):
    #控制相机 相机号/动作方向/动作步长 
    #动作方向  1水平左转  2 水平右转  3 垂直向下 4 垂直向上 5 放大 6缩小
    param_controlinfo = []
    param_controlinfo = str_controlinfo.split('/')

    print param_controlinfo
    cameracontrol_id = int(param_controlinfo[0])
    print cameracontrol_id
    cameracontrol_type = int(param_controlinfo[1])
    print cameracontrol_type
    cameracontrol_step = int(param_controlinfo[2])
    print cameracontrol_step

    cmd_getpanpos1 = [0xFF, 0x01, 0x00, 0x51, 0x00, 0x00, 0x52]
    cmd_gettiltpos1 = [0xFF, 0x01, 0x00, 0x53, 0x00, 0x00, 0x54]
    cmd_getzoompos1 = [0xFF, 0x01, 0x00, 0x55, 0x00, 0x00, 0x56]
    cmd_getpanpos2 = [0xFF, 0x02, 0x00, 0x51, 0x00, 0x00, 0x53]
    cmd_gettiltpos2 = [0xFF, 0x02, 0x00, 0x53, 0x00, 0x00, 0x55]
    cmd_getzoompos2 = [0xFF, 0x02, 0x00, 0x55, 0x00, 0x00, 0x57]

    if 1 == cameracontrol_id:
        #获取当前值
	print('@@@@@@@@@@@@111111111111111')
        current_ser = init_usbdev('/dev/ttyUSB0', 2400)
        current_val_panpos = get_panpos(current_ser, cmd_getpanpos1)
        current_val_tiltpos = get_tiltpos(current_ser, cmd_gettiltpos1)
        current_val_zoom = get_zoompos(current_ser, cmd_getzoompos1)
	print('@@@@@@@@@@@@111111111111111')
        control_camera_id(cameracontrol_id, current_ser, cameracontrol_type, cameracontrol_step, \
            current_val_panpos, current_val_tiltpos, current_val_zoom)
    elif 2 == cameracontrol_id:
        #获取当前值
        current_ser = init_usbdev('/dev/ttyUSB1', 2400)
        current_val_panpos = get_panpos(current_ser, cmd_getpanpos2)
        current_val_tiltpos = get_tiltpos(current_ser, cmd_gettiltpos2)
        current_val_zoom = get_zoompos(current_ser, cmd_getzoompos2)
        control_camera_id(cameracontrol_id, current_ser, cameracontrol_type, cameracontrol_step,  \
            current_val_panpos, current_val_tiltpos, current_val_zoom)
    else:
        pass

def control_camerareset(camera_id_reset):
    if 1 == camera_id_reset:
        ser = init_usbdev('/dev/ttyUSB0', 2400)
    elif 2 == camera_id_reset:
        ser = init_usbdev('/dev/ttyUSB1', 2400)
    else:
        pass

    cmd_setpanpos1 = [0xFF, 0x01, 0x00, 0x4B, 0x00, 0x00, 0x4C]
    cmd_settiltpos1 = [0xFF, 0x01, 0x00, 0x4D, 0x00, 0x00, 0x4E]
    cmd_setzoompos1 = [0xFF, 0x01, 0x00, 0x4F, 0x00, 0x00, 0x50]
    cmd_setpanpos2 = [0xFF, 0x02, 0x00, 0x4B, 0x00, 0x00, 0x4D]
    cmd_settiltpos2 = [0xFF, 0x02, 0x00, 0x4D, 0x00, 0x00, 0x4F]
    cmd_setzoompos2 = [0xFF, 0x02, 0x00, 0x4F, 0x00, 0x00, 0x51]

    if 1 == camera_id_reset:
        set_panpos(ser, cmd_setpanpos1)
        set_tiltpos(ser, cmd_settiltpos1)
        set_zoompos(ser, cmd_setzoompos1)
    elif 2 == camera_id_reset:
        set_panpos(ser, cmd_setpanpos2)
        set_tiltpos(ser, cmd_settiltpos2)
        set_zoompos(ser, cmd_setzoompos2)
    else:
        pass

if __name__ == '__main__':
    pass
    '''
    #初始化相机
    init_camera()

    #点号/任务号/相机号/预置位号
    str_planeinfo = '01/01-1/1/1'
    get_planeimage(str_planeinfo)

    #获取检查面
    msg = Roi_cam()
    msg.begin_x = [163, 1617, 746]
    msg.begin_y = [625, 378, 712]
    msg.end_x = [180, 1855, 1284]
    msg.end_y = [649, 619, 722]
    get_singleimage(msg)

    #控制相机 相机号/动作方向/动作步长 
    #动作方向  1水平左转  2 水平右转  3 垂直向下 4 垂直向上 5 放大 6缩小
    str_controlinfo = '1/2/1'
    control_camera(str_controlinfo)

    #设置相机归位
    control_camerareset(1)
    control_camerareset(2)
    '''
