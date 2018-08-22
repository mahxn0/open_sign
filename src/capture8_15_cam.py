#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface')

import time, rospy, cv2, sys
import numpy as np
import thread, datetime, math,threading
import scipy.misc, scipy.io, os
import sqlite3
#import queue as Queue
import Queue
#import requests

from sensor_msgs.msg import CompressedImage
from yidamsg.msg import LiveImage
#from yidamsg.msg import CaptureImage
from std_msgs.msg import String
from std_msgs.msg import Int64
from sensor_msgs.msg import Image
from math import *
from yidamsg.msg import Log
from yidamsg.msg import Roi_cam
from yidamsg.msg import InspectedResult
from yidamsg.msg import transfer
import controlmodulecp as ct

preimg = None
preimgFlag = False

detect_recog = None
isReceivedNormal = False
deal_ptz = False

captureFlag = False
preReceivedTime = 0.0
connecFailedFlag = False
registeredFlag = False
menuflag=0
content=''
count_capt = time.strftime("%H-%M-%S")
date_capt = time.strftime('%Y-%m-%d',time.localtime(time.time()))
database = None
startTimeFlag = None
timeId = None
minId = None 
posId = None
taskId = None
camId = None
picId = 0
subMultipId=0
subSingleId=0
ResultLable=''
ipstream_dev1=None
submsg = ''
startflag=False

MulboardPath='/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/Multi_image/'
SinboardPath='/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/Single_image/'
DevicePath='/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/Device_image/'
RecordPath='/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/output/'
#from ssd_surface import ssd_detect_surface
#from ssd_surface import ssd_detect_cut

import yolo_surface_cut
import board_device_detect

device_surface=None
yolo_surface = None

#初始化yolo surface=封装好的类的输出
def init_yolo_surface():
    global yolo_surface
    global device_surface
    yolo_surface = yolo_surface_cut.yolo_surface()
    device_surface=board_device_detect.yolo_surface()

def detect_surface(img,isCrop=True):
    global yolo_surface
    print('img:',img.shape)
    isSurface,surfacePic = yolo_surface.detect(img,isCrop)
    return isSurface,surfacePic

def fun_timer():
    global preReceivedTime
    curTime = time.time()
    global registeredFlag
    if curTime - preReceivedTime > 2.0 and registeredFlag:
        global connecFailedFlag
        connecFailedFlag = True
    global timer1
    timer1 = threading.Timer(3, fun_timer)  # 1秒调用一次函数
    timer1.start()

def getXY(i):
    return int(i)

def printnum(x):
	
    print("@@@@@@@:",int(x))

def printstr(x):

    print("@@@@@@:",str(x))

def ConnectionDetectTimer():
    curTime = time.time()
    connectionDetect_pub.publish(int(curTime))
    #print(int(curTime))
    global timer2
    timer2 = threading.Timer(1, ConnectionDetectTimer)  # 1秒调用一次函数
    timer2.start()

  
def imageRotate(img, degree):

    height,width=img.shape[:2]
    heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
    widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))

    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)

    matRotation[0,2] +=(widthNew-width)/2
    matRotation[1,2] +=(heightNew-height)/2

    imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))
    return imgRotation



def callback_CaptureImage1(data): #preImage process
    print(data.header.frame_id)
    global capture_data
    #capture_data = data
    #print(data)
    rcvData = {}
    np_arr_front = np.fromstring(data.data, np.uint8)
    captureimg = cv2.imdecode(np_arr_front, cv2.IMREAD_COLOR)
    captureimg = imageRotate(captureimg,90)
    #print(data.header.frame_id)
    content = data.header.frame_id
    #print(data.header.frame_id)
    frame_id = content.split('/')[0]
    if frame_id == '1': # many surface
        global captureFlag
        captureFlag = True
        print('received many surface image')
        #global count_capt
        #global date_capt
        global startTimeFlag
        startTimeFlag = content.split('/')[1]
        startTimeFlag = startTimeFlag[:4] + startTimeFlag[5:7] + startTimeFlag[8:10] + \
                        startTimeFlag[11:13] + startTimeFlag[14:16] + startTimeFlag[17:]
        print('startTimeFlag',startTimeFlag)
        #global posId
        posId = content.split('/')[2]
        #global taskId
        taskId = content.split('/')[3] #011
        #global camId
        camId = content.split('/')[-1]
        print('########3',posId,taskId,camId)

        rcvData['img'] = captureimg
        rcvData['startTimeFlag'] = startTimeFlag
        rcvData['posId'] = posId
        rcvData['taskId'] = taskId
        rcvData['camId'] = camId

        global captureQueue
        captureQueue.put(rcvData)


    # print('received capture image')
    #cv2.imshow("im_temp", captureimg)
    #cv2.waitKey(1)
    # 按照当前时间保存高清图
    # captureimg1 = imageRotate(captureimg, 90)
    # global count_capt
    # global date_capt
    saveTime = time.time()
    cv2.imwrite('/home/zdyd/workspace/src/detectAndRecog/src/capture-multiboard/'+str(saveTime)+'.jpg', captureimg)
    print('received capture img')


def callback_CaptureImageFlag(data):
    if data.data == 'robotstop':
        global captureFlag
        captureFlag = True
        print('received robotstop')


def callback_monitorOnLine(data):
    print('monitor is online',data)
    if data.data == 'monitor_is_online':
        global registeredFlag
        if registeredFlag is False:
            registeredFlag = True
        elif registeredFlag:
            global connecFailedFlag
            if connecFailedFlag is False:
                conn = sqlite3.connect('/home/zdyd/workspace/src/detectAndRecog/src/config/test.db')
                db = conn.cursor()
                db.execute('select * from watch')
                res = db.fetchall()
                if len(res) != 0:
                    for line in res:
                        # print(res)
                        if line[14] != 1:  # published

                            msg = InspectedResult()
                            print(line[8])
                            print(line[9])
                            watch = cv2.imread(line[8])
                            sign = cv2.imread(line[9])
                            msg.equipimage = np.array(cv2.imencode('.jpg', watch)[1]).tostring()
                            msg.nameplates = np.array(cv2.imencode('.jpg', sign)[1]).tostring()
                            msg.result = str(line[10])
                            if line[10] != '':
                                msg.success = 1
                            else:
                                msg.success = 0
                            msg.picid = int(line[0])
                            msg.equipid = line[0]
                            msg.camid = 1
                            inspectedresult_pub.publish(msg)

def callback_connectMonitor(data):
    #if data.data == :
    global preReceivedTime
    preReceivedTime = time.time()  
    detect_heart_break.publish(data)

#解析接收到的消息
def callback_transferPub(data):
    global startTimeFlag
    global menuflag
    global content
    print(data)
    content = data.data
    menuflag = data.flag
#    content='2018-05-04 18:00:00/01/01/01-1/1/1'
#    menuflag=0
    if menuflag==0:
        startTimeFlag = content.split('/')[0]
        #时间戳sss
        startTimeFlag = startTimeFlag[:4] + startTimeFlag[5:7] + startTimeFlag[8:10] + \
        		    startTimeFlag[11:13] + startTimeFlag[14:16] + startTimeFlag[17:]
        print('startTimeFlag',startTimeFlag)
        #时间戳
        #点号
        posId = content.split('/')[1]
        taskId = content.split('/')[2]
        #相机号
        camId = content.split('/')[3] #011
        #预置位
        camposId = content.split('/')[4]
        print("camposId=:",camposId)
        print('into-----------------------tranpub')
        rcvData['startTimeFlag'] = startTimeFlag
        rcvData['posId'] = posId
        rcvData['taskId'] = taskId
        rcvData['camId'] = camId
        rcvData['camposId'] = camposId
        global captureQueue
        captureQueue.put(rcvData)
        
        #发布话题时的topic msg
        if startTimeFlag is None and camposId is None:
            print('SubMsg Error')
        else:
            global submsg
            global startflag
            startflag=True
            submsg=str(posId)+'/'+str(taskId)+'/'+str(camId)+'/'+str(camposId)
            print('########',startTimeFlag,taskId,camId,posId,camposId)
#    if menuflag=1:
#	#相机号/动作方向/动作步长
#	#动作方向  1水平左转  2 水平右转  3 垂直向下 4 垂直向上 5 zoomin 6 zoomout
#	submsg=''
#	submsg=data.data
#    if menuflag=2:
#	submsg=''
#	submsg=data.data
#    if menuflag=3:
#	submsg=data.data
def sub_captureImage():
    rospy.Subscriber("captureimage",CompressedImage, callback_CaptureImage1)
    rospy.spin()

def sub_captureImageFlag():
    rospy.Subscriber("capture_command",String,callback_CaptureImageFlag)
    rospy.spin()

def sub_connectSuccess():
	rospy.Subscriber('monitorOnLine',String,callback_monitorOnLine)
	rospy.spin

def sub_connectMonitor():
    rospy.Subscriber('connectionDetect_monitor',Int64,callback_connectMonitor)
    rospy.spin

def sub_info():
    print('into----------sub_info--------')
    rospy.Subscriber('transfer_pub',transfer,callback_transferPub)
    rospy.spin
def pub_record():
    while True:
        try:
            global startflag
            if startflag is True:
                global submsg
                if submsg != '':
    	            #image000 = cv2.imread('/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/output.jpg')
		    global ipstream_dev1
		    if ipstream_dev1.isOpened():
        		_, img = ipstream_dev1.read()   		    
		    	global picId
    		    	cv2.imwrite(RecordPath+str(picId)+'.jpg',img)
                    	msg = CompressedImage()                                                                                                      
                    	msg.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()                                                         
                    	msg.header.frame_id = submsg + '/'+str(picId)+'/'+str(subMultipId)+'-'+str(subSingleId)+'-'+ResultLable
                    	record_pub.publish(msg)
                    	picId = int(picId)+1
	    
		
        except:
            pass

if __name__=='__main__':
    #初始化相机
    ser_dev1, ipstream_dev1 = ct.init_camera('/dev/ttyUSB0', '192.168.8.222')
    while True:
        r = '0000'
	if True:
		rospy.init_node('takephoto1', anonymous=True)
		single_image = rospy.Publisher("captureimage", CompressedImage, queue_size=1)
		record_pub = rospy.Publisher("savecaptureimage", CompressedImage, queue_size=1)
		ptz_pos = rospy.Publisher("ptz_pos", Roi_cam, queue_size=1)
		inspectedresult_pub = rospy.Publisher("detect_result", InspectedResult, queue_size=1)
		connectionDetect_pub = rospy.Publisher('connectionDetect_business',Int64,queue_size = 1)
		isfinish_onesingle_detect=rospy.Publisher("meter_flag",String,queue_size=1)
		# 预览图
		is_receviced_liveImg_pub = rospy.Publisher("/flag_receive_liveImg", Log, queue_size=1)
		# 检测到面
		is_detected_device_pub = rospy.Publisher("/flag_device_detect", Log, queue_size=1)

		thread.start_new_thread(sub_captureImage, ())
		thread.start_new_thread(sub_info,())
		thread.start_new_thread(init_yolo_surface, ())
		thread.start_new_thread(sub_connectSuccess, ())
		thread.start_new_thread(sub_connectMonitor,())
		thread.start_new_thread(pub_record,())
		global captureQueue
		captureQueue = Queue.Queue(10)
		capture_dir = '/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/output/'
		index = 1
		while True:
			if  menuflag==0:
		            global picId
		            t1 = time.clock()
		            if True:
		                rcvData = {}
		                global captureQueue
		                if not captureQueue.empty():
							data = captureQueue.get()
							print('@@@@@@@data:',data)
							print(captureQueue.qsize())
							global submsg
							captureimg=ct.get_planeimage(submsg, ser_dev1,ipstream_dev1)
							global picId 
							print(captureimg)
							taskId = data['taskId']
							posId = data['posId']
							camposId = data['camposId']
							camId = data['camId']
							startTimeFlag = data['startTimeFlag']
							begin_x = []
							begin_y = []
							end_x = []
							end_y = []
							if  True:
								processDb = None
								conn = None
								print('taskId,posId',taskId,posId)
								if taskId is not None and posId is not None:
									fileName = '/home/nvidia/workspace/src/detectAndRecog/src/config/process.db'
									conn = sqlite3.connect(fileName)
									processDb = conn.cursor()
									sq = 'CREATE TABLE IF NOT EXISTS process%s (time TEXT,posId TEXT, taskId TEXT,manySurfaceImg TEXT,manySurfaceCount INT);' % str(startTimeFlag)
									processDb.execute(sq)

									global yolo_surface
									if yolo_surface is not None :
									
										surfaceBox = yolo_surface.detect(captureimg)
										multiPicId = picId #图片序号
										global subMultipId
										global submsg
										subMultipId=0 #多面数
										print(surfaceBox)
										rectImg=captureimg
										if len(surfaceBox)>0:
											for box in surfaceBox:
												subMultipId+=1
												msg = Roi_cam()
												msg.begin_x = [box[2]]
												msg.begin_y = [box[3]]
												msg.end_x = [box[4]]
												msg.end_y = [box[5]]
												cv2.rectangle(rectImg,(int(box[2]),int(box[3])),(int(box[4]),int(box[5])),(0,0,255),5)
												cv2.imwrite(MulboardPath+str(multiPicId)+'-'+str(subMultipId)+'.jpg',rectImg)
												mulmsg=CompressedImage()
												mulmsg.data = np.array(cv2.imencode('.jpg',rectImg)[1]).tostring()
												mulmsg.header.frame_id = submsg+'/'+str(multiPicId)+'/'+str(subMultipId)+'-'+str(subSingleId)+'-'+ResultLable
												record_pub.publish(mulmsg)
												img,prex,prey,lagerfacter,SingleId = ct.get_singleimage(submsg,ser_dev1,ipstream_dev1,msg,single_image,picId)
												singleBox=yolo_surface.detect(img)
												single_img=img  
												resList=[]   
												subSingleId=0
												if len(singleBox)>0:
													for res in singleBox:
														sig=Roi_cam()
														x1=getXY(res[2])
														y1=getXY(res[3])
														
														x2=getXY(res[4])
														y2=getXY(res[5])
														
														
														sig.begin_x=x1
														sig.begin_y=y1
														sig.end_x=x2
														sig.end_y=y1
														subSingleId+=1
														single_img_rect=single_img[y1:y2,x1:x2]
														cv2.rectangle(single_img,(x1,y1),(x2,y2),(0,255,0),5)
														#multiPicId=SingleId
														cv2.imwrite(SinboardPath+str(SingleId)+'-'+str(subMultipId)+'-'+str(subSingleId)+'.jpg',single_img)
														sig=CompressedImage();
														sig.data = np.array(cv2.imencode('.jpg',single_img)[1]).tostring()
														sig.header.frame_id = submsg+'/'+str(SingleId)+'/'+str(subMultipId)+'-'+str(subSingleId)+'-'+ResultLable
														record_pub.publish(sig)
														global device_surface
														device_Box=device_surface.detect(single_img_rect)
													
														print("device_BOX:",device_Box)
														if len(device_Box)>0:
															ResultLable=''						
															for d_Box in device_Box:
																ResultLable=d_Box[0]
																dig=Roi_cam()
																dig.begin_x = [int(d_Box[2])+x1]
																dig.begin_y = [int(d_Box[3])+y1]
																dig.end_x = [int(d_Box[4])+x1]
																dig.end_y = [int(d_Box[5])+y1]
																cv2.rectangle(single_img,(int(d_Box[2])+x1,int(d_Box[3])+y1),(int(d_Box[4])+x1,int(d_Box[5])+y1),(255,0,0),4)
																cv2.imwrite(DevicePath+str(SingleId)+'-'+str(subMultipId)+'-'+str(subSingleId)+'-'+ResultLable+'.jpg',single_img)
																msg = CompressedImage()
																msg.data = np.array(cv2.imencode('.jpg',single_img)[1]).tostring()
																msg.header.frame_id = submsg+'/'+str(SingleId)+'/'+str(subMultipId)+'-'+str(subSingleId)+'-'+ResultLable
																record_pub.publish(msg)
																final_img,DeviceId=ct.get_deviceimage(submsg,ser_dev1,ipstream_dev1,dig,single_image,picId,prex,prey,lagerfacter)
																#result_img=final_img[int(d_Box[3])+y1-180:int(d_Box[5])+y1+60,int(d_Box[2])+x1-20:int(d_Box[4])+x1+20]
																if final_img !=None:
																	final_Box=device_surface.detect(final_img)
																	print('final_Box:',final_Box)
																	for f_box in final_Box:
																		if f_box[0]==ResultLable:
																			global picId
																			device_img_rect=final_img[int(f_box[3]):int(f_box[5]),int(f_box[2]):int(f_box[4])]
																			cv2.imwrite(DevicePath+str(DeviceId)+'-'+str(subMultipId)+'-'+str(subSingleId)+'-'+ResultLable+'.jpg',device_img_rect)
																			cv2.rectangle(final_img,(int(f_box[2]),int(f_box[3])),(int(f_box[4]),int(f_box[5])),(180,100,160),5)
																			resultmsg = CompressedImage()
																			resultmsg.data=np.array(cv2.imencode('.jpg',final_img)[1]).tostring()
																			#DeviceId=picId
																			resultmsg.header.frame_id=submsg+'/'+str(DeviceId)+'/'+str(subMultipId)+'-'+str(subSingleId)+'-'+ResultLable
																			msg = CompressedImage()
																			msg.data = np.array(cv2.imencode('.jpg',device_img_rect)[1]).tostring()
																			msg.header.frame_id = submsg+'/'+str(DeviceId)+'/'+str(subMultipId)+'-'+str(subSingleId)+'-'+f_box[0]
																			record_pub.publish(resultmsg)
																			single_image.publish(msg)
																			print ("%s Send Success", f_box[0])
																		#	dig.task_flag=str(posId) + '/' + str(taskId) + '/' + str(camId) + '/' + str(camposId)
																			ptz_pos.publish(dig)	
																	ResultLable=''			
												
						
		                                    
									else:
										msg = Roi_cam()
										msg.begin_x = begin_x
										msg.begin_y = begin_y
										msg.end_x = end_x
										msg.end_y = end_y
										msg.task_flag = str(camId) + '/' + str(posId) + '/' + str(taskId)
										ptz_pos.publish(msg)
										print('detect failed')
										count_capt = time.strftime("%H-%M-%S")
										date_capt = time.strftime('%Y-%m-%d', time.localtime(time.time()))
										imgDir = capture_dir + date_capt + '_' + count_capt + '.jpg'
										insertTime = date_capt + '-' + count_capt
										manySurfaceImg = imgDir
										manySurfaceCount = len(surfaceBox)
								global startflag
								startflag=False 
								conn.commit()
								captureimg = None
								data = None


			elif menuflag==1:
			    global content
			    str_controlinfo=str(content)
			    ct.control_camera(str_controlinfo)
			    menuflag=5

			elif menuflag==2:
			    global content
			    ct.control_camerareset(1)
			    #ct.control.camerareset(2)
			    menuflag=5

			elif menuflag==3:
			    global content
			    str_controlinfo=content
		
			    ct.hmset(ser_dev1,str_controlinfo)
		            menuflag=5
			else:
			    pass


