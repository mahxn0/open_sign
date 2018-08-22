import json
import base64
import sys
import numpy as np
from PIL import Image

import re
import os
import math

input_json_dir = '/home/cris/works/yjx/maskRCNN/resize/1_1_board4_75.json'
input_img_dir = '/home/cris/works/yjx/maskRCNN/resize/1_1_board4_75.jpg'

output_json_dir = '/home/cris/works/yjx/maskRCNN/json'
output_img_dir = '/home/cris/works/yjx/maskRCNN/rgb'

image = Image.open(input_img_dir)
size = (image.width,image.height)
h = image.height
w = image.width
centerPoint = [0.5*w, 0.5*h]
print(centerPoint)


def rotatePoint(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = [temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]]
    return temp_point

#print rotatePoint((1,1),(2,2),45)

with open(input_json_dir,'r') as load_f:
	load_dict = json.load(load_f)
	#print(load_dict)
	#print(load_dict['shapes'])
	#print(load_dict['imageData'])
	points = load_dict['shapes'][0]['points']
	ori_points = tuple(points)
	print(points)
	for i in range(0,360):
		for j in range(0,len(points)):
			print('the old point is:',ori_points[j])
			points[j] = rotatePoint(centerPoint, ori_points[j], i)
			print('the revolve angle is:', i, 'the new point is:',points[j])

		
		imagePath = output_img_dir + '/rgb_'+ str(i+720)+'.jpg'
		img =open(imagePath, 'rb')
		img_data =base64.b64encode(img.read())
		load_dict['imageData'] = img_data
		img.close()
		
		output_json_name = output_json_dir +'/rgb_'+ str(i+720)+'.json'

		
		f = open(output_json_name,'w')  
		f.write(json.dumps(load_dict))  
		f.close()




