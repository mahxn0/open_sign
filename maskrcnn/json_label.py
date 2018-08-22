import json
import base64
import sys
import numpy as np
from PIL import Image

import re
import os
import math

root = '/media/zdyd/file/zdyd/yuanfei/maskRCNN/watch/json/'



for name in os.listdir(root):
	print name
	if name.endswith(".json"):
		#print os.path.join(root,name)
		with open(os.path.join(root,name),'r') as load_f:
			load_dict = json.load(load_f)
			label = load_dict['shapes'][0]['label']
			print(label)
			load_dict['shapes'][0]['label'] = 'needle2'

			#new_name = name.split('rgb_')[1]
			#f = open(os.path.join(root,new_name),'w')
			f = open(os.path.join(root,name),'w')
			f.write(json.dumps(load_dict))  
			f.close()
				




