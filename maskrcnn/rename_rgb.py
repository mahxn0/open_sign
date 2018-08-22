#!/usr/bin/python
# -*- coding: utf-8 -*-
import Image
import os

def del_files(rgb_path,json_path):
    for root , dirs, files in os.walk(rgb_path):
        print(root,'   ',dirs)
        i = 0
        for name in files:

            #rename jpg
            if name.endswith(".jpg"):
                print name
                new_name = 'rgb_'+str(i)
                #new_name = 'rgb_'+str(i)
                #img = Image.open(os.path.join(root, name))
                #img.save(root +'/'+new_name+'.jpg')
                #os.remove(os.path.join(root, name))
                new_rgb_name = new_name+'.jpg'
                print (os.path.join(root, new_name))
                os.rename(os.path.join(root, name),os.path.join(root, new_rgb_name))
                

                #rename json
                json_name = name[:-4]+'.json'
            	new_json_name = new_name+'.json'
            	os.rename(os.path.join(json_path, json_name),os.path.join(json_path, new_json_name))
                i=i+1

# test
if __name__ == "__main__":
    #path = '/media/zdyd/file/zdyd/yuanfei/3DCNN-speaker-recognition/OpenSLR/CSLT public data/thuyg20-sre/zip/wav/ubm'
    rgb_path = '/media/zdyd/file/zdyd/yuanfei/maskRCNN/20180507multi_board/rgb'
    json_path = '/media/zdyd/file/zdyd/yuanfei/maskRCNN/20180507multi_board/json'
    
    del_files(rgb_path, json_path)
