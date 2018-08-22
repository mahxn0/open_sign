# -*- coding: utf-8 -*-
import os
import cv2
# 导解析器的包
from xml.dom.minidom import parse
# 导 minidom 的包
import xml.dom.minidom
def xml_cut(dirinput,diroutput):
    for fpathe,dirs,fs in os.walk(dirinput):
        for f in fs:
          if f[len(f)-3:] == "jpg":# and f.find("_out") > 0:
                
                filename = os.path.join(fpathe,f)
                subDir = filename[len(dirinput):]
                folderPos = subDir.find("/")
                folderName = subDir[:folderPos]#文件夹名字
                if not os.path.exists(diroutput+folderName):
                     os.makedirs(diroutput+folderName)
                without=f[0:len(f)-3]
                xml_name=without+"xml"
                xmlfilepath = os.path.join(fpathe,xml_name)
                DOMTree = parse(xmlfilepath)
                # 获得根节点
                root = DOMTree.documentElement
                # 获得所有 book 节点
                books = DOMTree.getElementsByTagName("object")
                j=0
                # 遍历
                for book in books :
                  j+=1
                  i=0
                  # 打印 book 节点的 id 属性
                  #print("***book id=%s"%(book.getAttribute("id")))
                  # 获得 name 属性的第一个元素对象，获得该元素对象的第一个子节点对象的  data
                  name = book.getElementsByTagName("name")[0].childNodes[0].data
                  if name=="box_watch":
                      i+=1
                      subElementObj1 = book.getElementsByTagName("xmin")
                      xmin=subElementObj1[0].firstChild.data
                      subElementObj2 = book.getElementsByTagName("ymin")
                      ymin=subElementObj2[0].firstChild.data
                      subElementObj3 = book.getElementsByTagName("xmax")
                      xmax=subElementObj3[0].firstChild.data
                      subElementObj4 = book.getElementsByTagName("ymax")
                      ymax=subElementObj4[0].firstChild.data
                      img = cv2.imread(filename)
                      roi = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                      cv2.imwrite(diroutput+folderName+'/'+str(j)+"_"+str(i)+"_"+f,roi)
                      print("root element is:%s"%(xmin))
if __name__ == "__main__":
    ROOT_DIR=os.getcwd()
    dirinput=ROOT_DIR+"/xmlboard/"
    dirlist=os.listdir(dirinput)
    diroutput=ROOT_DIR+"/board/"
    xml_cut(dirinput,diroutput)
    
