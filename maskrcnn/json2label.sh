#!/bin/bash
s1="/home/cris/works/yjx/maskRCNN/json/rgb_"
s2=".json"
for((i=0;i<1080;i++))
do 
s3=${i}
labelme_json_to_dataset ${s1}${s3}${s2}
done

