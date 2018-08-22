from PIL import Image
import os
import os.path
 
inputdir = '/home/cris/works/yjx/maskRCNN/resize/1_1_board4_75.jpg'  
output_dir = '/home/cris/works/yjx/maskRCNN/rgb' 

print('the full name of the original file is :' + inputdir)

for i in range(0,360):

    im = Image.open(inputdir)    
    #clockwise angle:i 
    out = im.rotate(360-i)#this has to be 360 - i , otherwise, the generated mask won't match
    
    #anticlockwise angle:i 
    #out = im.rotate(i)
    newname = output_dir+'/'+ 'rgb_'+ str(i+720) +'.jpg'
    print('the new file is: '+ newname)
    out.save(newname)
