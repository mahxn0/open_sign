import os
ROOT_DIR = os.getcwd()
dirinput=ROOT_DIR+"/json/"
dirlist  = os.listdir(dirinput)
diroutput=ROOT_DIR+"/rgb/"
for name in dirlist:
  jpg_name=name[0:-4]
  status=os.path.exists(diroutput+jpg_name+'jpg')
  print diroutput+jpg_name+'jpg'
  if bool(status)==False:
      os.remove(dirinput+name)
