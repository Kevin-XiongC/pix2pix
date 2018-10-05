import shutil
import os
os.chdir("C:\\Users\\XC\\Desktop\\PRO\\pix2pix\\indoors\\val")
for i in range(1000,1449):
    os.rename("dep_%d.jpg"%i,"dep_%d.jpg"%(i-1000))
    os.rename("img_%d.jpg"%i, "img_%d.jpg"%(i-1000))