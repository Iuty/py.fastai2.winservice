from IutyLib.commonutil.config import Config 
import os
import numpy as np
import tensorflow as tf
from PIL import Image

default_groups = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5"}
default_imgw=96
default_imgh=96
default_imgf="bmp"

class BmpGroup:
    def checkGroup(dirpath,filehead,formatter=None,groups=None):
        if not groups:
            groups = default_groups
        
        if not formatter:
            formatter = "bmp"
            
        for k in groups:
            fpath = os.path.join(dirpath,filehead+"_"+groups[k]+"."+formatter)
            if not os.path.exists(fpath):
                return False
        return True

    def getGroupHead(filename):
        
        fgname = ""
        fnames = filename.split('_')
        if len(fnames) < 2:
            return fgname
            
        else:
            
            for i in range(0,len(fnames)-1):
                if len(fgname) > 0:
                    fgname += "_"
                fgname += fnames[i]
            
            return fgname

    def walkDir(dir,cls = "0",formatter=None,groups=None):
        if not formatter:
            formatter = "bmp"
        data = []
        label = []
        for mdir,subdir,filename in os.walk(dir):
            
            if len(filename) > 0:
                for f in filename:
                    
                    fsplit = f.split('.')
                    if len(fsplit)>1:
                        if fsplit[-1] == formatter:
                            
                            fghead = BmpGroup.getGroupHead(fsplit[0])
                                    
                            if fghead != "":
                                
                                if os.path.join(mdir,fghead) in data:
                                    continue
                                else:
                                    suc = BmpGroup.checkGroup(mdir,fghead,formatter=formatter,groups=groups)
                                            
                                    if suc:
                                        data.append(os.path.join(mdir,fghead))
                                        label.append(int(cls))
                                    else:
                                        continue
                            else:
                                continue
        return data,label
        
    def getInputData(dir,classes,groups=None,formatter=None):
        data = []
        label = []
        for cls in classes:
            #for mdir,subdir,filename in os.walk(dir+cls+"_"+classes[cls]):
            d,l = BmpGroup.walkDir(os.path.join(dir,classes[cls]),cls,formatter=formatter,groups=groups)
            data += d
            label += l
        
        temp = np.array([data, label])
        temp = temp.transpose()
        np.random.shuffle(temp)
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [int(float(i)) for i in label_list]
        return image_list,np.array(label_list)
        
    def readGroup(files,imgw=None,imgh=None,imgf=None,groups=None):
        if not groups:
            groups = default_groups
        if not imgw:
            imgw = default_imgw
        if not imgh:
            imgh = default_imgh
        if not imgf:
            imgf = default_imgf
        
        image_group = []
        
        for file in files:
            img_group = []
            for k in groups:
                apd = "_"+groups[k]+"."+imgf
                img_content = file + apd
                image_contents = tf.io.read_file(img_content)
                if imgf == "bmp":
                    image=tf.image.decode_bmp(image_contents)
                    image = image.numpy()
                    image = image/255.0
                    
                img_group.append(image)
            image_group.append(img_group)
        #image_group = tf.reshape(image_group,[imgw,imgh,len(img_group)])
        load_image_group = np.array(image_group).reshape(len(image_group),imgw,imgh,len(groups))
        return load_image_group
    
    def load_data(path,classes):
        
        x,y = BmpGroup.getInputData(path,classes)
        
        x = BmpGroup.readGroup(x)
        print("load data shape: {},label shape: {}".format(x.shape,y.shape))
        return x,y
    
    def load_pictures(dir,file = None):
        data,label = BmpGroup.walkDir(dir,"0")
        image_list = []
        for d in data:
            if file:
                end = len(file)
                if d[-end:] == file:
                    image_list.append(d)
            else:
                image_list.append(d)
        print("load picture count = "+str(len(image_list)))
        data_list = BmpGroup.readGroup(image_list)
        
        return image_list,data_list
        
    
if __name__ == "__main__":
    #x,y = BmpGroup.load_data(r"D:\FastCNN\Projects\Lightning\train",{"0":"leak_亮点","1":"暗点","2":"亮暗并列","3":"亮点"})
    #x = BmpGroup.load_pictures(r"D:\FastCNN\Projects\Lightning\train")
    #print(x)
    
    #x = BmpGroup.load_pictures(r"D:\FastCNN\Projects\Lightning\train","20200831000133_TH083230AB08_D000_te")
    #print(x)
    pass