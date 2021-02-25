import os,datetime,random,time
import tensorflow as tf
from multiprocessing import Process

from IutyLib.commonutil.config import Config

from prx.PathProxy import PathProxy
from prx.ClassProxy import ClassProxy

from object.Config import CNNDivParam,CNNDivSetting
import object.datasets as datasets
from object.models import getModel

import numpy as np




# -*- coding: utf-8 -*-  

def runTensorBoard(logdir):
    print("run tensorboard,logdir="+logdir)
    #killTensorBoard()
    os.system(r"start D:\Python37\Scripts\tensorboard.exe --logdir={}".format(logdir))
    pass

def startTensorBoard(tensorboardpath):
    
    os.system(r"call {}".format(tensorboardpath))
    pass

def killTensorBoard():
    os.system(r'taskkill /f /t /im "tensorboard.exe" ')
    pass

class TrainProxy:
    """
    api here
    """
    def stop():
        rtn = {'success':False}
        
        ins_TrainProxy.stopTrain()
        
        rtn['success'] = True
        return rtn
    
    def start(projectname,tag):
        rtn = {'success':False}
        
        if ins_TrainProxy.Run():
            rtn['error'] = "System has in train process"
            return rtn
        
        if not ins_TrainProxy.startTrain(projectname,tag):
            rtn['error'] = "The coach can not start train process"
            return rtn
        
        rtn['success'] = True
        return rtn
    
    
    
    def runTensorBoard():
        rtn = {'success':False}
        log_dir = ins_TrainProxy.log_dir
        if log_dir != "":
            ins_TrainProxy.tensorboard = Process(target=runTensorBoard,args=(log_dir,))
            ins_TrainProxy.tensorboard.start()
        
        rtn['success'] = True
        return rtn
    
    """
    project init
    """
    def getTimeStamp():
        now = datetime.datetime.now()
        return datetime.datetime.strftime(now,"%Y%m%d_%H%M%S")
    
    """
    获取tag路径和tag
    """
    def createDir(projectname,tag):
        #if not tag:
        if True:
            tag = TrainProxy.getTimeStamp()

            dir = PathProxy.getModelTagDir(projectname,tag)
            PathProxy.mkdir(dir)

            setting = CNNDivSetting(projectname)
            param = CNNDivParam(projectname,tag)
        
            setting.copy2(param)
            TrainProxy.recordClasses(param,projectname)
        else:
            dir = PathProxy.getModelTagDir(projectname,tag)
        return dir,tag
    
    def recordClasses(paramcfg,projectname):
        modeltype = paramcfg.Type()
        if modeltype == "cnn-div":
            
            classes = ClassProxy.getClasses(projectname)
            for k in classes:
                
                paramcfg.set("Classes",k,classes[k])
        
        if modeltype == "divtf2":
            
            classes = ClassProxy.getClasses(projectname)
            for k in classes:
                
                paramcfg.set("Classes",k,classes[k])
        pass
    
    def getDataset(projectname,datatype,classes):
        trainpath = PathProxy.getProjectTrainDir(projectname)
        testpath = PathProxy.getProjectTestDir(projectname)
        
        if datatype == "cnn-div":
            train_x,train_y = datasets.BmpGroup.load_data(trainpath,classes)
            test_x,test_y = datasets.BmpGroup.load_data(testpath,classes)
            
        else:
            raise Exception("unknown data type")
        
        return train_x,train_y,test_x,test_y
    
            
    
    def preporocess(x,y):
        x = tf.cast(x,dtype=tf.float32)
        y = tf.cast(y,dtype=tf.int32)
        return x,y
    
    def load(self,projectname,tag):
        try:
            dir,tag = TrainProxy.createDir(projectname,tag)
            
            return True,tag
        except Exception as err:
            print(err)
            return False,tag
        
    
    def startTrain(self,projectname,tag):
        loadtag = tag
        
        loaded,tag = self.load(projectname,tag)
        if not loaded:
            return False
        """
        if self.tensorboard:
            self.tensorboard.kill()
            self.tensorboard.join()
            self.tensorboard = None
        """
        killTensorBoard()
        
        tagpath = PathProxy.getModelTagDir(projectname,tag)
        batpath = os.path.join(tagpath,"train_vis.bat")
        f = open(batpath,"w+")
        
        batcontent = '@echo off \r\ntaskkill /f /t /im "tensorboard.exe" \r\nstart D:\\Python37\\Scripts\\tensorboard.exe --logdir=%cd%\\logs'
        f.write(batcontent)
        f.close()
        #log
        log_dir = os.path.join(tagpath,'logs')
        self.log_dir = log_dir
        
        self.process = Process(target=TrainProxy.trainModel,args=(projectname,tag,loadtag))
        self.process.start()
        
        tensorboard = Process(target=runTensorBoard,args=(log_dir,))
        tensorboard.start()
        
        #self.tensorboard = Process(target=startTensorBoard,args=(batpath,))
        #self.tensorboard.run()
        
        
        return True
    """
    instance here
    """
    
    def __init__(self):
        
        self.param = None
        self.process = None
        self.tensorboard = None
        
        self.log_dir = ""
        
        self.runflag = False
        self.curstep = 0
        self.period = 1
        self.maxperiod = 0
        
        pass
    
    def __del__(self):
        self.stopTrain()
        pass
    
    
    def trainModel(projectname,tag,loadtag):
        
        
        param = CNNDivParam(projectname,tag)
        datasetname = param.Type()
        modelname = param.Model()
        classes = param.Classes()
        #modelpath = PathProxy.getModelTagDir(projectname,tag)
            
        batch_size = param.Batch()
        
        train_x,train_y,val_x,val_y = TrainProxy.getDataset(projectname,datasetname,classes)
        model = getModel(modelname)
            
        maxperiod = param.MaxPeriod()
            
        learnrate = param.LearnRate()
        adam = tf.keras.optimizers.Adam(lr=learnrate)
        #loss here
        model.compile(optimizer=adam,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
            
        #meters
        acc_meter = tf.keras.metrics.Accuracy()
        loss_meter = tf.keras.metrics.Mean()
        
        tagpath = PathProxy.getModelTagDir(projectname,tag)
        loadtagpath = PathProxy.getModelTagDir(projectname,loadtag)
        
        #checkpoint
        checkpoint_load_path = os.path.join(loadtagpath,"checkpoints","save.ckpt")
        checkpoint_save_path = os.path.join(tagpath,"checkpoints","save.ckpt")
        
        
        #log
        log_dir = os.path.join(tagpath,'logs')
        
        
        if os.path.exists(checkpoint_load_path + '.index'):
            print('-------------load the model-----------------')
            model.load_weights(checkpoint_load_path)
        
        #callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                        save_freq='epoch',
                                                        save_weights_only=True,
                                                        save_best_only=True)
        
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=0)
        
        #fit
        history = model.fit(
                            train_x, train_y, 
                            batch_size=batch_size, 
                            epochs=maxperiod, 
                            #steps_per_epoch=10,
                            #validation_steps=10,
                            
                            validation_data=(val_x, val_y), 
                            validation_freq=1,
                            callbacks=[cp_callback,tensorboard])
    
    
    
        
    
    def Run(self):
        if not self.process:
            return False
        else:
            return self.process.is_alive()
    
    
    
    def stopTrain(self):
        
        if self.Run():
            #CoachProxy.stop()
            self.process.kill()
            self.process.join()
            self.process = None
        pass
    
    pass

ins_TrainProxy = TrainProxy()
