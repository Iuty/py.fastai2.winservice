import os,datetime,random
import tensorflow as tf
from multiprocessing import Process

from IutyLib.mutithread.threads import LoopThread
from IutyLib.commonutil.config import Config

from prx.PathProxy import PathProxy
from prx.ClassProxy import ClassProxy

from object.Config import CNNDivParam,CNNDivSetting
import object.datasets as datasets
from object.models import getModel

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
np.set_printoptions(threshold=np.inf)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def runTensorBoard(logdir):
    
    os.system(r"tensorboard --logdir={}".format(logdir))
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
    
    """
    project init
    """
    def getTimeStamp():
        now = datetime.datetime.now()
        return datetime.datetime.strftime(now,"%Y%m%d_%H%M%S")
    
    def createDir(projectname,tag):
        if not tag:
            tag = TrainProxy.getTimeStamp()

            dir = PathProxy.getModelDir(projectname) + tag + "/"
            PathProxy.mkdir(dir)

            setting = CNNDivSetting(projectname)
            param = CNNDivParam(projectname,tag)
        
            setting.copy2(param)
            TrainProxy.recordClasses(param,projectname)
        else:
            dir = PathProxy.getModelDir(projectname) + tag + "/"
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
    
    def getDataset(self,projectname,datatype,classes,batch_size):
        trainpath = PathProxy.getProjectTrainDir(projectname)
        testpath = PathProxy.getProjectTestDir(projectname)
        print(datatype)
        if datatype == "cnn-div":
            
            train_x,train_y = datasets.BmpGroup.load_data(trainpath,classes)
            test_x,test_y = datasets.BmpGroup.load_data(testpath,classes)
            
        else:
            raise Exception("unknown data type")
        
        seed = random.randint(1,60000)
            
        self.db_train = tf.data.Dataset.from_tensor_slices((train_x,train_y))    #   将x,y分成一一对应的元组
        self.db_train = self.db_train.map(TrainProxy.preporocess)                                    #   执行预处理函数
        self.db_train = self.db_train.shuffle(seed).batch(batch_size)                          #   打乱加分组
        #   测试数据
        self.db_test = tf.data.Dataset.from_tensor_slices((test_x,test_y))
        self.db_test = self.db_test.map(TrainProxy.preporocess)
        self.db_test = self.db_test.shuffle(seed).batch(batch_size)
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
    
    def preporocess(x,y):
        x = tf.cast(x,dtype=tf.float32)
        y = tf.cast(y,dtype=tf.int32)
        return x,y
    
    def init(self,projectname,tag):
        try:
            dir,tag = TrainProxy.createDir(projectname,tag)
            param = CNNDivParam(projectname,tag)
            
            datasetname = param.Type()
            modelname = param.Model()
            classes = param.Classes()
            modelpath = PathProxy.getModelTagDir(projectname,tag)
            
            batch_size = param.Batch()
            self.batch = batch_size
            self.getDataset(projectname,datasetname,classes,batch_size)
            self.model = getModel(modelname)
            
            self.maxperiod = param.MaxPeriod()
            
            learnrate = param.LearnRate()
            adam = tf.keras.optimizers.Adam(lr=learnrate)
            self.model.compile(optimizer=adam,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
            
            
            self.period = param.Period()
            #meters
            self.acc_meter = tf.keras.metrics.Accuracy()
            self.loss_meter = tf.keras.metrics.Mean()
            return True
        except Exception as err:
            raise err
            return False
        
    
    def startTrain(self,projectname,tag):
        
        if not self.init(projectname,tag):
            return False
        
        """
        self._loop_thread = LoopThread(TrainProxy.doTrain)
        self._loop_thread.start()
        """
        TrainProxy.doTrain()
        return True
    """
    instance here
    """
    
    def __init__(self):
        self._loop_thread = None
        
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        
        self.batch = None
        self.learnrate = None
        self.optimizer = None
        
        self.model = None
        
        self.train_logit = None
        self.train_loss = None
        self.train_op = None
        self.train_acc = None
        
        self.test_logit = None
        self.test_loss = None
        self.test_acc = None
        
        self.runflag = False
        self.curstep = 0
        self.period = 1
        self.maxperiod = 0
        
        self.db_train = None
        self.db_test = None
        pass
    
    def __del__(self):
        self.stopTrain()
        pass
    
    
    def doTrain():
        log_dir = os.path.join('logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        #p = Process(target=runTensorBoard,args=[log_dir,])
        #p.start()
        
        checkpoint_save_path = "./checkpoint/mnist.ckpt"
    
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            #ins_TrainProxy.model.load_weights(checkpoint_save_path)
            
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_weights_only=False,
                                                         save_best_only=False)
            
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)
            
        history = ins_TrainProxy.model.fit(
                            ins_TrainProxy.train_x, ins_TrainProxy.train_y, 
                            batch_size=ins_TrainProxy.batch, 
                            epochs=ins_TrainProxy.maxperiod, 
                            #steps_per_epoch=10,
                            #validation_steps=10,
                            validation_data=(ins_TrainProxy.test_x, ins_TrainProxy.test_y), 
                            validation_freq=1,
                            callbacks=[cp_callback,tensorboard])
            
    
    def Run(self):
        if not self._loop_thread:
            return False
        else:
            return self._loop_thread._running
    
    
    
    def stopTrain(self):
        if self.Run():
            #CoachProxy.stop()
            self._loop_thread.stop()
        pass
    
    pass

ins_TrainProxy = TrainProxy()
