import os,time

from PIL import Image
import numpy as np
import tensorflow as tf

from IutyLib.mutithread.threads import LoopThread
from IutyLib.commonutil.config import Config

from multiprocessing import Process,Manager

from prx.PathProxy import PathProxy
from prx.ClassProxy import ClassProxy
from prx.CoachProxy import CoachProxy
#from prx.CoachProxy import ins_CoachProxy

from object.Config import CNNDivParam,CNNDivSetting
import object.datasets as datasets
from object.models import getModel

def killTensorBoard():
    os.system(r'taskkill /f /t /im "tensorboard.exe" ')
    pass

class TestProxy:
    """
    api here
    """
    def testPicture(project,tag,path,isfile):
        isfile = False
        rtn = {'success':False}
        
        succ = False
        data = {}
        details = {}
        
        manager = Manager()
        return_dict = manager.dict()
        
        process = Process(target=TestProxy.predictMulti,args=(project,tag,path,isfile,return_dict))
        
        process.start()
        process.join()
        
        process.kill()
        """
        succ,data,details = TestProxy.predictModel(project,tag,path,isfile)
        if not succ:
            rtn['error'] = data
            return rtn
        rtn['data'] = data
        rtn['details'] = details
        
        """
        rtn["data"] = return_dict["data"]
        rtn["details"] = return_dict["details"]
        rtn['success'] = True
        return rtn
    
        
    """
    methods here
    """
    def getStatistics(test_result):
        result = {}
        for item in test_result:
            if not item["result"] in result:
                result[item["result"]] = 0
            result[item["result"]] += 1
        return result
    
    def getGroup(datatype,path,file):
        if datatype == "cnn-div":
            group_x,data_x = datasets.BmpGroup.load_pictures(path,file)
        
        else:
            raise Exception("unknown data type")
        return group_x,data_x
    
    def predictMulti(projectname,tag,path,file,rtn):
        
        succ,data,details = TestProxy.predictModel(projectname,tag,path,file,rtn)
        
        pass
    
    def predictModel(projectname,tag,path,file,out_rtn = {}):
        killTensorBoard()
        
        param = CNNDivParam(projectname,tag)
        datasetname = param.Type()
        modelname = param.Model()
        classes = param.Classes()
        
        model = getModel(modelname)
        
        """
        model.compile(optimizer=adam,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        """
        tagpath = PathProxy.getModelTagDir(projectname,tag)
        
        checkpoint_path = os.path.join(tagpath,"checkpoints","save.ckpt")
        
                
        result = []
        if os.path.exists(checkpoint_path + '.index'):
            print('-------------load the model-----------------')
            model.load_weights(checkpoint_path)
        else:
            raise Exception("No model called")
        
        group_x,data_x = TestProxy.getGroup(datasetname,path,file)
        """
        for g_x in group_x:
            print(g_x)
            r = model.predict(np.array[g_x,])
            result.append(r)
        """
        
        result = model.predict(data_x,use_multiprocessing=True)
        result = tf.argmax(result, axis=1)
        result = result.numpy()
        
        details={}
        
        rtn = {}
        for i in range(len(group_x)):
            rtn[group_x[i]] = classes[str(result[i])]
            if not classes[str(result[i])] in details:
                details[classes[str(result[i])]] = 0
            details[classes[str(result[i])]]+=1
        out_rtn['success'] = True
        out_rtn['data'] = rtn
        out_rtn['details'] = details
        
        return True, rtn,details

if __name__ == "__main__":
    TestProxy.predictModel("Lightning","20210222_131420",r"D:\FastCNN\Projects\Lightning\test\暗点")
    
