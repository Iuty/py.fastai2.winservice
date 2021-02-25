from IutyLib.commonutil.config import Config
from prx.PathProxy import PathProxy
import os

"""
Server Config
"""
class ServerConfig:
    def __init__(self):
        self._config = Config(PathProxy.getConfigPath())
        pass
    
    def set(self,secticon,key,value):
        self._config.set(secticon,key,str(value))
        pass
    
    def createConfig(self):
        self.set("Server","port","7738")
        pass
    
    def Port(self):
        return self._config.get("Server","port")
    pass
    
"""
Project Setting
"""
class ProjectSetting:
    def __init__(self,projectname):
        self._config = Config(PathProxy.getSettingPath(projectname))
        pass
    
    def set(self,secticon,key,value):
        self._config.set(secticon,key,str(value))
        pass
    
    pass

class CNNDivSetting(ProjectSetting):
    d_value = [
        ('Common','type','cnn-div'),
        
        ('Image','width','64'),
        ('Image','height','64'),
        ('Image','depth','3'),
        ('Image','formatter','bmp'),
        
        ('ImgGroup','enable','1'),
        ('ImgGroup','0','0'),
        ('ImgGroup','1','1'),
        ('ImgGroup','2','2'),
        ('ImgGroup','3','3'),
        ('ImgGroup','4','4'),
        ('ImgGroup','5','5'),
        
        ('Train','batch','128'),
        ('Train','learnrate','0.0001'),
        ('Train','period','10'),
        ('Train','saveperiod','10'),
        ('Train','maxperiod','100000'),
        
        ('Logit','model','AlexNet8'),
        ('Logit','conv1depth','32'),
        ('Logit','conv2depth','64')
    ]
    
    def __init__(self,projectname):
        ProjectSetting.__init__(self,projectname)
        pass
    
    def createConfig(self):
        cfg = self._config
        for section,type,default in self.d_value:
            cfg.set(section,type,default)
        pass
    
    def copy2(self,cfg):
        #cfg = Config(path)
        cfg0 = self._config
        for section,key,default in self.d_value:
            
            cfg0.copy(cfg,section,key,default)
        pass
    pass
    
"""
Train Param
"""
class CNNTrainParam:
    def __init__(self,projectname,tag):
        path = PathProxy.getModelParamPath(projectname,tag)
        self._exists = os.path.exists(path)
        self._config = Config(path)
        pass
    
    def Exists(self):
        return self._exists
    
    def set(self,secticon,key,value):
        self._config.set(secticon,key,str(value))
        pass
    
    def Type(self):
        return self._config.get("Common","type")
    
    def Width(self):
        return int(self._config.get("Image","width"))
    
    def Height(self):
        return int(self._config.get("Image","height"))
    
    def Depth(self):
        return int(self._config.get("Image","depth"))
    
    def Formatter(self):
        return self._config.get("Image","formatter")
        
    pass

class CNNDivParam(CNNTrainParam):
    def __init__(self,projectname,tag):
        CNNTrainParam.__init__(self,projectname,tag)
        pass
    
    def Batch(self):
        return int(self._config.get("Train","batch"))
    
    def LearnRate(self):
        return float(self._config.get("Train","learnrate"))
        
    def Period(self):
        return int(self._config.get("Train","period"))
    
    def SavePeriod(self):
        return int(self._config.get("Train","saveperiod"))
    
    def MaxPeriod(self):
        return int(self._config.get("Train","maxperiod"))
    
    def Model(self):
        return self._config.get("Logit","model")
    
    def Conv1Depth(self):
        return int(self._config.get("Logit","conv1depth"))
    
    def Conv2Depth(self):
        return int(self._config.get("Logit","conv2depth"))
    
    def Classes(self):
        classes = {}
        for i in range(0,1000):
            classname = self._config.get("Classes",str(i))
            if classname:
                classes[str(i)] = classname
            else:
                break
        
        return classes
    
    def GroupEnable(self):
        try:
            v = int(self._config.get("ImgGroup","enable"))
            return bool(v)
        except:
            return False
    
    def Groups(self):
        groups = {}
        for i in range(0,1000):
            groupname = self._config.get("ImgGroup",str(i))
            if groupname:
                groups[str(i)] = groupname
            else:
                break
        
        return groups
    
    #overwrite depth
    def Depth(self):
        if self.GroupEnable():
            return len(self.Groups())
        else:
            return int(self._config.get("Image","depth"))
    pass
    