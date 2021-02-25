from flask_restful import Resource
from flask import request
from prx.TrainProxy import TrainProxy,ins_TrainProxy
from prx.TestProxy import TestProxy
from prx.ProjectProxy import ProjectProxy

class FastCnnApi(Resource):
    def start():
        _projectname = request.form.get('projectname')
        _tag = request.form.get('tag')
        
        rtn = {'success':False}
        if not _projectname:
            rtn['error'] = "projectname is nesserary"
            return rtn
        
        try:
            rtn = TrainProxy.start(_projectname,_tag)
            return rtn
        except Exception as err:
            rtn['error'] = str(err)
            return rtn
    
    def stop():
        try:
            return TrainProxy.stop()
        except Exception as err:
            return {'success':False,'error':str(err)}
    
    def testPictures():
        rtn = {"success":False}
        projectname = request.form.get('projectname')
        tag = request.form.get('tag')
        
        path = request.form.get('path')
        isfile = request.form.get('isfile')
        
        if not projectname:
            rtn["error"] = "projectname is nesserary"
            return rtn
        
        if not tag:
            rtn["error"] = "tag is nesserary"
            return rtn
            
        if not path:
            rtn["error"] = "path is nesserary"
            return rtn
        
        try:
            isfile = bool(isfile)
        except Exception as err:
            isfile = False
        try:
            return TestProxy.testPicture(projectname,tag,path,isfile)
        except Exception as err:
            return {'success':False,'error':str(err)}
        
    def getProjectNames():
        rtn = ProjectProxy.getProjectNames()
        return rtn
    
    def getProjectTags():
        _projectname = request.form.get('projectname')
        
        if not _projectname:
            return {"success":False,"error":"projectname is nesserary"}
        rtn = ProjectProxy.getTagNames(_projectname)
        return rtn
    
    def runTensorboard():
        return TrainProxy.runTensorBoard()
    
    def post(self):
        _cmd = request.form.get('cmd')
        
        if _cmd == "start":
            rtn = FastCnnApi.start()
            
        if _cmd == "stop":
            rtn = FastCnnApi.stop()
        
        if _cmd == "status":
            rtn = {"success":True}
        #abort
        if _cmd == "runTensorboard":
            rtn = FastCnnApi.runTensorboard()
        
        if _cmd == "testPictures":
            rtn = FastCnnApi.testPictures()
            
        if _cmd == "getProjectNames":
            rtn = FastCnnApi.getProjectNames()
        
        if _cmd == "getProjectTags":
            rtn = FastCnnApi.getProjectTags()
        
        if not rtn:
            rtn = {'success':False,'error':'no cmd detected'}
        rtn["status"] = ins_TrainProxy.Run()
        
        return rtn