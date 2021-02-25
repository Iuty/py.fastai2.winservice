
import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from IutyLib.file.log import SimpleLog
from IutyLib.commonutil.config import Config
from flask import Flask
from flask_restful import *#Api,Resource
from flask_cors import *
import multiprocessing
from prx.PathProxy import PathProxy

import tensorflow as tf

from api.project_ import *
from api.class_ import *
from api.train_ import *
from api.test_ import *
from api.fastcnn_ import *
import numpy as np
np.set_printoptions(threshold=np.inf)

app_path = PathProxy.app_path
project_path = PathProxy.project_path
PathProxy.mkdir(project_path)

app_log = SimpleLog(os.path.join(app_path,"logs")+"\\")

#tf.logging.set_verbosity(tf.logging.ERROR)

config = Config(PathProxy.getConfigPath())

app = Flask(__name__)
api = Api(app)
CORS(app,supports_credentials=True)

api.add_resource(FastCnnApi,'/api/nn/fastcnn')
api.add_resource(ProjectApi,'/api/project')
api.add_resource(ClassApi,'/api/class')

api.add_resource(TrainApi,'/api/train')
api.add_resource(TestApi,'/api/test')

host = '0.0.0.0'
port = int(config.get("Server","port"))

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import GPUOptions
tfconfig = ConfigProto()
tfconfig.gpu_options.per_process_gpu_memory_fraction = 1

tfconfig.gpu_options.allow_growth = True
session = InteractiveSession(config=tfconfig)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app.run(host=host,port=port,debug=False ,use_reloader=False)
    