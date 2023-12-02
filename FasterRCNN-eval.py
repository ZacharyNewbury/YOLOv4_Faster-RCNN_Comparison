#@title import some basic libraries and create variables for future use
import os, json, ntpath, time, re
from shutil import copyfile

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def print_time(log):
  current_time = time.strftime("%H:%M:%S", time.localtime())
  print(current_time+": "+log)

#@markdown path to LISA directory
LISA_path = "/content/drive/MyDrive/thesis/LISA dataset/" #@param {type:"string"}
#@markdown path where Detectron2 directory will be made
COCO_path = "/content/drive/MyDrive/thesis/COCO" #@param {type:"string"}
#@markdown path where Darknet directory will be made
YOLO_path = "/content/drive/MyDrive/thesis/YOLO/dataset" #@param {type:"string"}
#@markdown path where videos for getting FPS are stored
VIDEO_path = "/content/drive/MyDrive/thesis/VIDEOS" #@param {type:"string"}


print_time("creating model path")

#function to create path to model model weight path
def getModel(num):
  mnum = (str)(num)
  return os.path.join(YOLO_path,'backup','model'+(str)(mnum))

#@markdown set model path based on model number
model_num =  6 #@param {type:"number"}
model_path = getModel(model_num)


#@markdown Will create model folder and subfolders if they don't already exist
weight_path = os.path.join(model_path, 'weights')
cfg_path = os.path.join(model_path, 'cfg')
eval_path = os.path.join(model_path, 'eval')

print_time("creating folders for model if they don't already exist")
for x in [model_path, weight_path, cfg_path, eval_path]:
  if not os.exists(x):
    print_time("making folder " +ntpath.basename(x))
  os.makedirs(x, exist_ok=True)


print_time("finished setup of variables and imports")


#@title update env (Compute Capability) variable for compilation
# Compute Capability can be found at https://developer.nvidia.com/cuda-gpus
# don't include decimal in number i.e. 6.1 -> 61
# Change the number depending on what GPU is listed.
# some cloud based GPU based on whats used at google colab as of 2021-06-10
# Tesla K80: 30
# Tesla P100: 60
# Tesla V100: 70
# Tesla T4: 75

import os
os.environ['GPU_TYPE'] = str(os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read())

def getGPUArch(argument):
  try:
    argument = argument.strip()
    # All Colab GPUs
    archTypes = {
        "Tesla V100-SXM2-16GB": "-gencode arch=compute_70,code=[sm_70,compute_70]",
        "Tesla K80": "-gencode arch=compute_37,code=sm_37",
        "Tesla T4": "-gencode arch=compute_75,code=[sm_75,compute_75]",
        "Tesla P40": "-gencode arch=compute_61,code=sm_61",
        "Tesla P4": "-gencode arch=compute_61,code=sm_61",
        "Tesla P100-PCIE-16GB": "-gencode arch=compute_60,code=sm_60",
        "GeForce GTX 1080":"-gencode arch=compute_61,code=sm_61"
      }
    return archTypes[argument]
  except KeyError:
    return "GPU must be added to GPU Commands"
os.environ['ARCH_VALUE'] = getGPUArch(os.environ['GPU_TYPE'])

print("GPU Type: " + os.environ['GPU_TYPE'])
print("ARCH Value: " + os.environ['ARCH_VALUE'])

arch_value = os.environ['ARCH_VALUE']

def getEnv(inp):
  archTypes = {
    "Tesla V100-SXM2-16GB":70,
    "Tesla K80": 30,
    "Tesla T4": 75,
    "Tesla P40": 61,
    "Tesla P4": 61,
    "Tesla P100-PCIE-16GB": 60
  }
  return archTypes[inp.strip('\n')]

environment=os.environ['GPU_TYPE']
envinum=getEnv(environment)
%env compute_capability=$envinum