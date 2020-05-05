import torch.nn as nn
from .head2head_model import Head2HeadModelG

def create_model(opt):
    modelG = Head2HeadModelG()
    modelG.initialize(opt)
    if opt.isTrain and len(opt.gpu_ids):
        modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
        from .head2head_model import Head2HeadModelD
        from .flownet import FlowNet
        modelD = Head2HeadModelD()
        flowNet = FlowNet()
        modelD.initialize(opt)
        flowNet.initialize(opt)
        modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)
        flowNet = nn.DataParallel(flowNet, device_ids=opt.gpu_ids)
        return [modelG, modelD, flowNet]
    else:
        return modelG
