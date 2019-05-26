# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model.utils.config import cfg
from model.deconv.deconv import _Deconv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import ipdb
import torch.utils.model_zoo as model_zoo

#__all__=['ResNet','resnet18','resnet34','resnet50','resnet101','resnet152']
__all__=['ResNet','resnet101']
model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}
def conv3x3(in_planes,out_planes,stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride = stride,
            padding = 1,bias = False)


###############一个三层卷积的残差块####
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self,inplanes,planes,stride = 1,downsample = None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace= True)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
#######一个三层卷积的残差块###
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,inplanes,planes,stride = 1,downsample = None):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size = 1,stride = stride,padding = 0,bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size = 3,stride = 1,padding = 1,bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size = 1,stride = 1,padding = 0,bias = False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = residual+out
        out = self.relu(out)

        return out

######extra layers: kernal_size = 2,3,1 stride:2,1,1 padding 0,1,0 residual:k=2,s=2,p=0
def weight_init(m):
    #for layer in m:
# 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
# m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Bottleneck_smooth(nn.Module):
    expansion = 4

    def __init__(self,inplanes,planes,stride = 1):
        super(Bottleneck_smooth,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size = 1,stride = stride,padding = 0,bias = False)
        weight_init(self.conv1)
        self.bn1 = nn.BatchNorm2d(planes)
        weight_init(self.bn1)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size = 3,stride = 1,padding = 1,bias = False)
        weight_init(self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        weight_init(self.bn2)
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size = 1,stride = 1,padding = 0,bias = False)
        weight_init(self.conv3)
        self.bn3 = nn.BatchNorm2d(planes*4)
        weight_init(self.bn3)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,planes*4,
                    kernel_size = 1,stride  =1,padding =0,bias = False),
                nn.BatchNorm2d(planes*4),
                )
        for layer in self.downsample:
            weight_init(layer)
        self.stride = stride

    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = residual+out
        out = self.relu(out)

        return out
class ExtraBottleneck1(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,stride = 1,downsample = None):
        """
        here:
            inplanes = 2048
            planes = 256
        """
        super(ExtraBottleneck1,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size = 2,stride = 2,padding = 0,bias = False)
        weight_init(self.conv1)
        self.bn1 = nn.BatchNorm2d(planes)
        weight_init(self.bn1)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size = 3,stride = 1,padding = 1,bias = False)
        weight_init(self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        weight_init(self.bn2)
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size = 1,stride = 1,padding = 0,bias = False)
        weight_init(self.conv3)
        self.bn3 = nn.BatchNorm2d(planes*4)
        weight_init(self.bn3)
        
        self.relu = nn.ReLU(inplace = True)
        self.downsample =  nn.Sequential(
                nn.Conv2d(inplanes,planes*4,
                    kernel_size = 2,stride  =2,padding =0,bias = False),
                nn.BatchNorm2d(planes*4),
                )
        for layer in self.downsample:
            weight_init(layer)


        self.stride = stride

    def forward(self,x):
        residual = x  #nn.Conv2d(inplanes,planes*4,kernal_size = 2,stride = 2,padding = 0)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)
        return out

##########deconv module############


###########extra layeer:kernal_size:3,3,1 stride:1,1,1 padding:0,1,0 residual:k=2,s=2,p=0
class ExtraBottleneck2(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,stride = 1,downsample = None):
        """
        here:
            inplanes = 1024
            planes = 256
        """
        super(ExtraBottleneck2,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size = 3,stride = 1,padding = 0)
        weight_init(self.conv1)
        self.bn1 = nn.BatchNorm2d(planes)
        weight_init(self.bn1)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size = 3,stride = 1,padding = 1)
        weight_init(self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        weight_init(self.bn2)
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size = 1,stride = 1,padding = 0)
        weight_init(self.conv3)

        self.bn3 = nn.BatchNorm2d(planes*4)
        weight_init(self.bn3)

        self.relu = nn.ReLU(inplace = True)
        self.downsample =  nn.Sequential(
                nn.Conv2d(  inplanes,planes*4,
                    kernel_size = 3,stride  =1,padding =0,bias = False),
                nn.BatchNorm2d(planes*4),
                )
        for layer in self.downsample:
            weight_init(layer)

        self.stride = stride

    def forward(self,x):
        residual = x#nn.Conv2d(inplanes,planes*4,kernal_size = 3,stride = 1,padding = 0)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes = 1000):
    
        self.inplanes = 64
        super(ResNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size = 7,stride = 2,padding = 3,bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 0,ceil_mode = True)

        self.layer1 = self._make_layers(block,64,layers[0])
        self.layer2 = self._make_layers(block,128,layers[1],stride = 2)
        
        self.layer3 = self._make_layers(block,256,layers[2],stride = 2)
        self.layer4 = self._make_layers(block,512,layers[3])

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512*block.expansion,num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1]* m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layers(self,block,planes,blocks,stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size = 1,
                        stride = stride,bias = False),
                    nn.BatchNorm2d(planes*block.expansion),
                    )

        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for i in range (1,blocks):
            layers.append(block(self.inplanes,planes))
        
        return nn.Sequential(*layers)


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x


######################调用类ResNet创建resnet18,32,54,101等
def resnet101(pretrained = True):
    """
    Construct a ResNet101 model
    [3,4,23,3]
    Args:
        pretrained (bool)：If True,return a model trained on Imagenet
    """
    model = ResNet(Bottleneck,[3,4,23,3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model
def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model

class resnet(_Deconv):
    def __init__(self,classes,num_layers = 101,pretrained = False,class_agnostic = False):
        self.model_path = 'data/pretrained_model/resnet50_caffe.pth'
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        #初始化FPN的结构
        _Deconv.__init__(self,classes,class_agnostic)
    
    def _init_modules(self):
        resnet = resnet50()
        

        if self.pretrained == True:
            print("Load pretrained weights from %s"%(self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

        ###base model 
        self.reslayer1 = nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu,resnet.maxpool)
        self.reslayer2 = nn.Sequential(resnet.layer1)
        self.reslayer3 = nn.Sequential(resnet.layer2)
        self.reslayer4 = nn.Sequential(resnet.layer3)
        self.reslayer5 = nn.Sequential(resnet.layer4)

        #extra layers
        ####extra layers####
        res6 = ExtraBottleneck1(2048,256)#self._make_layers(block1,256,2)
        res7 = ExtraBottleneck1(1024,256)#self._make_layers(block2,256,3)
        #res8 = ExtraBottleneck2(1024,256)
        res8 = ExtraBottleneck2(1024,256)
        res9 = ExtraBottleneck2(1024,256)

        self.reslayer6 = nn.Sequential(res6)
        self.reslayer7 = nn.Sequential(res7)
        self.reslayer8 = nn.Sequential(res8)
        self.reslayer9 = nn.Sequential(res9)

        #deconv block
        self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels = 1024,out_channels=512,kernel_size = 3,stride = 1,padding = 0),  
                #nn.ConvTranspose2d(in_channels = 1024,out_channels=512,kernel_size = 2,stride = 2,padding = 0),  

                nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512)
                
                )
        self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels = 512,out_channels=512,kernel_size = 3,stride = 1,padding = 0),  
                #nn.ConvTranspose2d(in_channels = 512,out_channels=512,kernel_size = 2,stride = 2,padding = 0),  

                nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512)
                
                )
        self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(in_channels = 512,out_channels=512,kernel_size = 2,stride = 2,padding = 0),  
                nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512)
                
                )
        self.relu = nn.ReLU(True)
        ###lateral layers
        self.lateral3 = nn.Sequential(
                nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True),
                nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True),
                )
        self.lateral5 = nn.Sequential(
                nn.Conv2d(2048,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True),
                nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True),
                )
        self.lateral6 = nn.Sequential(
                nn.Conv2d(1024,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True),
                nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True)
                )

        
        self.lateral7 = nn.Sequential(
                nn.Conv2d(1024,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True),
                nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True)
                )
        self.lateral8 = nn.Sequential(
                nn.Conv2d(1024,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True),
                nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
                nn.BatchNorm2d(512)
                
                )
        self.smooth9 = Bottleneck_smooth(1024,256)
        self.smooth8 = Bottleneck_smooth(512,256)
        self.smooth7 = Bottleneck_smooth(512,256)
        self.smooth6 = Bottleneck_smooth(512,256)        
        self.smooth5 = Bottleneck_smooth(512,256)
        self.smooth3 = Bottleneck_smooth(512,256)

        #ROI pool feature downsampling
        #self.RCNN_roi_feat_ds = Bottleneck(512,256)#in:512,256,out:1024
        self.RCNN_roi_feat_ds = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.RCNN_top = nn.Sequential(
            nn.Conv2d(1024,1024,kernel_size = cfg.POOLING_SIZE,stride = cfg.POOLING_SIZE,padding = 0),
            nn.ReLU(True),
            nn.Conv2d(1024,1024,kernel_size = 1,stride = 1,padding = 0),
            nn.ReLU(True)
        )
        self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(1024, 4 * self.n_classes)
        
        #fix blocks
        for p in self.reslayer1[0].parameters(): p.requires_grad = False
        for p in self.reslayer1[1].parameters(): p.requires_grad = False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.reslayer4.parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.reslayer3.parameters(): p.requires_grad = False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.reslayer2.parameters(): p.requires_grad = False
        
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False
        
        self.reslayer1.apply(set_bn_fix)                                        
        self.reslayer2.apply(set_bn_fix)
        self.reslayer3.apply(set_bn_fix)
        self.reslayer4.apply(set_bn_fix)
        self.reslayer5.apply(set_bn_fix)
        self.reslayer6.apply(set_bn_fix)
        self.reslayer7.apply(set_bn_fix)
        self.reslayer8.apply(set_bn_fix)
        self.reslayer9.apply(set_bn_fix)

    def _head_to_tail(self,pool5):
        block5 = self.RCNN_top(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7
    def train(self,mode = True):
        #Override train so that the training mode is set as what we want
        nn.Module.train(self,mode)
        if mode:
            # Set fixed blocks to be in eval mode
            
            self.reslayer1.eval()
            self.reslayer2.eval()
            
            self.reslayer3.train()
            self.reslayer4.train()
            self.reslayer5.train()
            self.reslayer6.train()
            self.reslayer7.train()
            self.reslayer8.train()
            self.reslayer9.train()

            self.lateral3.train()
            self.lateral5.train()
            self.lateral6.train()
            self.lateral7.train()
            self.lateral8.train()

            self.smooth3.train()
            self.smooth5.train()
            self.smooth6.train()
            self.smooth7.train()
            self.smooth8.train()
            self.smooth9.train()

            self.RCNN_top.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            #self.reslayer0.apply(set_bn_eval)
            self.reslayer1.apply(set_bn_eval)
            self.reslayer2.apply(set_bn_eval)
            self.reslayer3.apply(set_bn_eval)
            self.reslayer4.apply(set_bn_eval)
            self.reslayer5.apply(set_bn_eval)
            self.reslayer6.apply(set_bn_eval)
            self.reslayer7.apply(set_bn_eval)
            self.reslayer8.apply(set_bn_eval)
            self.reslayer9.apply(set_bn_eval)






        


