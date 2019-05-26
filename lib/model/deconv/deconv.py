# -*- coding: utf-8 -*-
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
import numpy as np
import torchvision.utils as vutils
from model.rpn.rpn_deconv import _RPN_Deconv
#from model.rpn.rpn import _RPN
from model.utils.config import cfg

# import ipdb
# ipdb.set_trace()
#from model.roi_pooling.modules.roi_pool import _ROIPooling
from model.roi_crop.modules.roi_crop import  _RoICrop
from model.roi_pooling.modules.roi_pool import _RoIPooling

from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss,_crop_pool_layer,_affine_grid_gen,_affine_theta
import time
import ipdb


USE_ONE_FEATURE = 0 

####################conv-deconv module############        
class _Deconv(nn.Module):
    """
    DECONV
      
    here:
        inplanes:2048
        planes:256
        out_dimension:1024=256*4
        
    """
    def __init__(self,classes,class_agnostic):
        super(_Deconv,self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        

        #loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.maxpool2d = nn.MaxPool2d(1,stride = 2)

        #define rpn
        # if USE_ONE_FEATURE == 0:
        self.RCNN_rpn = _RPN_Deconv(self.dout_base_model)
        # else:
        #     self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE,cfg.POOLING_SIZE,1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE,cfg.POOLING_SIZE,1.0/16.0)
        self.grid_size = cfg.POOLING_SIZE*2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
    def _init_weights(self):

        def normal_init(m,mean = 0,stddev = 0.01,truncated = False):
            """
            weight initalizer:truncated normal and random normal.
            """
            #x is a parameter
         
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean,stddev)
                m.bias.data.zero_()
        
        def weights_init(m,mean = 0,stddev = 0.01,truncated = False):
            """

            """
            for layer in m:

                classname = layer.__class__.__name__
                if classname.find('Conv') != -1:
                    layer.weight.data.normal_(0.0,0.02)
                    layer.bias.data.fill_(0)
                elif classname.find('BatchNorm') != -1:
                    layer.weight.data.normal_(1.0,0.02)
                    layer.bias.data.fill_(0)

        weights_init(self.lateral8, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.lateral7, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.lateral6, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.lateral5, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.lateral3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.deconv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.deconv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.deconv3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_roi_feat_ds, 0, 0.01, cfg.TRAIN.TRUNCATED)

        #init multi featuremap RPN
        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _deconv_add1(self,x,y,method = 'EltwiseSUM'):
        '''
        Deconv and add two feature maps
        Args:
            x: (Variable) top feature map to be deconvolutioned.
            y: (Variable) lateral feature map.

        Return:
            (Variable) added feature map.
        Note in Pytorch, when input size is odd, the upsampled feature map
        with 'F.upsample(...,scale_factor = 2,mode = 'nearest')'
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        deconvolutioned deatu
        original input size:[]
        '''
        #deconv: deconv
        #       conv
        #       bn
        # self.deconv = nn.Sequential(
        #         nn.ConvTranspose2d(in_channels = inplanes,out_channels=512,kernel_size = k,stride = s,padding = 0),  
        #         nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
        #         nn.BatchNorm2d(512)
                
        #         )
        #c = str(x)
        
        if method == 'EltwiseSUM':
            out = self.deconv1(x)+y
            
        elif method == 'EltwisePROD':
            out = self.deconv1(x)*y
            
        else:
            assert False,"Wrong method name :{}".format(method)
        out = self.relu(out)
        return out
        
    def _deconv_add2(self,x,y,method = 'EltwiseSUM'):
            
        if method == 'EltwiseSUM':
            out = self.deconv2(x)+y
            
        elif method == 'EltwisePROD':
            out = self.deconv2(x)*y
            
        else:
            assert False,"Wrong method name :{}".format(method)
        out = self.relu(out)
        return out
    def _deconv_add3(self,x,y,method = 'EltwiseSUM'):
            
        if method == 'EltwiseSUM':
            out = self.deconv3(x)+y
            
        elif method == 'EltwisePROD':
            out = self.deconv3(x)*y
            
        else:
            assert False,"Wrong method name :{}".format(method)
        out = self.relu(out)
        return out
    def _deconv_add(self,x,y,method = 'EltwiseSUM'):
        _,_,H,W = y.size()
        out = F.upsample(x,size = (H,W),mode = 'bilinear')
        if method == 'EltwiseSUM':
            out = out+y
            
        elif method == 'EltwisePROD':
            import ipdb
            ipdb.set_trace()
            out = out*y
            
        else:
            assert False,"Wrong method name :{}".format(method)
        out = self.relu(out)
        return out
    def _PyramidRoI_Feat(self,feat_maps,rois,im_info):
        """
        roi pool on ptramid feature map
        """
        img_area = im_info[0][0]*im_info[0][1]
        h = rois.data[:,4]-rois.data[:,2]+1
        w = rois.data[:,3]-rois.data[:,1]+1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        ##################################################################
        ####################not use feature pyramid#######################
        ##################################################################
        roi_level[:] = 2


        # roi_level.fill_(5)
        if cfg.POOLING_MODE == 'crop':
            #NOTE:multi levels roi_features has not been implemented
            grid_xy = _affine_grid_gen(rois,base_feat.size()[2:],self.grid_size())
            grid_yx = torch.stack([grid_xy.data[:,:,:,1],grid_xy.data[:,:,:,0]],3).contiguous()
            roi_pool_feat = self.RCNN_roi_crop(base_feat,Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                roi_pool_feat = F.max_pool2d(roi_pool_feat,2,2)

        elif cfg.POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            feat = self.RCNN_roi_align(base_feat,rois[5],scale)
            box_to_level = torch.cat(roi_pool_feats,0)
        elif cfg.POOLING_MODE == 'pool':
            roi_pool_feats = []
            box_to_levels = []
            # import ipdb
            # ipdb.set_trace()
            #for i,l in enumerate(range(2,6)):
            for i,l in enumerate(range(2,3)):
                if(roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().view(-1)
                box_to_levels.append(idx_l)
                # import ipdb
                # ipdb.set_trace()
                scale = float(feat_maps[i].size(2) / im_info[0][0])

                feat = self.RCNN_roi_pool(feat_maps[i],rois[idx_l],scale)
                roi_pool_feats.append(feat)
            
            roi_pool_feats = torch.cat(roi_pool_feats,0)
            box_to_level = torch.cat(box_to_levels,0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feats = roi_pool_feats[order]
        return roi_pool_feats

    def forward(self,im_data,im_info,gt_boxes,num_boxes):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        #feed image data to base model to obtain base feature map
        #encoder
        res1 = self.reslayer1(im_data)#im_data:[1,3,850,600],c1[1,64,212,150]
        res2 = self.reslayer2(res1)#c2[1, 256, 212, 150]
        res3 = self.reslayer3(res2)#c3[1, 512, 106, 75]
        res4 = self.reslayer4(res3)#c4[1, 1024, 53, 38]
        res5 = self.reslayer5(res4)#c5[1, 2048, 53, 38]

        res6 = self.reslayer6(res5)
        res7 = self.reslayer7(res6)
        res8 = self.reslayer8(res7)
        res9 = self.reslayer9(res8)
        #decoder
        import ipdb
        ipdb.set_trace()
        p9 = self.smooth9(res9)
        combined8 = self._deconv_add1(res9,self.lateral8(res8),'EltwisePROD')
        p8 = self.smooth8(combined8)
        combined7 = self._deconv_add2(combined8,self.lateral7(res7),'EltwisePROD')
        p7 = self.smooth7(combined7)

        combined6 = self._deconv_add3(combined7,self.lateral6(res6),'EltwisePROD') 
        p6 = self.smooth6(combined6)
        combined5 = self._deconv_add3(combined6,self.lateral5(res5),'EltwisePROD')
        p5 = self.smooth5(combined5)
        combined3 = self._deconv_add3(combined5,self.lateral3(res3),'EltwisePROD')
        p3 = self.smooth3(combined3)

        ##################################################################
        ####################not use feature pyramid#######################
        ##################################################################
        rpn_feature_maps = [p3]#,p5,p6,p7,p8]
        mrcnn_feature_maps = [p3]#,p5,p6,p7]
        
        
        #rpn网络前传
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feature_maps,im_info,gt_boxes,num_boxes)

        #ipdb.set_trace()
        #if it is training phrase,then use ground truth bboxes for refining???
        if self.training:
            #import ipdb
            #ipdb.set_trace()
            roi_data = self.RCNN_proposal_target(rois,gt_boxes,num_boxes)
            rois,rois_label,gt_assign,rois_target,rois_inside_ws,rois_outside_ws = roi_data
            #rois,rois_label,rois_target,rois_inside_ws,rois_outside_ws = roi_data

            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]
            
            
            
            #################################################
            #################################################
            #Edited by fangfang
            #rois_label = Variable(rois_label.view(-1).long())
            #rois_target = Variable(rois_target.view(-1,rois_target.size(2)))

            rois = rois.view(-1,5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()

            
            pos_id = rois_label.nonzero().squeeze()
            gt_assign_pos = rois_label[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id

            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1,rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1,rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1,rois_outside_ws.size(2)))
            
            
            #rois_inside_ws = Variable(rois_inside_ws.view(-1,rois_inside_ws.size(2)))
            #rois_outside_ws = Variable(rois_outside_ws.view(-1,rois_outside_ws.size(2)))
        else:
            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]
            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1,5)
            pos_id = torch.arange(0,rois.size(0)).long().type_as(rois).long()
            rois_label_pos_ids = pos_id 
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        #rois = Variable(rois)

        #pooling feature based on rois,output 14*14 map
        #ipdb.set_trace()
        ##################################################################
        ####################not use feature pyramid#######################
        ##################################################################
        # if USE_ONE_FEATURE == 1:
        #     roi_pool_feat  = self.RCNN_roi_pool(base_feat,rois.view(-1,5),1.0/16.0)
        # else:
        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps,rois,im_info)

       
       
        #feed pooled features to top model
        pooled_feat = self._head_to_tail(roi_pool_feat)

        #compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # import ipdb
        # ipdb.set_trace()
        if self.training and not self.class_agnostic:
            #select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0),int(bbox_pred.size(1)/4),4)
            #bbox_pred_select = torch.gather(bbox_pred_view,1,rois_label.view(rois_label.size(0),1,1).expand(rois_label.size(0),1,4))
            bbox_pred_select = torch.gather(bbox_pred_view,1,rois_label.long().view(rois_label.size(0),1,1).expand(rois_label.size(0),1,4))

            bbox_pred = bbox_pred_select.squeeze(1)

        #compute object classification probability

        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_bbox = 0
        RCNN_loss_cls = 0

        if self.training:
            #loss (cross entropy) for object classification
            RCNN_loss_cls = F.cross_entropy(cls_score,rois_label)
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred,rois_target, rois_inside_ws,rois_outside_ws)

        rois = rois.view(batch_size,-1,rois.size(1))
        #cls_prob = cls_prob.view(batch_size,rois.size(1),-1)
        #bbox_pred = bbox_pred.view(batch_size,rois.size(1),-1)
        cls_prob = cls_prob.view(batch_size,-1,cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size,-1,bbox_pred.size(1))

        if self.training:
            rois_label = rois_label.view(batch_size,-1)
        
        return rois,cls_prob,bbox_pred,rpn_loss_cls,rpn_loss_bbox,RCNN_loss_cls,RCNN_loss_bbox,rois_label



