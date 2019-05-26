#coding:utf-8
from __future__ import division
import xml.etree.cElementTree as ET 
import os
from collections import Counter 
import shutil
import numpy as np
import cv2
import pickle
import sys
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import precision_recall_curve
from itertools import cycle

def count(pathdir,name):
    scales = []
    path = pathdir+'Annotations/'
    import ipdb
    #ipdb.set_trace()
    xml_path = os.listdir(path)
    xml_path.sort()
    #countind=0
    #countimg=0
    for index,xml in enumerate(xml_path):
        root = ET.parse(os.path.join(path,xml))
        objects = root.findall('object')
        imgfile = pathdir+'JPEGImages/'+xml.replace('xml','jpg')
        #img=cv2.imread(imgfile)
        #sizes=root.findall('size')
        img_height = int(root.findtext('./size/height'))
        img_width = int(root.findtext('./size/width'))
        img_area = img_width*img_height
        #countimg=countimg+1
        # ==================select images which has a special object=============
        for obj in objects:
            obj_label = obj.find('name').text
            #=obj.find('bndbox')
            
            #countind=countind+1
            if obj_label == 'nest' or obj_label == 'text' or obj_label == 'hammer' or obj_label == 'bar' or name == 'voc':#bar,nest,text
                print(xml)
                obj_height = float(obj.findtext('bndbox/ymax'))-float(obj.findtext('bndbox/ymin'))
                obj_width = float(obj.findtext('bndbox/xmax'))-float(obj.findtext('bndbox/xmin'))
                obj_area = obj_height*obj_width
                scale  = obj_area / img_area 
                print(scale)
                scales.append(scale)
            

    print(len(scales))
    sorted_ind = np.argsort(scales)
    sorted_scales = np.sort(scales)
    print(sorted_scales[:100])
    import ipdb
#    ipdb.set_trace()
    factors = np.ones(len(scales))
    factors = np.cumsum(factors)
    factors /= factors[-1]
    print(factors[:100])
   
    with open(os.path.join('scale_factor',name+'.pkl'), 'wb') as f:
       pickle.dump({'scales': sorted_scales, 'factors': factors}, f)
    

    
    #print('total',countind,countimg)
if __name__ == '__main__':
    dianwangpathdir = 'data/VOCdevkit/VOC2007/'
    vocpathdir = '/opt/data/fangfang/VOC/VOCdevkit/VOC2012_all/'
#    count(dianwangpathdir,'dianwang')
#    count(vocpathdir,'voc')
    files = os.listdir('scale_factor')
    files.sort()
    for file in files:
        with open(os.path.join('scale_factor',file),'rb') as f:
            data = pickle.load(f)
            scales = data['scales']
            factors = data['factors']
            method = file.split('.')[0]
            if method == 'dianwang':
                pl.plot(scales,factors,lw = 2, label = '{}'.format('Small Electrical Devices Dataset'))
                
                ##Most 
                index = 48300
                pl.plot([scales[index], scales[index]],[0,factors[index]], color = 'red', linewidth = 2.5, linestyle = '--')
                pl.plt.scatter([scales[index],], [factors[index],], 50, color = 'black')
                pl.plt.annotate('Most({:.4f},{:.4f})'.format(scales[index],factors[index]),xy = (scales[index],factors[index]),xycoords = 'data',xytext = (+10,-30),textcoords = 'offset points', fontsize = 8,
                        arrowprops = dict(arrowstyle = "->", connectionstyle = "arc3,rad=.2"))
                pl.plt.scatter([scales[25000],], [factors[25000],], 50, color = 'black')
                pl.plt.annotate('Median({:.4f},{:.4f})'.format(scales[25000],factors[25000]),xy = (scales[25000],factors[25000]),xycoords = 'data',xytext = (+10,+30),textcoords = 'offset points', fontsize = 8,
                        arrowprops = dict(arrowstyle = "->", connectionstyle = "arc3,rad=.2"))

            else:
                pl.plot(scales,factors,lw = 2, label = '{}'.format('PASCAL VOC Dataset'))
                ## Median
                index=20000
                pl.plot([scales[index], scales[index]],[0,factors[index]], color = 'red', linewidth = 2.5, linestyle = '--')
                pl.plt.scatter([scales[index],], [factors[index],], 50, color = 'black')
                pl.plt.annotate('Median({:.4f},0.5)'.format(scales[index]),xy = (scales[index],factors[index]),xycoords = 'data',xytext = (+10,-30),textcoords = 'offset points', fontsize = 8, arrowprops = dict(arrowstyle = "->", connectionstyle = "arc3,rad=.2"))

    pl.xlabel('Relative Scale')
    pl.ylabel('CDF(Scale)')
    plt.grid(True)
    pl.xlim([-0.05,1.0])
    pl.ylim([0.0,1.05])
    pl.legend(loc = 'lower right')
    plt.savefig('scales-factors.png')    
    plt.show()
