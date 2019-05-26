#coding:utf-8
import xml.etree.cElementTree as ET 
import os
from collections import Counter 
import shutil
import cv2
def count(pathdir,despath):
    instance_class = []
    img_class=[]
    area_hammer=[]
    area_bar=[]
    area_text=[]
    area_nest=[]
    path = pathdir+'Annotations/'
    import ipdb
    #ipdb.set_trace()
    xml_path = os.listdir(path)
    xml_path.sort()
    countind=0
    countimg=0
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
        img_temp=[]
        # ==================select images which has a special object=============
        for obj in objects:
            obj_label = obj.find('name').text
            #=obj.find('bndbox')
            
            #countind=countind+1
            if obj_label == 'nest' or obj_label == 'text' or obj_label == 'hammer' or obj_label == 'bar':#bar,nest,text
                print(xml)
                obj_height = int(obj.findtext('bndbox/ymax'))-int(obj.findtext('bndbox/ymin'))
                obj_width = int(obj.findtext('bndbox/xmax'))-int(obj.findtext('bndbox/xmin'))
                obj_area = obj_height*obj_width
                
                if obj_label == 'nest':
                    area_nest.append(obj_area)
                    despath ='data/transformer/nest/'
                elif obj_label == 'text':
                    area_text.append(obj_area)
                    despath ='data/transformer/text/'
                elif obj_label == 'hammer':
                    area_hammer.append(obj_area)
                    despath  ='data/transformer/hammer/'
                elif obj_label == 'bar':
                    area_bar.append(obj_area)
                    despath  ='data/transformer/bar/'
                else:
                    pass
                if not os.path.exists(despath):
                    os.makedirs(despath)
                img_despath = despath+xml.replace('xml','jpg')
                shutil.copyfile(imgfile,img_despath)
            
        instance_class +=[ob.find('name').text for ob in objects]
        img_temp +=[ob.find('name').text for ob in objects]
        img_temp = set(img_temp)
        img_class +=img_temp 
        
    print(Counter(instance_class))
    print(Counter(img_class))
    total_num_instances = sum([value for key,value in Counter(instance_class).items()])
    total_num_imgs = sum([value for key,value in Counter(img_class).items()])
    print('total_num_instances',total_num_instances)
    print('total_num_imgs',total_num_imgs)
    print('hammer mean area:',sum(area_hammer)//len(area_hammer))
    print('bar mean area:',sum(area_bar)//len(area_bar))
    print('text mean area:',sum(area_text)//len(area_text))
    print('nest mean area:',sum(area_nest)//len(area_nest))
    #print('total',countind,countimg)
if __name__ == '__main__':
    pathdir = 'data/VOCdevkit/VOC2007/'
    despath = 'data/transformer/'
    count(pathdir,despath)