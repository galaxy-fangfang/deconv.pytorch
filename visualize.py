# coding: utf-8
 
from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers import Activation
from pylab import *
import keras
import argparse
import ipdb
from PIL import Image
from skimage import exposure
  

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row,col
 
def visualize_feature_map(img_batch):
    #
    print(img_batch.shape)
    feature_map = np.squeeze(img_batch,axis=0)
    print (feature_map.shape)
 
    feature_map_combination=[]
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row,col = get_row_col(num_pic)
 
    for i in range(0,num_pic):
        feature_map_split=feature_map[:,:,i]
        feature_map_combination.append(feature_map_split)
        #ipdb.set_trace()
        #feature_map_split = feature_map_split.convert("RGB")
        #plt.subplot(row,col,i+1)
        #plt.imshow(feature_map_split)

        #axis('off')
        #title('feature_map_{}'.format(i))
    #cv2.merge()
    
    #plt.savefig('feature_map_{}.jpg'.format(i))
    #plt.show()
 
    # 各个特征图按1：1 叠加
    #ipdb.set_trace()
    feature_map_sum = sum(ele for ele in feature_map_combination)
    feature_map_sum = exposure.rescale_intensity(feature_map_sum, out_range=(0, 255))
    feature_map_sum=np.uint8(feature_map_sum)
    feature_map_color = cv2.applyColorMap(feature_map_sum,cv2.COLORMAP_JET)
    #cv2.imshow("featuremap.jpg",feature_map_color)
    #cv2.waitKey(0)
    cv2.imwrite("featuremap_color.jpg",feature_map_color)
    #plt.imshow(feature_map_sum)
    #plt.savefig("feature_map_sum.jpg")
 
def create_model():
    model = Sequential()
 
    # 第一层CNN
    # 第一个参数是卷积核的数量，第二三个参数是卷积核的大小
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
 
    #第二层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
 
    # 第三层CNN
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    # 第四层CNN
    model.add(Convolution2D(9, 3, 3, input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    return model
 
 
if __name__ == "__main__":
    #args = parser.parse_args()
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')

    parser.add_argument('--img', dest='img',
                    help='inputimage', default="images/001000.jpg",
                    type=str)
    args=parser.parse_args()
    #print(args.img)
    img = cv2.imread(args.img)
 
    model = create_model()
 
    img_batch = np.expand_dims(img, axis=0)
    
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
 
    visualize_feature_map(conv_img)
