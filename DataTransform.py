# 图片尺寸转化:re_build(imagedir)
# 标记图片:make_labels(imagedir)
# TFRecord生成:make_tfrecord(imagelist, labellist, savedir, name)


import os
import tensorflow as tf
import numpy as np
import cv2 as cv
import re


'''
名称：图片尺寸转化
功能：剪裁图片同一大小
参数：dir:根文件夹位置
返回：无
'''
def re_build(imagedir):
    for root, dirs, files in os.walk(imagedir):
        for file in files:
            filepath = os.path.join(root, file)  # 获取文件地址
            try:
                image = cv.imread(filepath)  # 读取源图片
                dim = (227, 227)  # 目标尺寸
                reimage = cv.resize(image, dim)  # 重塑图片
                path = 'D:\\Test_Image\\train\\' + file  # 保存地址
                cv.imwrite(path, reimage)  # 保存图片
            except TypeError:  # 如果图片损坏 打印出地址
                print(filepath)
    cv.waitKey(0)


'''
名称：标记图片
功能：生成图片位置和标记 如：['D:\\Test_Image\\pre_action\\0_after100.jpg' '0.0']
参数：imagedir:图片所在目录
返回：imagelist：每张图片位置list
     labellist：每张图片对应标记list
'''
def make_labels(imagedir):
    images = []
    labels = []
    for root, dirs, files in os.walk(imagedir):
        for filename in files:
            images.append(os.path.join(root, filename))  # 图片所在目录list
    for prefolder in images:
        letter = prefolder.split('\\')[-1]
        if re.match('0_', letter):  # 匹配图片名称
            labels = np.append(labels, [0])
        else:
            labels = np.append(labels, [1])
    temp = np.array([images, labels])  # 将图片地址和标记存入矩阵中
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)  # 打乱元素
    print(temp)
    imagelist = list(temp[:, 0])  # 第一列的所有元素
    labellist = list(temp[:, 1])
    labellist = [int(float(i)) for i in labellist]  # 将标记转化为整形

    return imagelist, labellist


'''
返回TFRrecord的特征存储空间
'''
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


'''
名称：TFRrecord生成
功能：根据make_labels返回的数据生成TFRrecord
参数：imagelist：每张图片位置list
     labellist：每张图片对应标记list
     savedir:TFRrecord存储位置
     name：TFRrecord的名字
返回：无
'''
def make_tfrecord(imagelist, labellist, savedir, name):
    tfrname = os.path.join(savedir, name + '.tfrecords')  # 文件存储地址
    numsamples = len(labellist)  # 样本大小
    writer = tf.python_io.TFRecordWriter(tfrname)  # 创建writer对象
    print('\nSTART!!!!!!!!!')
    for i in range(0, numsamples):
        try:
            image = cv.imread(imagelist[i])  # 读取图片
            imbytes = image.tostring()  # 转化array为字符型 （存储要求
            label = int(labellist[i])  # 转化标记为整形
            example = tf.train.Example(features=tf.train.Features(feature={'label': int64_feature(label),
                                                                           'imbytes': bytes_feature(imbytes)}))
            #  装载图片与标记
            writer.write(example.SerializeToString())  # 数据序列化存储
        except IOError:  # 抓取图片错误
            print("I cant read this image:", imagelist[i])
    writer.close()
    print('OVER!!!!!!!!!!')


re_build('D:\\Test_Image\\source_image')
imagedata, labeldata = make_labels('D:\\Test_Image\\train\\')
make_tfrecord(imagedata, labeldata, '', 'train')
