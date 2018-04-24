import tensorflow as tf
import numpy as np
import os
import re
# np.set_printoptions(threshold=100000)

'''
功能：获取训练数据地址，存储标签
参数：imagedir
imagedir: 训练数据地址
返回：imagelist, labellist
imagelist：图片位置列表
labellist：数据标签列表
'''


def get_file(imagedir):
    images = []
    labels = []
    for root, dirs, files in os.walk(imagedir):
        for filename in files:
            images.append(os.path.join(root, filename))  # 图片所在目录list
    for prefolder in images:
        letter = prefolder.split('\\')[-1]
        # print(letter)
        if re.match('0_', letter):  # 匹配图片名称
            labels = np.append(labels, [0])
        elif re.match('1_', letter):
            labels = np.append(labels, [1])
        elif re.match('2_', letter):
            labels = np.append(labels, [2])
        elif re.match('3_', letter):
            labels = np.append(labels, [3])
    temp = np.array([images, labels])  # 将图片地址和标记存入矩阵中
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)  # 打乱元素
    np.random.shuffle(temp)  # 打乱元素
    np.random.shuffle(temp)  # 打乱元素
    # print(temp)
    imagelist = list(temp[:, 0])  # 第一列的所有元素
    labellist = list(temp[:, 1])
    labellist = [int(float(i)) for i in labellist]  # 将标记转化为整形
    # print(labellist)
    return imagelist, labellist


'''
功能：获取训练数据地址，存储标签
参数：image_list, label_list, img_width, img_height, batch_size, capacity, channel
image_list: 图片位置列表
label_list：数据标签列表
img_width：训练图片size
img_height：训练图片size
batch_size：训练batch size
capacity：线程队列里面包含的数据数量
channel：输入数据通道数
返回：image_batch, label_batch
image_batch：图片batch
label_batch：数据标签batch
'''


def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity, channel):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    # 读入图片数据
    image_contents = tf.read_file(input_queue[0])
    # 解码图片为二进制
    image = tf.image.decode_jpeg(image_contents, channels=channel)
    # 统一图片数据尺寸
    image = tf.image.resize_images(image, (img_height, img_width))
    # 减均值 去除平均亮度值
    image = tf.image.per_image_standardization(image)
    # 随机队列
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=2,
                                                      min_after_dequeue=200,
                                                      capacity=capacity)
    # 重构标签shape
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch
