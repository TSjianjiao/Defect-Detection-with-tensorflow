import tensorflow as tf
import numpy as np


'''
名称：读取tfrecord
功能：批量读取tfrecord文件
参数：tfrecordfile
     batchsize
'''
def read_tfrecord(tfrecordfile, batchsize, random_crop=False, random_clip=False):
    filequeue = tf.train.string_input_producer([tfrecordfile])  # 将tfrecordfile读入队列
    reader = tf.TFRecordReader()  # 创建reader对象
    ret, example = reader.read(filequeue)  # 读取序列化对象
    imfeatures = tf.parse_single_example(example, features={'label': tf.FixedLenFeature([], tf.int64),
                                                            'imbytes': tf.FixedLenFeature([], tf.string)})
    #  转化tfrecord内容 将数据还原为输入的数据格式
    image = tf.decode_raw(imfeatures['imbytes'], tf.uint8)  # 将string转化为unint8的张量
    image = tf.reshape(image, [227, 227, 3])  # 调整图片大小
    '''
    数据增强部分 随机剪裁 左右反转
    '''
    if random_crop:
        image = tf.random_crop(image, [227, 227, 3])
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, 227, 227)
    if random_clip:
        image = tf.image.random_flip_left_right(image)

    label = tf.cast(imfeatures['label'], tf.int32)  # 转化lable为32位整形
    imbatch, labatch = tf.train.shuffle_batch([image, label],
                                              batch_size=batchsize,
                                              min_after_dequeue=100,
                                              num_threads=4,
                                              capacity=200)  # 随机读取数据
    # capacity：队列一次性放的样本
    # num_threads：使用多少个线程操作
    # min_after_dequeue ：队列维持的最小长度
    print()
    return imbatch, tf.reshape(labatch, [batchsize])


'''
名称：标签格式重构
功能：将标签重构为one-hot编码
参数：labels：输入标签
返回：ohlables：one-hot编码的标签
eg:[1, 1, 0, 1, 2, 3]
--->one-hot
--->
一个矩阵每一行的1的索引代表标签值
|0. 1. 0. 0.|-->1
|0. 1. 0. 0.|-->1
|1. 0. 0. 0.|-->0
|0. 1. 0. 0.|-->1
|0. 0. 1. 0.|-->2
|0. 0. 0. 1.|-->3
'''
def to_one_hot(labels):
    numsample = len(labels)  # 行数
    numclass = max(labels) + 1  # 列数 lable最大值加一
    ohlabels = np.zeros((numsample, numclass))  # 创建空矩阵
    ohlabels[np.arange(numsample), labels] = 1  # 在对应位置 置1
    return ohlabels

