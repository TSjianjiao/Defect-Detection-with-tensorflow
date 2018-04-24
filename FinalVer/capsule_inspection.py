import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import Data_Trans as DT
import os
import cv2 as cv
import tensorflow.contrib.layers as layers
import re

# 输入数据尺寸
imw = 224
imh = 224
# 输入数据通道
cn = 3
# bn层flag 训练的时候True 测试的时候False
flg = True
batch_size = 100

# 获取数据和标签list
X_train, y_train = DT.get_file("D:\\Image\\train\\")
image_batch, label_batch = DT.get_batch(X_train, y_train, imw, imh, batch_size, 1000, cn)


def batch_norm(inputs, is_training, is_conv_out=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, 0.001)


def onehot(labels):
    n_sample = len(labels)
    n_class = n_classes
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


def relu(name, input_data):
    out = tf.nn.relu(input_data, name)
    return out


def conv(name, input_data, out_channel, kernel_h, kernel_w, stride_h, stride_w, padding="SAME"):
    print(str(name)+str(input_data.get_shape()))
    in_channel = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = tf.get_variable("weights", [kernel_h, kernel_w, in_channel, out_channel], dtype=tf.float32)
        biases = tf.get_variable("biases", [out_channel], dtype=tf.float32)
        conv_res = tf.nn.conv2d(input_data, kernel, [1, stride_h, stride_w, 1], padding=padding)
        out = tf.nn.bias_add(conv_res, biases)
    return out


def maxpool(name, input_data, kernel_h, kernel_w, stride_h, stride_w):
    print(str(name)+str(input_data.get_shape()))
    out = tf.nn.max_pool(input_data, [1, kernel_h, kernel_w, 1],
                         [1, stride_h, stride_w, 1], padding="SAME", name=name)
    return out


def avg_pool(name, input_data, kernel_h, kernel_w, stride_h, stride_w):
    print(str(name)+str(input_data.get_shape()))
    return tf.nn.avg_pool(input_data, [1, kernel_h, kernel_w, 1], [1, stride_h, stride_w, 1],
                          padding="VALID", name=name)


def fully_connected(input_data, out_calss, activation):
    return layers.fully_connected(input_data, out_calss, activation_fn=activation)


def inception(name, input_d, out_11,
              out_33_reduce, out_33,
              out_55_reduce, out_55,
              out_pool_reduce):

    # 1x1
    inception_1_1 = conv(name + 'inception_1_1', input_d, out_11, 1, 1, 1, 1)
    inception_1_1 = relu("relu", inception_1_1)

    # 3x3
    inception_3_3_reduce = conv(name + 'inception_3_3_reduce', input_d, out_33_reduce, 1, 1, 1, 1)
    inception_3_3_reduce = relu("relu", inception_3_3_reduce)

    inception_3_3 = conv(name + 'inception_3_3', inception_3_3_reduce, out_33, 3, 3, 1, 1)
    inception_3_3 = relu("relu", inception_3_3)

    # 5x5
    inception_5_5_reduce = conv(name + 'inception_5_5_reduce', input_d, out_55_reduce, 1, 1, 1, 1)
    inception_5_5_reduce = relu("relu", inception_5_5_reduce)

    inception_5_5 = conv(name + 'inception_5_5', inception_5_5_reduce, out_55, 5, 5, 1, 1)
    inception_5_5 = relu("relu", inception_5_5)

    # pool
    inception_pool = maxpool(name + 'inception_pool', input_d, 3, 3, 1, 1)
    inception_pool_1_1 = conv(name + 'inception_pool_1_1', inception_pool, out_pool_reduce, 1, 1, 1, 1)
    inception_pool_1_1 = relu("relu", inception_pool_1_1)

    return tf.concat([inception_1_1, inception_3_3, inception_5_5, inception_pool_1_1], 3)


# 模型参数
learning_rate = 0.0001
n_classes = 4

# 构建模型
x = tf.placeholder(tf.float32, [None, imh, imw, cn])
y = tf.placeholder(tf.int32, [None, n_classes])

# conv1
conv1 = conv('conv1', x, 64, 7, 7, 2, 2)
conv1 = relu("relu", conv1)

# max pool 1
conv1 = maxpool('max_pool1', conv1, 3, 3, 2, 2)
# conv2_reduce
conv2_reduce = conv('conv2_reduce', conv1, 64, 1, 1, 1, 1)
conv2_reduce = relu("relu", conv2_reduce)

# conv2
conv2 = conv('conv2', conv2_reduce, 192, 3, 3, 1, 1)
conv2 = relu("relu", conv2)
# max pool 2
conv2 = maxpool('max_pool2', conv2, 3, 3, 2, 2)

# inception 1a
inception_1ab= inception('inception_1a', conv2, 64, 96, 128, 16, 32, 32)
# inception 1b
inception_1b = inception('inception_1b', conv2, 128, 128, 192, 32, 96, 64)

# 全局平均池化
out_data = avg_pool("avgpool", inception_1b, 28, 28, 1, 1)
print(out_data)
out_data = tf.reshape(out_data, [-1, 1*1*480])

# liner
result = fully_connected(out_data, n_classes, activation=None)
pre = tf.nn.softmax(result)

# 定义损失
# l2正则化
# tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_conv['conv1'])
# tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_conv['conv2'])
# regularizer = layers.l2_regularizer(scale=0.01)
# reg_term = layers.apply_regularization(regularizer)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=y))
# 优化函数 自适应梯度下降
optimizer = tf.train.AdamOptimizer().minimize(loss)

# 评估模型
correct_pred = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 变量初始化
init = tf.global_variables_initializer()

# 模型存储位置
save_model = "./model/Model"
# GPU设置
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666, allow_growth=True)


def train(opech, is_continnue_train=False):
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        # train_writer = tf.summary.FileWriter(".//log", sess.graph)  # 输出日志的地方
        # 模型保存
        saver = tf.train.Saver()

        loss_temp = []  # loss值存数列表
        acc_temp = []  # 准确度存储列表
        start_time = time.time()  # 记录时间

        # 启动线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # 判断是不是继续训练
        if is_continnue_train:
            # 装载checkpoint
            savemodel = tf.train.latest_checkpoint('./model/')
            saver.restore(sess, savemodel)
        for i in range(opech):
            str_ing = 'lr=' + str(learning_rate) + '迭代=' + str(opech)

            image, label = sess.run([image_batch, label_batch])  # 获取输入数据和标签
            labels = onehot(label)  # 将标签转为one hot 格式

            sess.run(optimizer, feed_dict={x: image, y: labels})  # 启动优化函数
            loss_record, acc, c1, c2 = sess.run([loss,
                                                accuracy,
                                                conv1,
                                                conv2],
                                                feed_dict={x: image, y: labels})
            print("now the loss is %f " % loss_record)
            print("now the acc is %f " % acc)
            loss_temp.append(loss_record)
            acc_temp.append(acc)
            # 每100次训练保存一次输出图表
            if i % 100 == 0:
                saver.save(sess, save_model)
                plt.plot(loss_temp, 'b-', label='loss')
                plt.plot(acc_temp, 'r-', label='accuracy')
                plt.legend(loc='upper right')
                plt.xlabel('Iter')
                plt.ylabel('loss/acc')
                plt.title('lr=%f, ti=%d' % (learning_rate, opech))
                plt.tight_layout()
                plt.savefig('M2_' + str_ing + '.jpg', dpi=200)
                # 卷积层可视化
                # plt.subplot(4, 5, 3)
                # plt.imshow(image[2, :, :, 0])
                # plt.colorbar()
                # plt.axis('off')
                # plt.title("src image")
                # for ax_1 in range(4):
                #     con1 = [10, 29, 38, 47]
                #     plt.subplot(4, 4, 5+ax_1)
                #     plt.imshow(in_1a[0][2, :, :, ax_1+con1[ax_1]])
                #     plt.colorbar()
                #     plt.axis('off')
                #     plt.title("1x1 "+'channel:'+str(ax_1+con1[ax_1]))
                # for ax_2 in range(4):
                #     con2 = [30, 49, 97, 125]
                #     plt.subplot(4, 4, 9 + ax_2)
                #     plt.imshow(in_1[1][2, :, :, ax_2+con2[ax_2]])
                #     plt.colorbar()
                #     plt.axis('off')
                #     plt.title("3x3 " + 'channel:'+str(ax_2+con2[ax_2]))
                # for ax_3 in range(4):
                #     con3 = [5, 8, 13, 27]
                #     plt.subplot(4, 4, 13 + ax_3)
                #     plt.imshow(in_1[2][2, :, :, ax_3+con3[ax_3]])
                #     plt.colorbar()
                #     plt.axis('off')
                #     plt.title("5x5 " + 'channel:' + str(ax_3+con3[ax_3]))
                # plt.show()

                # global flg
                # # 测试
                # flg = False
                # for root, dirs, files in os.walk('D:/Image/mini_test/'):
                #     aa = 0
                #     bb = 0
                #     cc = 0
                #     dd = 0
                #     for filename in files:
                #         filepath = os.path.join(root, filename)  # 获取文件地址
                #         image_contents = tf.read_file(filepath)
                #         image = tf.image.decode_jpeg(image_contents, channels=cn)
                #         image = tf.image.resize_images(image, (imh, imw))
                #         image = tf.image.per_image_standardization(image)  # 减均值
                #         image = tf.expand_dims(image, axis=0)
                #         image = sess.run(image)
                #         prediction = sess.run(pre, feed_dict={x: image})
                #         # print('pre:', prediction)
                #         maxindex = np.argmax(prediction, 1)
                #         letter = filepath.split('/')[-1]
                #         if maxindex == 0:
                #             if re.match('0_', letter):
                #                 aa += 1
                #                 # print('aa: '+filepath)
                #         elif maxindex == 1:
                #             if re.match('1_', letter):
                #                 bb += 1
                #                 # print('bb: '+filepath)
                #         elif maxindex == 2:
                #             if re.match('2_', letter):
                #                 cc += 1
                #         elif maxindex == 3:
                #             if re.match('3_', letter):
                #                 dd += 1
                # print('完整20: ' + str(aa) + ' ' + str((aa/20)*100) + '%')
                # print('凹陷20: ' + str(bb) + ' ' + str((bb/20)*100) + '%')
                # print('帽子20: ' + str(cc) + ' ' + str((cc/20)*100) + '%')
                # print('身体20: ' + str(dd) + ' ' + str((dd/20)*100) + '%')
                # flg = True


            end_time = time.time()
            print('time: ', (end_time - start_time))
            start_time = end_time

            print("---------------%d onpech is finished-------------------" % i)
        print("Optimization Finished!")
        saver.save(sess, save_model)
        print("Model Save Finished!")
        coord.request_stop()
        coord.join(threads)
        plt.plot(loss_temp, 'b-', label='loss')
        plt.plot(acc_temp, 'r-', label='accuracy')
        plt.legend(loc='upper right')
        plt.xlabel('Iter')
        plt.ylabel('loss/acc')
        plt.title('lr=%f, ti=%d' % (learning_rate, opech))
        plt.tight_layout()
        plt.savefig('M2_' + str_ing + '.jpg', dpi=200)
        plt.show()


def tes_part(imagedir):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        st_time = time.time()
        savemodel = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, savemodel)
        image_contents = tf.read_file(imagedir)
        image = tf.image.decode_jpeg(image_contents, channels=cn)
        image = tf.image.resize_images(image, (imh, imw))
        image = tf.image.per_image_standardization(image)  # 减均值
        image = tf.expand_dims(image, axis=0)
        image = sess.run(image)
        prediction = sess.run(pre, feed_dict={x: image})
        end_time = time.time()
        print('time: ', (end_time - st_time))
        st_time = end_time
        print(prediction)
        maxindex = np.argmax(prediction)
        if maxindex == 0:
            return '完整'
        elif maxindex == 1:
            return '凹陷'
        elif maxindex == 2:
            return '帽子'
        elif maxindex == 3:
            return '身体'

# 摄像头拍照并预测
def video_test():
    capture = cv.VideoCapture(0)  # 选择哪一个摄像头 0、1、2、、、、
    index = 0
    while True:  # 死循环捕捉图像帧 不断更新以达到动画
        ret, frame = capture.read()
        cv.namedWindow('Video')
        cv.resizeWindow('Video', 227, 227)
        cv.imshow('Video', frame)
        c = cv.waitKey(50) & 0xFF  # 64位系统需要加上&0xFF

        if c == 27:  # 27是esc键的键值 只有按下才退出
            break
        elif c == 65:  # 13是回车键的键值 按下回车键截取一张图片
            imname = 'D:\\Image\\test\\' + '0_' + str(index) + '.jpg'
            frame = frame[0:227, 0:227]
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            cv.imwrite(imname, frame)  # 保存当前帧
            index += 1
            print(tes_part(imname))
        elif c == 66:
            imname = 'D:\\Image\\test\\' + '1_' + str(index) + '.jpg'
            frame = frame[0:227, 0:227]
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            cv.imwrite(imname, frame)  # 保存当前帧
            index += 1
            print(tes_part(imname))
        elif c == 67:
            imname = 'D:\\Image\\test\\' + '2_' + str(index) + '.jpg'
            frame = frame[0:227, 0:227]
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            cv.imwrite(imname, frame)  # 保存当前帧
            index += 1
            print(tes_part(imname))
        elif c == 68:
            imname = 'D:\\Image\\test\\' + '3_' + str(index) + '.jpg'
            frame = frame[0:227, 0:227]
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            cv.imwrite(imname, frame)  # 保存当前帧
            index += 1
            print(tes_part(imname))


train(400, is_continnue_train=True)


# 文件夹所有内容测试
# for root, dirs, files in os.walk('D:/Image/test/'):
#     for filename in files:
#         filepath = os.path.join(root, filename)  # 获取文件地址
#         print(filepath, tes_part(filepath))


# video_test()
