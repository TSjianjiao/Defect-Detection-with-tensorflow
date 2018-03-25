# 自定义抗锯齿:smooth(image, threshold)
# 图像二值化:threshold_op(gray, mode)
# 形态学操作:mor_op(binary, mode, iterations)
# 获取轮廓:contours_op(binary, image)
# 仿射变换：rot_op(rect, image)
# 直方图均衡化：hist_equalization(gray, mode)
# 霍夫变换：hough_line(gray, image)
# 图片预处理：image_pre(imname)
# 摄像头捕获：video_capture()


import numpy as np
import cv2 as cv
import math
index = 1  # 图片存储索引


'''
名称：自定义抗锯齿
功能：二值化图像边缘平滑
参数：image：输入图像
      threshold：二值化阈值
返回：处理后的图像
'''
def smooth(image, threshold):
    height = image.shape[0]
    width = image.shape[1]
    for i in range(height-1):
        for j in range(width-1):
            if image[i, j] < threshold:
                image[i, j] = 255
            else:
                image[i, j] = 0
    cv.imshow('my_blur', image)
    return image


'''
名称：图像二值化
功能：输出二值化图像
参数：gray：灰度图
     mode：0：全局二值化 1：局部二值化
返回:结果图
'''
def threshold_op(gray, mode):
    if mode == 0:  # 全局二值化
        # ret, binary = cv.threshold(gray, 0, 255,
        #                            cv.THRESH_BINARY | cv.THRESH_OTSU)  # 计算二值化阈值
        ret, binary = cv.threshold(gray, 155, 255,
                                   cv.THRESH_BINARY)  # 手动阈值
    if mode == 1:  # 局部二值化（自适应二值化）
        binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY, 25, 10)
    cv.imshow("threshold", binary)
    return binary


'''
名称：形态学操作
功能：对二值化图像
参数：binary:二值化图像
     mode（str）:形态学方法 包括开闭操作 分水岭操作
     iterations：迭代次数
返回：形态学操作后图像
'''
def mor_op(binary, mode, iterations):
    if mode == 'close':
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # 定义结构元素
        sure = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=iterations)  # 闭操作 去除白噪点
        morimage = cv.bitwise_not(sure)  # 取反
        #  morimage = cv.bilateralFilter(morimage, 0, 100, 15)  # 高斯双边滤波
        morimage = cv.medianBlur(morimage, 15)  # 中值滤波
        # smooth(morimage, 155)
        cv.imshow('morimage', morimage)
    elif mode == 'open':
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # 定义结构元素
        morimage = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=4)  # 开操作 去除白噪点
        cv.imshow('mor_op', morimage)
    return morimage


'''
名称：获取轮廓
功能：在原图画出轮廓
参数：binary：二值图像
     image：原图
返回：矩形
'''
def contours_op(binary, image):
    cloneImage, contours, heriachy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 发现轮廓
    contour = sorted(contours, key=cv.contourArea, reverse=True)[1]  # 选取第二大的轮廓
    rect = cv.minAreaRect(contour)  # 外接矩形
    box = np.int0(cv.boxPoints(rect))  # 获取顶点
    cv.drawContours(image, [box], -1, (0, 0, 0), 2)  # 绘制矩形
    cv.imshow('contours', image)
    return rect


'''
名称：仿射变换
功能：图形旋转 是图片水平
参数：rect：找到的外接矩形
     image：需要旋转的图片
返回：旋转后的图片
'''
def rot_op(rect, image):
    # size = tuple(image.shape)[0:2]  # 截取元组前两个元素 也就是长宽
    size = (tuple(image.shape)[1], tuple(image.shape)[0])
    width = rect[1][0]
    height = rect[1][1]
    # print(rect[2])  # 打印与水平旋转的角度
    box = np.int0(cv.boxPoints(rect))  # 获取顶点
    # print(box)
    cv.imshow('no_rot', image)
    if width <= height:  # 判断长边位置
        angle = 90-abs(rect[2])
        rotation = cv.getRotationMatrix2D(tuple(box[1]), angle, 1)  # 计算旋转矩阵
        image = cv.warpAffine(image, rotation, size, flags=1)  # 仿射变换
    else:
        angle = rect[2]
        rotation = cv.getRotationMatrix2D(tuple(box[1]), angle, 1)  # 计算旋转矩阵
        image = cv.warpAffine(image, rotation, size, flags=1)  # 仿射变换
    cv.imshow('rot', image)
    return image


'''
名称：直方图均衡化
功能：输出直方图均衡化图像
参数：gray：灰度图
     mode：0：全局直方图均衡化 1：局部自适应均衡化
返回：结果图
'''
def hist_equalization(gray, mode):
    if mode == 0:
        dst = cv.equalizeHist(gray)  # 全局直方图均衡
    elif mode == 1:
        clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))  # 自己调整值 把图像分割为8X8的小块
        dst = clahe.apply(gray)
    cv.imshow("equalHist", dst)
    return dst


'''
名称：霍夫变换
功能：侦测目标直线
参数：gray:输入图片
     image：绘制直线的目标图片
返回：avg_theta：直线平均倾斜角
'''
def hough_line(gray, image):
    the_ta = 0  # θ迭代缓存
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # 边缘提取
    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)  # 霍夫变换提取直线
    # lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)  # 画直线
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    print(lines)
    for line in lines:  # 计算直线平均倾斜角
        print(type(lines))
        rho, theta = line[0]
        the_ta += math.degrees(theta)  # 转换θ为角度
    avg_theta = the_ta/lines.size
    # cv.imshow("hough_line", image)
    print(avg_theta)
    return avg_theta


'''
名称：图片预处理
功能：读取存储的图像 进行预处理（剪裁，转为灰度图，。。。）
参数：imname:图像名称
返回：无
'''
def image_pre(imname):
    global index  # 使用全局变量
    bias = 0  # ROI截取偏量
    image = cv.imread(imname)  # 读取原图
    blur = cv.GaussianBlur(image, (3, 3), 0)  # 高斯模糊去噪
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)  # 转化为灰度图
    imname = 'D:/Test_Image/pre_action/' + '1_after' + str(index) + '.jpg'  # 预处理后图片名
    cv.imshow('gray', gray)  # 显示灰度图

    '''仿射变换操作'''
    binary = threshold_op(gray, 0)  # 二值化
    morimage = mor_op(binary, 'close', 4)  # 闭操作 先膨胀 再腐蚀 迭代4次 去除噪点
    rect = contours_op(morimage, image)  # 获取轮廓
    rotim = rot_op(rect, gray)  # 仿射变换 使灰度图水平

    '''获取目标外接矩形'''
    rebinary = threshold_op(rotim, 0)  # 第二次二值化
    morimage = mor_op(rebinary, 'close', 4)  # 闭操作 先膨胀 再腐蚀 迭代4次 去除噪点
    # print('rot后的二值图尺寸', morimage.shape)
    rect_2 = contours_op(morimage, morimage)  # 获取轮廓
    # print('轮廓矩形尺寸', rect_2[1])
    box_2 = np.int0(cv.boxPoints(rect_2))  # 获取顶点
    # print('矩形四个顶点\n', box_2)
    w = int(rect_2[1][0])  # 矩形尺寸
    h = int(rect_2[1][1])
    center_x = int(rect_2[0][0])  # 矩形中心坐标
    center_y = int(rect_2[0][1])

    '''截取ROI'''
    remorimage = morimage[center_y - int(h/2)-bias:center_y + int(h/2)+bias,
                          center_x - int(w/2)-bias:center_x + int(w/2)+bias]  # 截取ROI 在二值图上
    # print('ROI尺寸', remorimage.shape)
    regray = rotim[center_y - int(h/2)-bias:center_y + int(h/2)+bias,
                   center_x - int(w/2)-bias:center_x + int(w/2)+bias]  # 截取ROI  在灰度图上
    cv.imshow('regray', regray)
    # remorimage = cv.circle(morimage, (center_x, center_y), 2, (0, 0, 0))  # 画出中心点
    cv.imshow('remorimage', remorimage)
    reimage = cv.bitwise_and(remorimage, regray)  # 将提取的mask与上原来的灰度图
    cv.imshow('after', reimage)
    cv.imwrite(imname, reimage)  # 保存处理后的图片
    print(index)
    index += 1  # 准备下一张图


'''
名称：捕获摄像头图像
功能：获取摄像头图像，按下回车键存储当前帧，按下esc退出
参数：无
返回：无
'''
def video_capture():
    capture = cv.VideoCapture(0)  # 选择哪一个摄像头 0、1、2、、、、
    while True:  # 死循环捕捉图像帧 不断更新以达到动画
        ret, frame = capture.read()
        frame = cv.flip(frame, -1)  # flip函数调整画面镜像
        cv.imshow("video", frame)
        c = cv.waitKey(50) & 0xFF  # 64位系统需要加上&0xFF
        if c == 27:  # 27是esc键的键值 只有按下才退出
            break
        elif c == 13:  # 13是回车键的键值 按下回车键截取一张图片
            imname = 'D:/Test_Image/source_image/' + '1_src' + str(index) + '.jpg'
            cv.imwrite(imname, frame)  # 保存当前帧
            image_pre(imname)  # 图像预处理


# image_pre(imname='D:/Test_Image/source_image/' + '0_src1' + '.jpg')
# cv.waitKey(0)
video_capture()


