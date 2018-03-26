# Defect-Detection-with-tensorflow
# 本科毕业设计：
  主要目的是做一个基于图像的胶囊表面缺陷检测，结合opencv和图像处理技术，再通过深度学习训练个模型达到效果

  目前正在补充图像处理，机器学习，深度学习等相关知识

# 完成了数据收集部分：
  VideoPart.py：获取摄像头图像 并进行图片预处理
  
  DataTransform.py：将图片和标记生成TFRecord文件
  
  ReadFile.py：读取tfrecord文件并包含转换标签为one-hot编码的方法

