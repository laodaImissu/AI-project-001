# Deployment of CNNs on Computing-In-Memory Architectures![image](https://github.com/user-attachments/assets/25dd6a42-001f-456b-8491-be236e8f68a2)
# train.py Vgg16Net.py 用于训练模型参数，Vgg16Net定义了一个基本的vgg16模型类，后续工作在此基础上修改。原文：https://blog.csdn.net/m0_50127633/article/details/117047057?spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-117047057-blog-123270587.235%5Ev43%5Epc_blog_bottom_relevance_base7&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-117047057-blog-123270587.235%5Ev43%5Epc_blog_bottom_relevance_base7&utm_relevant_index=17
# draw_convnet.py 用于绘制vgg16的结构，来自：https://github.com/gwding/draw_convnet 
# 其余文件均由本人独立完成。
# Vgg16Net2.py在Vgg16Net.py基础之上进行修改，将其中部分卷积层和线性层替换为模拟CIM算子的层。
# class_linear.py class_conv.py 分别定义了CIM架构的线性层和卷积层类。
# flops.py用于计算单一卷积层的FLOPs
# test_.py用于测试模型精度
# CIM_call_time.txt用于记录CIM marco的调用次数，在test_.py中进行记录。
# datahub.zip 是训练和测试用到的数据集。
