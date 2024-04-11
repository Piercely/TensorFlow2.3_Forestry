# 基于tensorflow2.3的土壤识别系统

### 代码结构
```
images 目录主要是放置一些图片，包括测试的图片和ui界面使用的图片
models 目录下放置训练好的两组模型，分别是cnn模型和mobilenet的模型
results 目录下放置的是训练的训练过程的一些可视化的图，两个txt文件是训练过程中的输出，两个图是两个模型训练过程中训练集和验证集准确率和loss变化曲线
utils 是主要是我测试的时候写的一些文件，对这个项目没有实际的用途
get_data.py 爬虫程序，可以爬取百度的图片
window.py 是界面文件，主要是利用pyqt5完成的界面，通过上传图片可以对图片种类进行预测
testmodel.py 是测试文件，主要是用于测试两组模型在验证集上的准确率，这个信息你从results的txt的输出中也能获取
train_cnn.py 是训练cnn模型的代码
train_mobilenet.py 是训练mobilenet模型的代码
requirements.txt 是本项目需要的包
```

> ![img](https://img-blog.csdnimg.cn/direct/16df0128c2fc4761be10983e3e8c7ec5.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)
>
> 大抵是不用进厂了罢。





------

![img](https://img-blog.csdnimg.cn/837b11390fd2464cb748a84aef98ffc4.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

------

# 一、🌎Python简介

Python是一种高级、解释性、面向对象的编程语言。它具有简洁的语法和易于阅读的代码风格，使得它成为初学者和专业开发人员的首选语言之一。

Python具有广泛的应用领域，包括Web开发、数据分析、人工智能、科学计算、网络编程等。它拥有强大的第三方库和工具生态系统，如NumPy、Pandas、Matplotlib和TensorFlow，使得开发人员能够快速构建复杂的应用程序。

Python还是一种跨平台的语言，可以在多个操作系统上运行，如Windows、Linux和MacOS。

由于其易学易用的特性，Python已经成为编程教育的主流语言之一。许多大学和学校都将Python作为入门级编程语言进行教学。

总体而言，Python是一种功能强大、易于学习和使用的编程语言，适用于各种应用场景，并且在业界有着广泛的应用和支持。

------



# 二、🌎TensorFlow简介

TensorFlow 是由 Google 团队开发的深度学习框架之一，它是一个完全基于 Python 语言设计的开源的软件。TensorFlow 的初衷是以最简单的方式实现机器学习和深度学习的概念，它结合了**计算代数**的优化技术，使它便计算许多数学表达式。

TensorFlow 可以训练和运行**深度神经网络**，它能应用在许多场景下，比如，图像识别、手写数字分类、递归神经网络、单词嵌入、自然语言处理、视频检测等等。TensorFlow 可以运行在多个 CPU 或 GPU 上，同时它也可以运行在移动端操作系统上（如安卓、IOS 等），它的架构灵活，具有良好的可扩展性，能够支持各种网络模型（如OSI七层和TCP/IP四层）。

TensorFlow 官网（[https://tensorflow.google.cn/](https://link.zhihu.com/?target=https%3A//tensorflow.google.cn/)）提供了 TensorFlow 的官方学习文档以及最新版本的下载方式。

TensorFlow 这个词由 Tensor 和 Flow 两个词组成，这两者是 TensorFlow 最基础的要素。Tensor 代表张量（也就是数据），它的表现形式是一个多维数组；而 Flow 意味着流动，代表着计算与映射，它用于定义操作中的数据流。

tensorflow2.x版本对小白非常友好，2.x的api中对[keras](https://so.csdn.net/so/search?q=keras&spm=1001.2101.3001.7020)进行了合并，大家只需要安装tensorflow就可以使用里面封装好的keras，利用keras可以快速地加载数据集和构建模型，下面我们直接来看以下通过tensorflow2.3训练自己的分类数据集吧。

------

 

# 三、🌎效果演示

**通过Pyqt5来构建图形化界面**

![img](https://img-blog.csdnimg.cn/direct/e8d8582199bb487da0aed77424d843d7.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

 

 **通过flask构建网站页面**

![img](https://img-blog.csdnimg.cn/direct/8a568659903548d9b19873417ce8bc3f.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

![img](https://img-blog.csdnimg.cn/direct/140cbf8d79b44b6da0f11cefe2e82d12.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑 

------

# 四、🌎数据集整理

我们可以通过如下代码实现数据集的快速搜集！

```python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/11 20:29
# @Author  : Enovo
# @File    : get_data.py
# @Software: PyCharm
# @Brief   : 爬取所需要的数据集图片

import requests
import re
import os

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36'}
name = input('请输入要爬取的图片类别：')
num = 0
num_1 = 0
num_2 = 0
x = input('请输入要爬取的图片数量？（1等于60张图片，2等于120张图片）：')
list_1 = []
for i in range(int(x)):
    name_1 = os.getcwd()
    name_2 = os.path.join(name_1, 'data/' + name)
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + name + '&pn=' + str(i * 30)
    res = requests.get(url, headers=headers)
    htlm_1 = res.content.decode()
    a = re.findall('"objURL":"(.*?)",', htlm_1)
    if not os.path.exists(name_2):
        os.makedirs(name_2)
    for b in a:
        try:
            b_1 = re.findall('https:(.*?)&', b)
            b_2 = ''.join(b_1)
            if b_2 not in list_1:
                num = num + 1
                img = requests.get(b)
                f = open(os.path.join(name_1, 'data/' + name, name + str(num) + '.jpg'), 'ab')
                print('---------正在下载第' + str(num) + '张图片----------')
                f.write(img.content)
                f.close()
                list_1.append(b_2)
            elif b_2 in list_1:
                num_1 = num_1 + 1
                continue
        except Exception as e:
            print('---------第' + str(num) + '张图片无法下载----------')
            num_2 = num_2 + 1
            continue

print('下载完成,总共下载{}张,成功下载:{}张,重复下载:{}张,下载失败:{}张'.format(num + num_1 + num_2, num, num_1, num_2))
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

 **整理数据集**

放置到相应的子文件夹
 数据集收集完成之后，我们还需要对数据集进行整理，如果是爬虫爬取的图片可能会有一些质量比较差的图片，那么整理之前还需要进行数据的清洗，删除质量不好的图片，数据集整理其实很简单，我们只需要将数据集进行归类即可，即相同类别的图片放在一个文件夹下，比如下面的这个数据集，百合的文件夹下放的全是百合的图片，水仙的文件夹下则放的全是水仙的图片。

 

![img](https://img-blog.csdnimg.cn/direct/173d3a1eacfc400e89325cd08ee9d20d.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

![img](https://img-blog.csdnimg.cn/direct/18d9d9a6cc9041558e7bdbdee58186b7.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

![img](https://img-blog.csdnimg.cn/direct/2ec97e40d68d4c809ab8dfb059603015.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

------

 

### 划分训练集和测试集

注：如果是使用的开源数据集，开源数据集可能已经进行了数据集的划分，直接使用即可，不需要再次进行划分，比如这里是我下载到的农作物病虫害的数据集，已经分别提供了训练集、测试集和验证集，就不需要再次进行数据集的划分。

为了方便我们进行数据集的加载，我们还需要将图片划分为训练集和测试集，如果需要的话你还需要划分出验证集，验证集在一般的任务中是可选的，因为是自己收集的数据集的话，数据量比较少，如果再划分验证集的话可能会导致训练量不够，这里我写了一段数据集划分的代码逻辑，大家输入原始的数据集位置和划分之后的数据集位置，指定数据集划分的比例，即可完成数据集的划分。



![img](https://img-blog.csdnimg.cn/direct/edbc536f702e4b368a8bb70973da08b3.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

**数据集划分代码：** 

```python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/11 20:29
# @Author  : Enovo
# @File    : data_split.py
# @Software: PyCharm
# @Brief   : 将数据集划分为训练集、验证集和测试集
import os
import random
from shutil import copy2


def data_set_split(src_data_folder, target_data_folder, train_scale=0.7, val_scale=0.2, test_scale=0.1):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 
    :param target_data_folder: 目标文件夹 
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                # print("{}复制到了{}".format(src_img_path, train_folder))
                train_num = train_num + 1
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                copy2(src_img_path, val_folder)
                # print("{}复制到了{}".format(src_img_path, val_folder))
                val_num = val_num + 1
            else:
                copy2(src_img_path, test_folder)
                # print("{}复制到了{}".format(src_img_path, test_folder))
                test_num = test_num + 1

            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))
        print("测试集{}：{}张".format(test_folder, test_num))


if __name__ == '__main__':
    src_data_folder = "./data"   # todo 修改你的原始数据集路径
    target_data_folder = "./new_data"  # todo 修改为你要存放的路径
    data_set_split(src_data_folder, target_data_folder)
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

------

 

# 五、🌎环境搭建

**Python3.7+TensorFlow2.3**

**此处可以根据电脑选择安装GPU或者CPU版本**

本次教程需要大家实现配置好python的环境，我们需要使用到anaconda和pycharm，不熟悉环境配置的同学可以看我得这篇博客，我在这里就不再进行赘述了。

[深度学习环境配置超详细教程【Anaconda+Pycharm+PyTorch(GPU版)+CUDA+cuDNN】![img](https://csdnimg.cn/release/blog_editor_html/release2.3.6/ckeditor/plugins/CsdnLink/icons/icon-default.png?t=N7T8)http://t.csdnimg.cn/ZT91i](http://t.csdnimg.cn/ZT91i)

### 训练模型


 模型训练的代码种，以 cnn 模型的训练为例，**train_cnn.py** 是训练cnn模型的代码，只需要修改三处即可，如下所示

![img](https://img-blog.csdnimg.cn/direct/e6afb15f481449b3a9b5ebf4745c5b56.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

**train_mobilnet.py** 是训练 **mobilenet** 模型的代码，训练的模型将会保存在 **models** 目录下，这里也是只需修改三处即可。

![img](https://img-blog.csdnimg.cn/direct/7069acdde4c740f5a91e60ab35d75f2e.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

**注：代码最后一行的epochs指的是跑的训练的轮数，这里默认是30，大家可以根据自己的需要增加或减少训练的轮数**

------

修改之后直接运行即可，等代码跑完后模型就会保存在models目录下

 ![img](https://img-blog.csdnimg.cn/direct/6a9e3ba2944c4869b03d4631d0926603.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

另外，在results目录下你可以找到模型训练的过程图

![img](https://img-blog.csdnimg.cn/direct/9569aeb53fa04fe08890895f149b5245.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

模型训练的过程中会输出数据集的 **类名**，这里记录一下，在后面的模型使用中会用到。

 

测试模型
 模型的测试的代码为test_model.py，也是只需要改动几处代码即可完成测试 

改动如下：

![img](https://img-blog.csdnimg.cn/direct/b8f42fc887504ee5815d08a060655d89.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

测试的基本流程是：加载数据、加载模型、测试、保存结果

测试之后在命令行中会输出每个模型的准确率，并且会在results目录下生成相应的热力图
 热力图中对应了每个类别的准确率，如下所示，是mobilenet测试的热力图。

![img](https://img-blog.csdnimg.cn/direct/5a45fcd131bb4c26bd3b3a6dae92a3c1.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

使用模型
 模型的时候中，我们通过Pyqt5来构建图形化界面，用户可以上传图片，并在系统中调用我们训练好的模型进行图片类别的预测。

在window.py代码中修改四处即可完成基本功能，如下：

![img](https://img-blog.csdnimg.cn/direct/1c807135ce9543de97e5a97ee82aa481.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

此外我们增加了Flask网站页面显示，只需在 app.py 启动即可，并修改如下：

![img](https://img-blog.csdnimg.cn/direct/b26f85dc14ed4f00a04bd939e8966fd2.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

启动成功！！！

------

 

# 六、获取代码

 

正确的代码文件及路径，见下图：

![img](https://img-blog.csdnimg.cn/direct/2927c31040844e939d205d36348c596f.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)



第一步，下载源码压缩包，解压并打开文件夹，即为上图样式；

第二步，创建环境，我使用的是 **anaconda python3.7**；

第三步，打开 pycharm 导入项目，点击 **app.py 或者 ui.py** 文件运行；

以上就是我们此次TensorFlow实践作业的全部内容了，是否精彩呢？如果有好的建议或者想法可以联系我，一起交流🙇‍；
 ————————————————

------

至此，本篇文章就已经全部结束了，感谢大家的观看。

已许久许久许久……未更新。

忙于考试。

**加油加油加油！！！**

/(ㄒoㄒ)/~~

------

# **🥇Summary**

> 上述内容就是此次  的全部内容了，感谢大家的支持，相信在很多方面存在着不足乃至错误，希望可以得到大家的指正。🙇‍(ง •_•)ง

> 我非轻舟

> 2024年第四期，继续加油！！！

> **希望大家有好的意见或者建议，欢迎私信，一起加油**

------

**以上就是本篇文章的全部内容了**

 **~ 关注我，点赞博文~ 每天带你涨知识!**



1.看到这里了就 **[点赞+好评+收藏]** 三连 支持下吧，你的「点赞，好评，收藏」是我创作的动力。

2.关注我 ~ 每天带你学习 **:各种前端插件、3D炫酷效果、图片展示、文字效果、以及整站模板 、HTML模板 、C++、数据结构、Python程序设计、Java程序设计、爬虫等！ 「在这里有好多 开发者，一起探讨 前端 开发 知识，互相学习」！**

3.以上内容技术相关问题可以相互学习，可 **关 注 ↓公 Z 号 获取更多源码 !**


# 获取源码？私信？关注？点赞？收藏？WeChat?

> 👍+✏️+⭐️+🙇‍

有需要源码的小伙伴可以 关注下方微信公众号 **" Enovo开发工厂 "**🙇‍ 
