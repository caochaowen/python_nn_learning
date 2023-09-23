import numpy
import matplotlib.pyplot as plt
import scipy.special
import os


# 神经网络对象的搭建
class neuralNetwork:
    # 神经网络参数初始化,设置输入层，隐藏层，输出层的节点数量，定义神经网络的大小
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 形参实参赋值
        self.innodes = inputnodes
        self.hidnodes = hiddennodes
        self.outnodes = outputnodes
        self.lr = learningrate
        # 设置层之间的节点矩阵，初始数据为随机值,矩阵的生成利用numpy方法
        # wih为1~2之间的矩阵，who为2~3之间的矩阵
        # self.wih=(numpy.random.rand(self.hidnodes,self.innodes)-0.5)
        # self.who=(numpy.random.rand(self.outnodes,self.hidnodes)-0.5)
        # 采取正太分布产生矩阵参数
        self.wih = numpy.random.normal(0.0, pow(self.hidnodes, -0.5), (self.hidnodes, self.innodes))
        self.who = numpy.random.normal(0.0, pow(self.outnodes, -0.5), (self.outnodes, self.hidnodes))
        # 激活函数的定义
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # 神经网络的训练函数
    def train(self,input_list,target_list):
        #转化为二维数组
        inputs=numpy.array(input_list,ndmin=2).T
        targets=numpy.array(target_list,ndmin=2).T
        #隐藏层的输入信号与权重矩阵的点乘
        hidden_input=numpy.dot(self.wih,inputs)
        hidden_output=self.activation_function(hidden_input)
        final_input=numpy.dot(self.who,hidden_output)
        final_output=self.activation_function(final_input)
        #计算误差，目标与输出的差值
        output_errors=targets-final_output
        #反向传播调整前层的权重大小，误差反向调整
        hidden_errors=numpy.dot(self.who.T,output_errors)
        #更新二三层之间的权重大小
        self.who+=self.lr*numpy.dot((output_errors*final_output*(1.0-final_output)),numpy.transpose(hidden_output))
        #更新1~2层的权重大小
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_output*(1.0-hidden_output)),numpy.transpose(inputs))
        pass

    # 神经网络的节点训练查询
    def query(self, input_list):
        # 用数组的形式转化输入节点
        inputs = numpy.array(input_list, ndmin=2).T
        # 计算输入层的信号
        hidden_inputs = numpy.dot(self.wih,inputs)
        #输入信号经过激活函数处理形成输出信号
        hidden_outputs=self.activation_function(hidden_inputs)
        #输出信号经过权重处理进入下一层
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #激活
        final_outputs=self.activation_function(final_inputs)
        return final_outputs



#文件读取测试
#打开文件数据流，句柄赋值
# current_path=os.getcwd()
# print(current_path)
# data_file=open("mist_dataset/mnist_train_100.csv",'r')
# data_list=data_file.readlines()
#将一条文本记录以逗号分界分割
# all_values=data_list[0].split(',')

#asfarray=>将文本内容转换成实数    reshape=>记录折叠
# image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
# plt.imshow(image_array,cmap='Greys',interpolation='None')
# plt.show()

#将像素点值的范围0~255缩小到0.1~0.99
# scaled_input=(numpy.asfarray(all_values[1:])/255*0.99)+0.01
# print(scaled_input)

# matplotlib.use('TkAgg')
# print(image_array)
# print(data_list[0])
# data_file.close()


#框架测试
# inputnode=3
# hiddennode=3
# outputnode=3
# learningrate=0.5
# n=neuralNetwork(inputnode,hiddennode,outputnode,learningrate)
# print(n.query([1.0,0.5,-1.5]))




#正式训练测试
input_nodes=784
hidden_nodes=100
output_nodes=10

learning_rate=0.1

#创建神经网络，初始化数据
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#读取训练文件
training_data_file=open("mist_dataset/mnist_train.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()


#增加训练次数（世代）
epochs=2
for e in range(epochs):

    # 遍历文件中的记录
    for record in training_data_list:
        # 利用逗号进行分割
        all_values = record.split(',')
        # 缩小像素的范围
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 对输出节点给定目标值，方标计算误差范围在0.1到0.99
        targets = numpy.zeros(output_nodes) + 0.01
        # 将期望值标签打入目标节点
        targets[int(all_values[0])] = 0.99
        # 开始训练
        n.train(inputs, targets)
        pass

    pass


#用测试集来进行测试验证


#测试集读取
test_data_file=open("mist_dataset/mnist_test.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()
all_values=test_data_list[0].split(',')
print(all_values[0])

image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array,cmap="Greys",interpolation='None')
plt.show()

arry=n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01)
print(arry)

#查看测试集的准确度

#记录成功和失败的数组
scorecard=[]

#循环便利数据集
for record in test_data_list:
    #逗号分割数据集
    all_values=record.split(',')
    correct_label=int(all_values[0])
    print("correct label",correct_label)

    #重整输入的节点数据
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01

    #用query函数查询出输出数组
    outputs=n.query(inputs)

    #取输出数组中最大值
    label=numpy.argmax(outputs)
    print("network anwser ",label)

    if(correct_label==label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

print(scorecard)

scorecard_array=numpy.asarray(scorecard)
print("performance=",scorecard_array.sum()/scorecard_array.size)
