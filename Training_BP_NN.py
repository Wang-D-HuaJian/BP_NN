#coding: utf-8
import numpy as np
from math import sqrt
from sklearn.datasets import load_iris
from numpy import *



#BP神经网络模型的训练
def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    """计算隐含层的输入
    input:  feature(mat): 特征
            label(mat):  标签
            n_hidden(int): 隐含层的节点数
            maxCycle(int):  最大的迭代次数
            alpha(float):  学习率
            n_output(int):  输出层的节点数

    output:  w0(mat):  输入层到隐含层之间的的权重
             b0(mat):  输入层到隐含层之间的偏置
             w1(mat):  隐含层到输出层之间的权重
             b1(mat):  隐含层到输出层之间的偏置
    """
    m, n = np.shape(feature)
    #1. 随机初始化参数（权重，偏置， 网络层结构， 激活函数）
    w0 = np.mat(np.random.rand(n, n_hidden))
    w0 = w0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) -\
         np.mat(np.ones((n, n_hidden))) * (4.0 * sqrt(6) / sqrt(n + n_hidden))
    b0 = np.mat(np.random.rand(1, n_hidden))
    b0 = b0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) -\
         np.mat(np.ones((1, n_hidden))) * (4.0 * sqrt(6) / sqrt(n + n_hidden))
    w1 = np.mat(np.random.rand(n_hidden, n_output))
    w1 = w1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) -\
         np.mat(np.ones((n_hidden, n_output))) * (4.0 * sqrt(6) / sqrt(n_hidden + n_output))
    b1 = np.mat(np.random.rand(1, n_output))
    b1 = b1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) -\
         np.mat(np.ones((1, n_output))) * (4.0 * sqrt(6) / sqrt(n_hidden + n_output))

    #2. 训练
    i = 0
    while i <= maxCycle:
        #信号正向传播
        #计算隐含层的输入
        hidden_input = hidden_in(feature, w0, b0)
        #计算隐含层的输出
        hidden_output = hidden_out(hidden_input)
        #计算输出层的输入
        output_in = predict_in(hidden_output, w1, b1)
        #计算输出层的输出
        output_out = predict_out(output_in)

        #误差的反向传播
        #隐含层到输出层之间的残差
        delta_output = -np.multiply((label - output_out), partial_sig(output_in))
        #输入层到隐含层之间的残差
        delta_hidden = np.multiply((delta_output * w1.T), partial_sig(hidden_input))

        #修正权重和偏置
        w1 = w1 - alpha * (hidden_output.T * delta_output)
        b1 = b1 - alpha * np.sum(delta_output, axis=0) * (1.0 / m)
        w0 = w0 - alpha * (feature.T * delta_hidden)
        b0 = b0 - alpha * np.sum(delta_hidden, axis=0) * (1.0 / m)
        if i % 100 == 0:
            print("\t------iter: ", i, ", cost: ", (1.0/2) * get_cost(get_predict(feature, w0, w1, b0, b1) - label))
        i += 1
    return w0, w1, b0, b1

#计算隐含层的输入的hidden_in函数
def hidden_in(feature, w0, b0):
    """隐含层的输入
    input:  feature(mat): 特征
            w0(mat): 输入层到隐含层之间的权重
            b0(mat): 输入层到隐含层之间的偏置
    output:  hidden_in(mat): 隐含层的输入
    """
    m = np.shape(feature)[0]
    hidden_in = feature * w0
    for i in range(m):
        hidden_in[i, ] += b0
    return hidden_in

#计算隐含层的输出的hidden_out函数
def hidden_out(hidden_in):
    """隐含层的输出
    input:  hidden_in(mat): 隐含层的输入
    output: hidden_output(mat): 隐含层的输出
    """
    hidden_output = sig(hidden_in)
    return hidden_output

#计算输出层的输入的predict_in函数
def predict_in(hidden_out, w1, b1):
    """输出层的输入
    input:  hidden_out(mat): 隐含层的输出
            w1(mat): 隐含层到输出层之间的权重
            b1(mat): 隐含层到输出层之间的偏置
    output:  predict_in(mat): 输出层的输入
    """
    m = np.shape(hidden_out)[0]
    predict_in = hidden_out * w1
    for i in range(m):
        predict_in[i, ] += b1
    return predict_in

#计算输出层的输出的predict_out函数
def predict_out(predict_in):
    """输出层的输出
    input:  predict_in(mat): 输出层的输入
    output:  result(mat): 输出层的输出
    """
    result = sig(predict_in)
    return result

#Sigmoid函数
def sig(x):
    """Sigmoid激活函数
    input:  x(mat/float): 自变量（矩阵或者任意实数）
    output: Sigmoid值（mat/float）: Sigmoid函数的值
    """
    return 1.0 / (1 + np.exp(-x))

#partial_sig函数
def partial_sig(x):
    m, n = np.shape(x)
    out = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            out[i, j] = sig(x[i, j]) * (1 - sig(x[i, j]))
    return out

#计算损失函数值的get_cost函数
def get_cost(cost):
    m, n = np.shape(cost)
    cost_sum = 0.0
    for i in range(m):
        for j in range(n):
            cost_sum += cost[i, j] * cost[i, j]
    return cost_sum / m

#计算错误率的err_rate函数
def err_rate(label, pre):
    m = np.shape(label)[0]
    err = 0.0
    for i in range(m):
        if label[i, 0] != pre[i, 0]:
            err += 1
    rate = err / m
    return rate


# #导入数据的load_data函数
# def load_data(filename):
#     """导入训练数据
#     input：  filename(string): 文件名
#     output:  feature_name(mat): 特征
#             label_data(mat): 标签
#             n_class(int): 类别的个数
#     """
#     #1.获取特征
#     f = open(filename) #也可以用上下文管理器进行管理文件的打开和关闭 with open(filename) as f:这样就不用f.close进行关闭
#     feature_data = []
#     label_tmp = []
#     for line in f.readlines():
#         feature_tmp = []
#         lines = line.strip().split("\t")
#         for i in range(len(lines) - 1):
#             feature_tmp.append(float(lines[i]))
#         label_tmp.append(int(lines[-1]))
#         feature_data.append(feature_tmp)
#    # f.close()
#
#     #2.获取标签
#     m = len(label_tmp)
#     n_class = len(set(label_tmp))
#     label_data = np.mat(np.zeros((m , n_class)))
#     for i in range(m):
#         label_data[i, label_tmp[i]] = 1
#     return np.mat(feature_data), label_data, n_class

def load_data():
    dataset_iris = load_iris()
    data_iris = dataset_iris['data']
    label_tmp = dataset_iris['target']
    m = len(label_tmp)
    n_class = len(set(label_tmp))
    print(label_tmp)
    label_iris = np.mat(np.zeros((m, n_class)))
    for i in range(m):
        label_iris[i, label_tmp[i]] = 1
    print(label_iris)
    return data_iris, label_iris, n_class


#保存bp模型的save_model函数
def save_model(w0, w1, b0, b1):
    def write_file(filename, source):
        f = open(filename, "w")
        m, n = np.shape(source)
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(source[i, j]))
            f.write("\t".join(tmp) + "\n")
        f.close()
    write_file("weight_w0", w0)
    write_file("weight_w1", w1)
    write_file("weight_b0", b0)
    write_file("weight_b1", b1)

#对测试样本进行预测的get_predict函数
def get_predict(feature, w0, w1, b0, b1):
    return predict_out(predict_in(hidden_out(hidden_in(feature, w0, b0)), w1, b1))


#训练BP模型的主函数
if __name__ == "__main__":
    #1.导入数据
    print("----------1. load Data-------------")
    feature, label, n_class = load_data()
    #2.训练模型
    print("----------2. training--------------")
    w0, w1, b0, b1 = bp_train(feature, label, 30, 1000, 0.01, n_class)
    #3.保存模型
    print("----------3. save model-------------")
    save_model(w0, w1, b0, b1)
    #4.得到最终的预测结果
    print("----------4. get prediction----------")
    result = get_predict(feature, w0, w1, b0, b1)
    print("训练准确性为： ", (1 - err_rate(np.argmax(label, axis=1), np.argmax(result, axis=1))))

