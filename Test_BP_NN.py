#coding: utf-8
import numpy as np
#import sys
#sys.path.append("F:\Machine_Learning_Algorithm\BP_Neural_Network")
from Training_BP_NN import get_predict

#生成测试样本的generate_data函数
def generate_data():
    """在[-4.5, 4.5]之间随机生成20000组点"""
    #1. 随机生成数据点
    data = np.mat(np.zeros((200, 4)))
    m = np.shape(data)[0]
    x = np.mat(np.random.rand(200, 4))
    for i in range(m):
        data[i, 0] = x[i, 0] * 4 + 4
        data[i, 1] = x[i, 1] * 3 + 2
        data[i, 2] = x[i, 2] * 6 + 1
        data[i, 3] = x[i, 3] * 3

    #2. 将数据点保存到文件“test_data”中
    f = open("test_data", "w")
    m, n = np.shape(data)
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(str(data[i, j]))
        f.write("\t".join(tmp) + "\n")
    f.close()

#导入测试数据的load_data函数
def load_data(filename):
    f = open(filename)
    feature_data = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines)):
            feature_tmp.append(float(lines[i]))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data)

#导入BP神经网络模型的load_model函数
def load_model(file_w0, file_w1, file_b0, file_b1):
    def get_model(file_name):
        f = open(file_name)
        model = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = []
            for x in lines:
                model_tmp.append(float(x.strip()))
            model.append(model_tmp)
        f.close()
        return np.mat(model)
    w0 = get_model(file_w0)
    w1 = get_model(file_w1)
    b0 = get_model(file_b0)
    b1 = get_model(file_b1)
    return w0, w1, b0, b1

#保存最终的预测结果的save_predict函数
def save_predict(filename, pre):
    f = open(filename, "w")
    m = np.shape(pre)[0]
    result = []
    for i in range(m):
        result.append(str(pre[i, 0]))
    f.write("\n".join(result))
    f.close()



#对新数据进行预测的主函数
if __name__ == "__main__":
    generate_data()
    #1. 导入数据
    print("-------------1. load data--------------")
    dataTest = load_data("test_data")
    #2. 导入BP神经网络模型
    print("-------------2. load model--------------")
    w0, w1, b0, b1 = load_model("weight_w0", "weight_w1", "weight_b0", "weight_b1")
    #3. 得到最终的预测值
    print("-------------3. get prediction------------")
    result = get_predict(dataTest, w0, w1, b0, b1)
    #. 保存最终的预测结果
    print("-------------4. save result---------------")
    pre = np.argmax(result, axis=1)
    save_predict("result", pre)