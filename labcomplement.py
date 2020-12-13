#coding=utf-8
import numpy as np
import pandas as pd
import matplotlib as plt


user_artist=dict() #用户-艺术家字典，包括训练集与测试集
user_artist_train=dict() #训练集
user_artist_test=dict() #测试集
user_train_list=[]
artist_train_list=[]
user_test_list=[]
artist_test_list=[]

'''
此处将信息读取到user_artist字典,以及四个训练测试列表中
'''

R_train=np.zeros((len(user_train_list,artist_train_list)))
R_test=np.zeros((len(user_test_list,artist_test_list)))


#将训练集合的信息读取到R中，为训练集作准备






"""
@输入参数：
R：M*N 的评分矩阵
K：隐特征向量维度
max_iter: 最大迭代次数
alpha：步长
lamda：正则化系数

@输出：
分解之后的 P，Q
P：初始化用户特征矩阵 M*K
Q：初始化物品特征矩阵 N*K，Q 的转置是 K*N
"""


# 核心算法
def LMF_grad_desc(R, K=8, max_iter=1000, alpha=0.0001, lamda=0.002):
    #每次迭代后的误差
    costlist=[]

    # 定义基本的维度参数
    M = len(R)
    N = len(R[0])

    # PQ的初始值随机生成
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K)
    Q = Q.T

    # 迭代开始
    for steps in range(max_iter):
        # 对所有用户u，物品i作遍历，然后对对应的特征向量Pu，Qi做梯度下降
        for u in range(M):
            for i in range(N):
                # 对于每一个大于0的评分，求出预测评分误差 e_ui
                if (R[u][i] > 0):
                    e_ui = np.dot(P[u, :], Q[:, i]) - R[u][i]

                    # 代入公式，按照梯度下降算法更新当前的Pu,Qi
                    for k in range(K):
                        P[u][k] = P[u][k] - alpha * (2 * e_ui * Q[k][i] + 2 * lamda * P[u][k])
        # u,i 遍历完成，所有的特征向量更新完成，可以得到 P、Q，可以计算预测评分矩阵
        predR = np.dot(P, Q)

        # 计算当前损失函数（所有的预测误差平方后求和）
        cost = 0
        for u in range(M):
            for i in range(N):
                # 对于每一个大于 0 的评分，求出预测评分误差后，将所有的预测误差平方后求和
                if R[u][i] > 0:
                    cost += (np.dot(P[u, :], Q[:, i]) - R[u][i]) ** 2
                    # 加上正则化项
                    for k in range(K):
                        cost += lamda * (P[u][k] ** 2 + Q[k][i] ** 2)
        costlist.append(cost)
        if cost < 0.0001:
            # 当前损失函数小于给定的值，退出迭代
            break
    paramater=[K,max_iter,alpha,lamda]
    return P, Q.T, cost,costlist,paramater