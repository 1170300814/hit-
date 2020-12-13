import numpy as np
import pandas as pd


R = np.array([[4, 0, 2, 0, 1],
              [0, 2, 3, 0, 0],
              [1, 0, 2, 4, 0],
              [5, 0, 0, 3, 1],
              [0, 0, 1, 5, 1],
              [0, 3, 2, 4, 1],
             ])


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



def LMF_grad_desc(R, K=2, max_iter=1000, alpha=0.0001, lamda=0.002):

    M = len(R)
    N = len(R[0])

    
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K)
    Q = Q.T

    
    for steps in range(max_iter):
        
        for u in range(M):
            for i in range(N):
                
                if (R[u][i] > 0):
                    e_ui = np.dot(P[u, :], Q[:, i]) - R[u][i]
                    
                    for k in range(K):
                        P[u][k] = P[u][k] - alpha * (2 * e_ui * Q[k][i] + 2 * lamda * P[u][k])
        
        predR = np.dot(P, Q)

        
        cost = 0
        for u in range(M):
            for i in range(N):
                
                if R[u][i] > 0:
                    cost += (np.dot(P[u, :], Q[:, i]) - R[u][i]) ** 2
                    
                    for k in range(K):
                        cost += lamda * (P[u][k] ** 2 + Q[k][i] ** 2)
        if cost < 0.0001:
            
            break
    return P, Q.T, cost


P, Q, cost = LMF_grad_desc(R)

print(P)
print(Q)
print(cost)

predR = P.dot(Q.T)

print(R)
print(predR)