#coding=utf-8
import numpy as np
import pandas as pd
import matplotlib as plt

# 建立从user_id到矩阵下标的映射
user_to_index = dict()
# 建立从artist_id到矩阵下标的映射
artist_to_index = dict()
# user和artist的数量
user_number = 0
artist_number = 0

artist_alias = dict()   # wrong_id, correct_id
artist_data = dict()    # artist_id, artist_name
user_artist = dict()    # (user_index, artist_index), times



def getArtistidandName(line):
    sizel = len(line)
    for i in range(0, sizel):
        if not(ord(line[i])>=48 and ord(line[i])<=57):
            if line[i] == " " or line[i] == "\t":
                return line[0:i], line[i + 1:]
            else:
                return line[0:i], line[i:]





with open('./artist_alias.txt', 'r',encoding='utf-8') as f:
    for line in f.readlines():
        line = line.replace('\n', '')
        wrong, correct = line.split('\t', 2)
        artist_alias[wrong] = correct

with open('./artist_correct_format_data.txt', 'r',encoding='utf-8') as f:
    for line in f.readlines():
        line = line.replace('\n', '')
        artist_id, artist_name=getArtistidandName(line)
        artist_data[artist_id] = artist_name
        # 如果有重复的artist_id出现，去最后一次出现的值
        if artist_id not in artist_to_index:
            artist_to_index[artist_id] = artist_number
            artist_number = artist_number + 1

with open('./user_artist_data.txt', 'r',encoding='utf-8') as f:
    for line in f.readlines():
        line = line.replace('\n', '')
        user_id, artist_id, times = line.split('\t', 3)
        user_index = -1
        artist_index = -1
        if artist_id in artist_alias:
            artist_id = artist_alias.get(artist_id)

        if artist_id not in artist_to_index:
            continue    # 存在部分artist_id在artist_data.txt中不存在的现象
        else:
            artist_index = artist_to_index.get(artist_id)

        if user_id in user_to_index:
            user_index = user_to_index.get(user_id)
        else:
            user_to_index[user_id] = user_number
            user_index = user_number
            user_number = user_number + 1
        # 如果有重复的user_index, artist_index出现，取最后一次有效的值
        user_artist[(user_index, artist_index)] = eval(times)


def my_train_test_split(data, test_size=0.2, random_state=3):
    train_set = dict()
    test_set = dict()
    np.random.seed(random_state)

    for item in data.items():
        if np.random.rand() < test_size:
            test_set[item[0]] = item[1]
        else:
            train_set[item[0]] = item[1]
    return train_set, test_set


train_user_artist, test_user_artist = my_train_test_split(user_artist)
print(len(train_user_artist), len(test_user_artist))

print('用户数量', user_number)
print('艺术家数量', artist_number)


def rmse(M, U, V):
    n, d = U.shape
    d_1, m = V.shape
    assert d == d_1

    res = 0.0
    #     for i in range(n):
    #         for j in range(m):
    #             t = M.get((i, j), None)
    #             if t is None:
    #                 continue
    #             p = 0.0
    #             for k in range(d):
    #                 p = p + U[i][k] * V[k][j]
    #             res = res + (p - t) * (p - t)
    for item in M.items():
        i, j = item[0]
        # i = user_to_index.get(i, None)
        # j = artist_to_index.get(j, None)
        # if i is None or j is None:
        #     continue
        p = 0.0
        for k in range(d):
            p = p + U[i][k] * V[k][j]
        res = res + (p - item[1]) * (p - item[1])
    return res


def UV_decomposition(M, n, m, d=2, epsilon=1e-3):
    U = np.ones((n, d))
    V = np.ones((d, m))

    time = 0
    rmse_last = rmse(M, U, V)
    while True:
        print(time, rmse_last)
        time += 1
        u_turn = True   # 下一次进行对U中元素的更新
        ui = uj = vi = vj = 0
        u_finished = v_finished = False    # U和V矩阵元素完成标志
        while True:
            if u_finished and v_finished:
                break
            if v_finished or u_turn:
                u_turn = False
                numerator = 0
                denominator = 0
                for j in range(m):
                    p = 0
                    m_rj = M.get((ui, j), None)
                    if m_rj is None:
                        continue
                    for k in range(d):
                        if k != uj:
                            p = p + U[ui][k] * V[k][j]
                    numerator = numerator + V[uj][j] * (m_rj - p)
                    denominator = denominator + V[uj][j] * V[uj][j]
                # last_u = U[ui][uj]
                # if numerator == 0 or denominator == 0:
                #     continue
                U[ui][uj] = U[ui][uj] if numerator == 0 or denominator == 0 else numerator * 1.0 / denominator
                # rmse_this = rmse(M, U, V)
                # if rmse_this < rmse_last:
                #     rmse_last = rmse_this
                # else:
                #     U[ui][uj] = last_u
                uj += 1
                if uj >= d:
                    ui += 1
                    uj = 0
                    if ui >= n:
                        u_finished = True
                        
            elif u_finished or not u_turn:
                u_turn = True
                numerator = 0
                denominator = 0
                for i in range(n):
                    p = 0
                    m_is = M.get((i, vj), None)
                    if m_is is None:
                        continue
                    for k in range(d):
                        if k != vi:
                            p = p + U[i][k] * V[k][vj]
                    numerator = numerator + U[i][vi] * (m_is - p)
                    denominator = denominator + U[i][vi] * U[i][vi]
                # last_v = V[vi][vj]
                # if numerator == 0 or denominator == 0:
                #     continue
                V[vi][vj] = V[vi][vj] if numerator == 0 or denominator == 0 else numerator * 1.0 / denominator
                # rmse_this = rmse(M, U, V)
                # if rmse_this < rmse_last:
                #     rmse_last = rmse_this
                # else:
                #     V[vi][vj] = last_v
                vj += 1
                if vj >= m:
                    vi += 1
                    vj = 0
                    if vi >= d:
                        v_finished = True
        rmse_this = rmse(M, U, V)
        if abs(rmse_last - rmse_this) < epsilon:
            break
        else:
            rmse_last = rmse_this
    return U, V

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

'''
参数:
P:用户-特征矩阵
Q:特征-艺术家矩阵
i:预测矩阵的第i个用户
j:预测矩阵的第j个艺术家
返回值：
result：预测矩阵第i行第j列的结果

'''
def predict(P,Q,i,j):
    result=0
    for k in range(P[0]):
        result+=P[i][k]*Q[k][j]
    return result

U, V = UV_decomposition(user_artist, user_number, artist_number)





P, Q, cost,costlist,plist = LMF_grad_desc(R)

costsize=len(costlist)
xlabel=[]
for i in range(costsize):
    xlabel.append(i+1)
labelstring="HFVD="+str(plist[0])+","+"MNI="+str(plist[1])+","+"StepLength="+str(plist[2])+","+"regluar="+str(plist[3])
plt.plot(xlabel,costlist,label=labelstring)
plt.xlabel("Number of iterations")
plt.ylabel("Loss function value")
plt.legend()
plt.show()

