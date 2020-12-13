import numpy as np
user_artist = dict()    # (user_index, artist_index), times
user_set=[]
artist_set=[]
user_count=0
artist_count=0

with open("trainSETchoose1.txt","r") as f:#从文件导入user artist times 等信息并且对数组进行初始化
    for i in f.readlines():
        info=i.split("\t")
        user_num=int(info[0])
        artist_num = int(info[1])
        times = int(info[2])
        if user_num not in user_set:
            user_set.append(user_num)
        if artist_num not in artist_set:
            artist_set.append(artist_num)
        user_artist[(user_set.index(user_num), artist_set.index(artist_num))] = times

user_count=len(artist_set)
artist_count=len(artist_set)



def rmse(M, U, V):#对rmse的计算
    n, d = U.shape
    d_1, m = V.shape
    assert d == d_1

    res = 0.0
    for item in M.items():
        i, j = item[0]
        p = 0.0
        for k in range(d):
            p = p + U[i][k] * V[k][j]


        res = res + (p - item[1]) * (p - item[1])
    return res


def UV_decomposition(M, n, m, d=2, epsilon=1e-3):#将M分解成一个N*d和d*M的矩阵，rmse要求达到精度
    U = np.ones((n, d))
    V = np.ones((d, m))

    time = 0
    rmse_last = rmse(M, U, V)
    costlist=[]
    for i in range(100):
        print(time, rmse_last)
        time += 1
        u_turn = True   # 下一次进行对U中元素的更新
        ui = uj = vi = vj = 0
        u_finished = v_finished = False    # U和V矩阵元素完成标志
        while True:#轮流对U和V进行更新
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
                        if k != uj:#对另一行进行计算
                            p = p + U[ui][k] * V[k][j]
                    numerator = numerator + V[uj][j] * (m_rj - p)#分子
                    denominator = denominator + V[uj][j] * V[uj][j]#分母

                U[ui][uj] = U[ui][uj] if numerator == 0 or denominator == 0 else numerator * 1.0 / denominator#对U的更新 条件进行限制
#  if   numberator == 0  或  denominator==0就不修改
#  去除循环开始的0元素干扰
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
                V[vi][vj] = V[vi][vj] if numerator == 0 or denominator == 0 else numerator * 1.0 / denominator

                vj += 1
                if vj >= m:
                    vi += 1
                    vj = 0
                    if vi >= d:
                        v_finished = True
        rmse_this = rmse(M, U, V)
        costlist.append(rmse_this)#记录过程中的rmse值
        if abs(rmse_last - rmse_this) < epsilon:
            break
        else:
            rmse_last = rmse_this
    return U, V,costlist



#将分解出的UV写入文件
def putUVinTrixFile(U,V,uname,vname):
    u_row=len(U)
    u_column=len(U[0])
    v_row=len(V)
    v_column=len(V[0])

    with open(uname,"w") as fu:
        for i in range(u_row):
            for j in range(u_column):
                if not j==u_column-1:
                    fu.write(str(U[i][j])+"\t")
            if not i==u_row-1:
                fu.write("\n")

    with open(vname,"w") as fv:
        for i in range(v_row):
            for j in range(v_column):
                if not j==v_column-1:
                    fv.write(str(V[i][j])+"\t")
            if not i==v_row-1:
                fv.write("\n")

#main函数
if __name__ == '__main__':
    U, V, costlist_rmse = UV_decomposition(user_artist, user_count, artist_count)
    putUVinTrixFile(U,V,'U_file','V_file')
    print(costlist_rmse)

