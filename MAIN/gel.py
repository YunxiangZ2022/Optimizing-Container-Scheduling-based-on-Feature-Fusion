import numpy as np
import random
import math
import torch
import itertools
import subprocess

def standardlize(bayarray, stack, tier):
    bayarray_0=bayarray.reshape(stack,tier)
    bayarray=bayarray_0
    # 一维数组count存储每列容器数
    count = np.zeros((stack,1))
    for i in range(stack):
        m=0
        for j in range(tier):
            if bayarray[i][j]!=0:
                m=m+1
        count[i][0]=m
    #一维数组reverse_order存储每列逆序数
    reverse_order = np.zeros((stack, 1))
    blocking_count = np.zeros((stack, 1))
    for i in range(stack):
        reverse_count = 0
        for j in range(tier-1,0,-1):
            if bayarray[i][j] != 0:
                flag = 0
                for k in range(j):
                    if bayarray[i][k] < bayarray[i][j]:
                        reverse_count += 1
                        flag = 1
                if flag == 1:
                    blocking_count[i][0] = blocking_count[i][0] + 1
        reverse_order[i][0] = reverse_count
    # 一维数组tiermin存储每列最高优先级
    tiermin = np.min(np.where(bayarray == 0, bayarray.max() + 1, bayarray), axis=1)
    tiermin[tiermin == bayarray.max() + 1] = 0
    tiermin = tiermin.reshape(1, -1).T
    # 一维数组tiermax存储每列最低优先级
    tiermax = np.max(bayarray, axis=1)
    tiermax = tiermax.reshape(1, -1).T
    # 水平连接三个数组
    bayarray = np.hstack((blocking_count, reverse_order, count, tiermin, tiermax, bayarray))
    bayarray1 = bayarray.T
    a3 = np.lexsort((-bayarray1[0, :],-bayarray1[1, :], bayarray1[3, :], -bayarray1[4, :], bayarray1[2, :]))
    bayarray = bayarray[a3, :]
    bayarray_1 = bayarray[:, 5:]
    bayarray_2 = bayarray[:, :5]
    return bayarray_1.flatten(), bayarray_2.flatten()
    # return bayarray_1.flatten(), bayarray_2.flatten(), bayarray_0.flatten()

def LBlow(bayarray, stack, tier):
    bayarray=bayarray.reshape(stack,tier)
    countblock = []
    for i in range(stack):
        for j in range(tier-1,0,-1):
            canzhao=bayarray[i][j]
            if  canzhao==0:
                continue
            for k in range(j-1,-1,-1):
                if bayarray[i][k]<canzhao:
                    countblock.append(canzhao)
                    break
    return int(len(countblock))

def LBlow2(bayarray, stack, tier):
    bayarray=bayarray.reshape(stack,tier)
    countblock = 0
    bayarray[bayarray == 0] = 99
    indices = np.where(bayarray == 1)
    flag = 0
    for j in range(tier-1, indices[1][0], -1):
        canzhao=bayarray[indices[0][0]][j]
        if canzhao == 99:
            continue
        for i in range(stack):
            if (i != indices[0][0]) and bayarray[i][tier - 1] == 99:
                if all(canzhao < all_blocks for all_blocks in bayarray[i]):
                    flag = 1
        if flag == 0:
            countblock = countblock + 1
        flag = 0
    return int(LBlow(bayarray) + countblock)

def LBlow3(bayarray, stack, tier):
    LB1 = LBlow(bayarray, stack, tier)
    bayarray = bayarray.reshape(stack,tier)
    countblock = 0
    bayarray[bayarray == 0] = 99
    flag = 0
    while not np.all(bayarray == 99):
        min_index = np.unravel_index(np.argmin(bayarray), bayarray.shape)
        if (min_index[1] == tier-1) or (bayarray[min_index[0]][min_index[1] + 1] == 99):
            bayarray[min_index[0]][min_index[1]] = 99
            continue
        for j in range(tier-1, min_index[1], -1):
            canzhao = bayarray[min_index[0]][j]
            if canzhao == 99:
                continue
            for i in range(stack):
                if (i != min_index[0]) and bayarray[i][tier - 1] == 99 and all(canzhao < all_blocks for all_blocks in bayarray[i]):
                    flag = 1
                    break
            if flag == 0:
                countblock = countblock + 1
            flag = 0
        for j in range(tier-1, min_index[1] - 1, -1):
            bayarray[min_index[0]][j] = 99
    return int(LB1 + countblock)

def format_matrix(matrix):
    return '\n'.join([' '.join(map(lambda x: str(x) if x != 0 else '-', row)) for row in matrix])

def LBlow_exact(bayarray, stack, tier):
    bayarray=bayarray.reshape(stack,tier)
    n1 = stack
    n2 = np.count_nonzero(bayarray)
    # 一维数组count存储每列容器数
    count = np.zeros((stack,1))
    for i in range(stack):
        m=0
        for j in range(tier):
            if bayarray[i][j]!=0:
                m=m+1
        count[i][0]=m
    bayarray1 = np.array([[n1, n2]]).astype(int)
    bayarray2 = np.hstack((count, bayarray))
    non_zero_rows = np.where(bayarray2[:, 0] != 0)[0]
    bayarray2 = bayarray2[non_zero_rows].astype(int)
    executable_path = './rbrp_bb'
    input_data = f"{format_matrix(bayarray1)}\n{format_matrix(bayarray2)}"
    process = subprocess.Popen([executable_path]+['-T', str(tier), '-m', str(1)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input_data.encode())
    if process.returncode == 0:
        if stdout.decode().strip().isdigit():
            return int(stdout.decode().strip())
        else:
            return -1
    else:
        return -1

# 随机生成一个状态
def generate21(stacks, tiers,totalnumber):
    origin = np.zeros((stacks, tiers))
    #生成优先级打乱后的列表
    number = [i+1 for i in range(totalnumber)]
    random.shuffle(number)
    # 逐个随机堆入target优先级的容器
    for j in range(totalnumber):
        target = number[j]
        for t in range(10000):
            targetstack = random.randint(0, stacks - 1)
            if 0 not in origin[targetstack][:]:
                continue
            # 列全满继续搜寻
            else:
                break
            # 列不满中断并堆入容器：
        availabletier = np.where(origin[targetstack][:] == 0)[0][0]
        origin[targetstack][availabletier] = target
    baybay = origin.copy()
    # print(baybay,'baybay')
    bay = baybay.copy()
    # 取出贝中非零元素进行比较，若最高优先级不是1则标准化
    nonzerovalue = bay[np.nonzero(bay)]
    nonmin = np.min(nonzerovalue)
    if nonmin < 2:
        pass
    else:
        bay[np.where(bay >= 2)] = bay[np.where(bay >= 2)] - (nonmin - 1)
    return bay

def softmin(ll,p):
    ll=[i*2 for i in ll]
    hh=[(sum(ll)/len(ll)-i)/(0.99**(p)) for i in ll]
    a=sum([math.e**(i) for i in hh])
    aa=[math.e**(i)/a for i in hh]
    return aa

# n in range(1,totalnumber+1) 生成容器数为n的布局
def rand0(n, stack, tier):
    # 生成1到n的随机排列
    numbers = list(range(1,n+1))
    random.shuffle(numbers)
    # 初始化分组列表
    groups = [[] for _ in range(stack)]
    # 将数字分配到不同的组中
    for num in numbers:
        assigned = False
        while not assigned:
            # 随机选择一个组
            group_index = random.randint(0, stack - 1)
            if len(groups[group_index]) < tier:
                groups[group_index].append(num)
                assigned = True
    for group in groups:
        group.sort(reverse=True)
    for group in groups:
        while len(group) < tier:
            group.append(0)
    return groups