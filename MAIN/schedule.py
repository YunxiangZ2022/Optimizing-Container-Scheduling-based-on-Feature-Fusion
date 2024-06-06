import torch
import treelib
from FUSION104 import MODEL
import numpy as np
from gel import LBlow
from gel import softmin
from gel import LBlow_exact
from gel import standardlize
import os
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
stack=10
tier=4

model = MODEL().to(device)
save_directory = 'FUSION' + str(stack) + '-' + str(tier)
model_path = os.path.join(save_directory, str(stack) + '-' + str(tier) + '-' + str(stack * tier - 1) + 'model.pkl')
model.load_state_dict(torch.load(model_path))
model.eval()

# 容器堆出递归函数recursion，输入当前树根
def recursion(treenow,id):
    global a
    # 考虑到列无关性，对状态的一维数组形式以每列最高优先级、每列容器数为关键字进行排序标准化处理，为简化此函数计算
    treenow.nodes[id].data[0], _ = standardlize(treenow.nodes[id].data[0], stack, tier)
    while not np.all(treenow.nodes[id].data[0]==0):
        # 将状态数组转化为NumPy数组
        nowbay=np.array(treenow.nodes[id].data[0]).copy()
        nowbay=nowbay.reshape(stack,tier)
        ##简化问题
        # 计算阻塞数，若无阻塞则已是最优解，置零返回
        if LBlow(nowbay.flatten(), stack, tier)==0:
            treenow.update_node(id,data=[np.zeros(stack*tier),treenow.nodes[id].data[1],0])
            return
        # 若有优先级为1的容器
        if np.any(nowbay==1):
            x,y=np.where(nowbay == 1)
            x=int(x.item())
            y=int(y.item())
            nowbay=np.floor(nowbay)
            # 若其在绝对最高层则堆出，对新的容器布局进行标准化处理，并再次递归简化
            if y==tier-1:
                nowbay[x,y]=0
                data1, _ = standardlize(nowbay.flatten(), stack, tier)
                treenow.update_node(id,data=[data1, treenow.nodes[id].data[1],0])
                recursion(treenow,a)
                return
            # 相对最高层情况同理
            elif np.all(nowbay[x,y+1:]==0):
                nowbay[x,y]=0
                data1, _ = standardlize(nowbay.flatten(), stack, tier)
                treenow.update_node(id,data=[data1, treenow.nodes[id].data[1],0])
                recursion(treenow,a)
                return
            # 被阻挡的情况
            else:
                nowbay[x,y]=1
                #if there is blocking container, move it to other place
                for tierblocking,keyblocking in enumerate(np.flipud(nowbay[x,y+1:])):
                    tierblocking=tier-tierblocking-1
                    # 自上而下考察在其上层的容器
                    if keyblocking!=0:
                        nowbay[x,tierblocking]=0
                        cckk=[]
                        ccckk=[]
                        # 接受站数标识符（sig+1）
                        sig=-1
                        # 逐列寻找接受栈
                        for putstack in range(stack):
                            if (putstack!=x) and (np.any(nowbay[putstack]==0)):
                                tiertarget=np.where(nowbay[putstack]==0)[0][0]
                                # 重定位，maybe it will be wrong
                                nowbay[putstack,tiertarget]=keyblocking
                                # 重定位后的容器布局添加到ccckk
                                ccckk.append(nowbay.flatten())
                                nowbayss, global_feature = standardlize(nowbay, stack, tier)
                                global_feature = torch.Tensor(np.asarray([[global_feature]])).to("cuda")
                                raw_data1test = (torch.Tensor(nowbayss).flatten().reshape(1,1,stack,tier)).to("cuda")
                                # 选择对应容器数模型预测该容器布局表现
                                y_predtest = np.max((model(raw_data1test, global_feature)).item(),0)
                                # 将模型预测结果添加到cckk
                                cckk.append(y_predtest)
                                if LBlow(nowbay.flatten(), stack, tier)==0:
                                    # print(nowbay.reshape(stack, tier))
                                    sig=len(cckk)-1
                                # 复原 
                                nowbay[putstack,tiertarget]=0
                        # 一般情况
                        if sig==-1:
                            # 选择重定位次数预测最小的
                            nowbay=ccckk[cckk.index(min(cckk))]
                        # 解出情况
                        else:
                            nowbay=ccckk[sig]
                            a=a+1
                            data1, _ = standardlize(nowbay, stack, tier)
                            treenow.create_node(a,a,id,data=[data1,0,0])
                            return
                        a=a+1
                        data1, _ = standardlize(nowbay, stack, tier)
                        treenow.create_node(a,a,id,data=[data1,0,0])
                        recursion(treenow,a)
                        return
                return
        # 若无优先级为1的容器，则需归一化
        else:
            nonzerovalue=nowbay[np.nonzero(nowbay)]
            nonmin=np.min(nonzerovalue)
            nowbay[np.where(nowbay >=2)] = nowbay[np.where(nowbay >=2)] - (int(np.floor(nonmin))-1)
            data1, _ = standardlize(nowbay.flatten(), stack, tier)
            treenow.update_node(id,data=[data1,treenow.nodes[id].data[1],0])
            recursion(treenow,a)
            return
    # end while

totalnumber = 40
relocation = 0
relocation_exact = 0
best = 0
ta = 0
te = 0
with torch.no_grad():
    for i in range(601, 701):
        dataset_test_path = os.path.join(save_directory, str(tier) + '-' + str(stack) + '-' + str(totalnumber), '10' + str(i) + '.txt')
        with open(dataset_test_path, 'r') as file:
            lines = file.readlines()
        XX = np.zeros((stack, tier))
        for j, line in enumerate(lines[1:]):
            elements = line.split()
            for k, element in enumerate(elements[1:]):
                XX[j, k] = int(element)
        # print(XX)
        t1 = time.time()
        a = 0
        X=np.floor(XX).flatten()
        tt=treelib.Tree()
        tt.create_node(0,0, data=[X,0,0])
        recursion(tt,0)
        t2 = time.time()
        # print(a)
        relocation = relocation + a
        e = LBlow_exact(XX, stack, tier)
        t3 = time.time()
        ta = ta + t2 - t1
        te = te + t3 - t2
        if e == a:
            best = best + 1
        relocation_exact = relocation_exact + e
print('本方法的平均重定位次数是：{}'.format(relocation/100))
print('精确解的平均重定位次数是：{}'.format(relocation_exact/100))
print('最优率：{}'.format(best/100))
# print('本方法平均时间：{}'.format(ta/100))
# print('精确解平均时间：{}'.format(te/100))