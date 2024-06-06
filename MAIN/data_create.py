from gel import generate21
from gel import standardlize
from gel import LBlow_exact
from Dataset import CustomDataset1
import numpy as np
import torch
import os
import pickle

stack=10
tier=7
totalnumber=stack * tier - 1
Boundnum_high = stack * tier - tier + 1
train_size_per_target = 4096
test_size_per_target = 1024
train_size = train_size_per_target * (totalnumber - 2)
test_size = test_size_per_target * (totalnumber - 2)
print('本次生成参数：{}-{}-{}翻箱问题，train_size={}，test_size={}'.format(stack, tier, totalnumber,  train_size, test_size))

# 生成训练集
inputs = []
global_features = []
labels = []
count = [0 for i in range(Boundnum_high)]
while True:
    for target in range(2, Boundnum_high):
        XX = generate21(stack,tier,target+1)
        label = LBlow_exact(XX, stack, tier)
        if label == -1:
            continue
        count[target] = count[target] + 1
        if count[target] == train_size_per_target + 1:
            count[target] = count[target] - 1
            continue
        nowbayss, global_feature = standardlize(XX, stack, tier)
        input = nowbayss.reshape(stack, tier)
        inputs.append([input])
        global_features.append([global_feature])
        labels.append([label])
    if all(x == train_size_per_target for x in count[2:]):
        break
print('训练集part1已生成')
count = [0 for i in range(Boundnum_high, totalnumber)]
while True:
    for target in range(Boundnum_high, totalnumber):
        XX = generate21(stack,tier,target+1)
        label = LBlow_exact(XX, stack, tier)
        if label == -1:
             continue
        target = target - Boundnum_high
        count[target] = count[target] + 1
        if count[target] == train_size_per_target + 1:
            count[target] = count[target] - 1
            continue
        nowbayss, global_feature = standardlize(XX, stack, tier)
        input = nowbayss.reshape(stack, tier)
        inputs.append([input])
        global_features.append([global_feature])
        labels.append([label])
    if all(x == train_size_per_target for x in count):
        break
inputs = np.asarray(inputs)
global_features = np.asarray(global_features)
labels = np.asarray(labels)
custom_dataset_train = CustomDataset1(inputs, global_features, labels)
print('训练集已全部生成')
# 生成测试集
inputs = []
global_features = []
labels = []
count = [0 for i in range(Boundnum_high)]
while True:
    for target in range(2, Boundnum_high):
        XX = generate21(stack,tier,target+1)
        label = LBlow_exact(XX, stack, tier)
        if label == -1:
            continue
        count[target] = count[target] + 1
        if count[target] == test_size_per_target + 1:
            count[target] = count[target] - 1
            continue
        nowbayss, global_feature = standardlize(XX, stack, tier)
        input = nowbayss.reshape(stack, tier)
        inputs.append([input])
        global_features.append([global_feature])
        labels.append([label])
    if all(x == test_size_per_target for x in count[2:]):
        break
count = [0 for i in range(Boundnum_high, totalnumber)]
while True:
    for target in range(Boundnum_high, totalnumber):
        XX = generate21(stack,tier,target+1)
        label = LBlow_exact(XX, stack, tier)
        if label == -1:
            continue
        target = target - Boundnum_high
        count[target] = count[target] + 1
        if count[target] == test_size_per_target + 1:
            count[target] = count[target] - 1
            continue
        nowbayss, global_feature = standardlize(XX, stack, tier)
        input = nowbayss.reshape(stack, tier)
        inputs.append([input])
        global_features.append([global_feature])
        labels.append([label])
    if all(x == test_size_per_target for x in count):
        break
inputs = np.asarray(inputs)
global_features = np.asarray(global_features)
labels = np.asarray(labels)
custom_dataset_test = CustomDataset1(inputs, global_features, labels)
print('测试集已全部生成')

save_directory = 'dataset'
n = 2
os.makedirs(save_directory, exist_ok=True)
with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + '-' + str(n) + 'custom_dataset_train.pkl'), 'wb') as datase_train:
    pickle.dump(custom_dataset_train, datase_train)
with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + '-' + str(n) +   'custom_dataset_test.pkl'), 'wb') as datase_test:
    pickle.dump(custom_dataset_test, datase_test)