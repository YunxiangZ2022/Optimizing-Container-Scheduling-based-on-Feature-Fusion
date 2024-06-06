from torch.utils.data import DataLoader, ConcatDataset
import torch
from torcheval.metrics.functional import r2_score
import os
import pickle
from FUSION104 import MODEL
stack=10
tier=4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MODEL().to(device)
criterion = torch.nn.MSELoss(reduction='mean')
criterion1 = torch.nn.L1Loss(reduction='mean')
learning_rate=0.002
optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
Epoch = 200
batch_size = 64
shuffle = True

train_size_per_target = 24576
test_size_per_target = 6144
train_size = train_size_per_target * (stack * tier - 3)
test_size = test_size_per_target * (stack * tier - 3)

dataset_save_directory = 'dataset'
train = []
test = []
for n in range(6):
    with open(os.path.join(dataset_save_directory, str(stack) + '-' + str(tier) + '-' + str(n) + 'custom_dataset_test.pkl'), 'rb') as dataset_test:
        custom_dataset_test = pickle.load(dataset_test)
        test.append(custom_dataset_test)
    with open(os.path.join(dataset_save_directory, str(stack) + '-' + str(tier) + '-' + str(n) + 'custom_dataset_train.pkl'), 'rb') as dataset_train:
        custom_dataset_train = pickle.load(dataset_train)
        train.append(custom_dataset_train)
custom_dataset_train = ConcatDataset(train)
custom_dataset_test = ConcatDataset(test)
train_data = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=shuffle)
test_data = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=shuffle)
print('本次训练参数：{}-{}-{}翻箱问题，learning_rate={}，train_size={}，test_size={}，Epoch={}，'
      'batch_size={}'.format(stack, tier, stack * tier - 1, learning_rate, train_size,
                             test_size, Epoch, batch_size))

test_loss = []
train_loss = []
test_mae = []
test_r2 = []
test_accu = []
test_accu1 = []
train_accu1 = []

for epoch in range(Epoch):
    print('--------------第{}轮迭代开始--------------'.format(epoch + 1))
    #训练
    accu1_s = 0
    loss_s = 0
    num_batches = 0
    model.train()
    for i, item in enumerate(train_data):
        data, global_feature, label = item
        data, global_feature, label = data.to(device), global_feature.to(device), label.to(device)
        # 训练
        y_predict = model(data, global_feature)
        num_batches = num_batches + 1
        accu1 = ((abs(y_predict - label)) < 0.5).sum()
        accu1_s = accu1_s + accu1
        loss = criterion(y_predict, label)
        loss_s = loss_s + loss.item()
        # 梯度清零
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_size = num_batches * batch_size
    print('train_size:{}'.format(train_size))
    print('第{}轮训练集loss:{}'.format(epoch +1 , loss_s / num_batches))
    train_loss.append(loss_s / num_batches)
    print('第{}轮训练集accuTop1:{}'.format(epoch + 1, accu1_s / train_size))
    train_accu1.append(accu1_s / train_size)

    #测试
    model.eval()
    loss_s = 0
    mae_s = 0
    r2_s = 0
    accu_s = 0
    accu1_s = 0
    num_batches = 0
    with torch.no_grad():
        for i, item in enumerate(test_data):
            data, global_feature, label = item
            data, global_feature, label = data.to(device), global_feature.to(device), label.to(device)
            y_predict = model(data, global_feature)
            loss = criterion(y_predict, label)
            loss_s = loss_s + loss.item()
            mae = criterion1(y_predict, label)
            mae_s = mae_s + mae.item()
            r2 = r2_score(y_predict, label)
            r2_s = r2_s + r2.item()
            num_batches = num_batches + 1
            accu = ((abs(y_predict - label) < 1)).sum()
            accu1 = ((abs(y_predict - label)) < 0.5).sum()
            accu_s = accu_s + accu
            accu1_s = accu1_s + accu1
    test_size = num_batches * batch_size
    print('test_size:{}'.format(test_size))
    print('第{}轮测试集loss（mse）:{}'.format(epoch + 1, loss_s / num_batches))
    test_loss.append(loss_s / num_batches)
    print('第{}轮测试集mae:{}'.format(epoch + 1, mae_s / num_batches))
    test_mae.append(mae_s / num_batches)
    print('第{}轮测试集r2:{}'.format(epoch +1, r2_s / num_batches))
    test_r2.append(r2_s / num_batches)
    print('第{}轮测试集accuTop2:{}'.format(epoch + 1, accu_s / test_size))
    test_accu.append(accu_s / test_size)
    print('第{}轮测试集accuTop1:{}'.format(epoch + 1, accu1_s / test_size))
    test_accu1.append(accu1_s / test_size)

save_directory = 'FUSION' + str(stack) + '-' + str(tier)
os.makedirs(save_directory, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_directory, str(stack) + '-' + str(tier) + '-' + str(stack * tier - 1) + 'model.pkl'))
with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + 'test_loss.txt'), 'w') as test_los:
    test_los.write(str(test_loss))
with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + 'train_loss.txt'), 'w') as train_los:
    train_los.write(str(train_loss))
with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + 'test_loss_mae.txt'), 'w') as test_los_mae:
    test_los_mae.write(str(test_mae))
with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + 'test_r2.txt'), 'w') as test_r22:
    test_r22.write(str(test_r2))
with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + 'test_accTOP1.txt'), 'w') as test_acc:
    test_acc.write(str(test_accu1))
with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + 'test_accTOP2.txt'), 'w') as test_acc2:
    test_acc2.write(str(test_accu))
with open(os.path.join(save_directory, str(stack) + '-' + str(tier) + 'train_accTOP1.txt'), 'w') as train_acc:
    train_acc.write(str(train_accu1))
print('模型与数据已保存')
print('本次训练参数：{}-{}-{}翻箱问题，learning_rate={}，train_size={}，test_size={}，Epoch={}，'
      'batch_size={}'.format(stack, tier, stack * tier - 1, learning_rate, train_size,
                             test_size, Epoch, batch_size))