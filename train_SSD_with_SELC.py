import torch
import sys
sys.path.append('../..')
import random
from models import causal_cnn
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from dataloader_ssd import ssd_dataloader
import argparse
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import datetime as dt
import time


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym', help='sym or asym')
parser.add_argument('--model', default='TCN', type=str)
parser.add_argument('--op', default='Adam', type=str, help='optimizer')
parser.add_argument('--alpha', default=0.9, help='alpha in SELC')
# parser.add_argument('--lr_s', default='MultiStepLR', type=str, help='learning rate scheduler')
parser.add_argument('--lr_s', default=0, type=str, help='learning rate scheduler')
parser.add_argument('--loss', default='SELCLoss', type=str, help='loss function')
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=42)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--data_path', default='', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)

parser.add_argument('--channels', default=40, type=int)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--in_channels', default=27, type=int)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--out_channels', default=80, type=int)
parser.add_argument('--reduced_size', default=40, type=int)
parser.add_argument('--clf_hidden_node', default=80, type=int)
parser.add_argument('--clf_dropout_rate', default=0.2, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.seed:
    torch.backends.cudnn.deterministic = False  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation
    
def set_env(args):
    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

set_env(args)


# class SELCLoss(torch.nn.Module):
#     def __init__(self, labels, num_classes, es=10, momentum=0.9):
#         super(SELCLoss, self).__init__()
#         self.num_classes = num_classes
#         self.soft_labels = torch.zeros(len(labels), num_classes, dtype=torch.float).cuda()
#         self.soft_labels[torch.arange(len(labels)), labels] = 1
#         self.es = es
#         self.momentum = momentum
#         self.CEloss = torch.nn.CrossEntropyLoss()

#     def forward(self, logits, labels, index, epoch):
#         pred = F.softmax(logits, dim=1)
#         if epoch <= self.es:
#             ce = self.CEloss(logits, labels)
#             return ce.mean()
#         else:
#             pred_detach = F.softmax(logits.detach(), dim=1)
#             self.soft_labels[index] = self.momentum * self.soft_labels[index] \
#                                       + (1 - self.momentum) * pred_detach

#             selc_loss = -torch.sum(torch.log(pred) * self.soft_labels[index], dim=1)
#             return selc_loss.mean()

class SELCLoss(torch.nn.Module):
    def __init__(self, labels, num_classes, es=10, momentum=0.9):
        super(SELCLoss, self).__init__()
        self.num_classes = num_classes
        self.soft_labels = torch.zeros(len(labels), num_classes, dtype=torch.float).cuda()
        self.soft_labels[torch.arange(len(labels)), labels] = 1
        self.es = es
        self.momentum = momentum
        self.CEloss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels, index, epoch):
        if epoch <= self.es:
            return self.CEloss(logits, labels)
        else:
            pred_detach = F.softmax(logits.detach(), dim=1)
            self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * pred_detach
            pred = F.log_softmax(logits, dim=1) #원래 코드에서 -inf 문제 발생, torch.log 대신 F.log_softmax를 사용해서, 언더플로우나 오버플로우 방지
            selc_loss = -torch.sum(pred * self.soft_labels[index], dim=1)
            return selc_loss.mean()

estimated_es = {'cifar10_sym0.2': 40, 'cifar10_sym0.4': 30, 'cifar10_sym0.6': 30, 'cifar10_sym0.8': 40,
                'cifar10_asym0.4': 40,
                'cifar100_sym0.2': 30, 'cifar100_sym0.4': 20, 'cifar100_sym0.6': 30, 'cifar100_sym0.8': 40,
                'cifar100_asym0.4': 20}

loader = ssd_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                          num_workers=5,
                          root_dir=args.data_path,
                          args=args,
                          noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

all_trainloader, noisy_labels, clean_labels = loader.run('train')
test_loader = loader.run('test')
test_scaled = loader.run('final_test')
#eval_train_loader, _, _ = loader.run('eval_train')

if args.model == 'TCN':
    model = causal_cnn.TCNClassifier(in_channels=args.in_channels,
                        channels=args.channels,
                        depth=args.depth,
                        reduced_size=args.reduced_size,
                        out_channels=args.out_channels,
                        kernel_size=args.kernel_size,
                        clf_hidden_node=args.clf_hidden_node,
                        clf_dropout_rate=args.clf_dropout_rate,
                        num_class=args.num_class).to(args.gpuid)
    

if args.op == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    
if args.op == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

# if args.lr_s == 'MultiStepLR':
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

# criterion = SELCLoss(noisy_labels, args.num_class, estimated_es[args.dataset + '_' + args.noise_mode + str(args.r)],
#                      args.alpha)
# I set es to 5. ~ warm_up stage
criterion = SELCLoss(noisy_labels, args.num_class, 5, args.alpha)

# def train(args, model, train_loader, optimizer, epoch):
#     model.train()
#     loss_per_batch = []
#     correct = 0
#     acc_train_per_batch = []

#     for batch_idx, (data, target, index) in enumerate(train_loader):
#         data, target = data[:,1:,:].to(args.gpuid), target.type(torch.LongTensor).to(args.gpuid)
#         optimizer.zero_grad()
#         output = model(data)

#         loss = criterion(output, target, index, epoch)

#         loss.backward(retain_graph=True)
#         optimizer.step()
#         loss_per_batch.append(loss.item())

#         # save accuracy:
#         pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.view_as(pred)).sum().item()
#         acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

#         if batch_idx % args.log_interval == 0:
#             print(
#                 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
#                     epoch, batch_idx * len(data), len(train_loader.dataset),
#                            100. * batch_idx / len(train_loader), loss.item(),
#                            100. * correct / ((batch_idx + 1) * args.batch_size),
#                     optimizer.param_groups[0]['lr']))
#     loss_per_epoch = [np.average(loss_per_batch)]
#     acc_train_per_epoch = [np.average(acc_train_per_batch)]

#     return loss_per_epoch, acc_train_per_epoch

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    loss_per_batch = []
    correct = 0
    total = 0
    start_time = time.time()

    num_iter = (len(train_loader.dataset) // args.batch_size) + 1

    for batch_idx, (data, target, index) in enumerate(train_loader):
        data, target = data[:,1:,:].to(args.gpuid), target.type(torch.LongTensor).to(args.gpuid)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target, index, epoch)
        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = "{:0>2}:{:0>2}:{:02.0f}".format(int(hours), int(minutes), seconds)

        # Log output
        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] | Elapsed Time %s | Iter[%3d/%3d]  loss: %.4f Acc: %.2f%%'
                         % (epoch, args.num_epochs, time_str, batch_idx + 1, num_iter, loss.item(), 100. * correct / total))
        sys.stdout.flush()

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [100. * correct / total]

    return loss_per_epoch, acc_train_per_epoch


def test_cleaning(test_batch_size, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data[:,1:,:].to(device), target.type(torch.LongTensor).to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return loss_per_epoch, acc_val_per_epoch

def per_disk_test(model, test_scaled, window_size=90):
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    with torch.no_grad():
        model.eval()
        FEATURE_COLUMNS = test_scaled.columns[2:-2]
        predictions, labels = [], []
        disk_id_list = []
        prob_history = []
        for disk_id, group in tqdm(test_scaled.groupby("disk_id"), leave=True):
            labels.append(1 if group['label'].sum()>0 else 0)
            predict_result_sum = 0  # 각 디스크의 모든 예측값의 합을 저장할 변수
            for i in range(group.shape[0]-window_size):
                sequence = group.iloc[i:i + window_size][FEATURE_COLUMNS]              
                sequence = torch.Tensor(sequence.to_numpy().T.reshape(1,len(FEATURE_COLUMNS),window_size)).to(DEVICE)
                predict = model(sequence)
                predict_result = torch.argmax(predict, dim=1)  # 가장 확률이 높은 클래스의 인덱스
                predict_result_sum += (predict_result == 1).sum().item()  # 클래스 인덱스가 1인 경우만 합산
            predictions.append(1 if predict_result_sum > 0 else 0)  # 하나라도 실패한 예측이 있으면 1, 아니면 0
            disk_id_list.append(disk_id)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    FDR = recall_score(labels, predictions) * 100
    FAR = (1 - precision_score(labels, predictions)) * 100
    print(classification_report(labels, predictions, target_names=['healthy', 'failed'], digits=4))
    print('\n')
    print(confusion_matrix(labels, predictions))
    print('\n')
    print(f"""Final test result : Acc : {accuracy:.4f},
          Macro_f1 : {macro_f1:.4f}, 
          Weighted_f1 : {weighted_f1:.4f}, 
          FDR : {FDR:.4f}, 
          FAR : {FAR:.4f}""")
    
    df = pd.DataFrame({'disk_id':disk_id_list,
                       'label':labels,
                       'pred':predictions})
    now = dt.datetime.now()
    df.to_csv(f'./test_result/test_result_{now.month}_{now.day}_{now.hour}.csv',index=False)

    return accuracy, macro_f1, weighted_f1, FDR, FAR

exp_path = os.path.join('./',
                        'dataset={0}_models={1}_loss={2}_opt={3}_lr_s={4}_epochs={5}_bs={6}_alpha_{7}'.format(
                            args.dataset,
                            args.model,
                            args.loss,
                            args.op,
                            args.lr_s, args.num_epochs,
                            args.batch_size, args.alpha),
                        args.noise_mode + str(args.r) + '_es=' + str(
                            estimated_es[args.dataset + '_' + args.noise_mode + str(args.r)]) + '_seed=' + str(
                            args.seed))
if not os.path.isdir(exp_path):
    os.makedirs(exp_path)

t = torch.zeros(50000, args.num_class).to(args.gpuid)
cont = 0
acc_train_per_epoch_model = np.array([])
loss_train_per_epoch_model = np.array([])
acc_val_per_epoch_model = np.array([])
loss_val_per_epoch_model = np.array([])

for epoch in range(1, args.num_epochs + 1):

    loss_train_per_epoch, acc_train_per_epoch = train(
        args,
        model,
        all_trainloader,
        optimizer,
        epoch)
    # scheduler.step()

    # note that we check the accuracy for each epoch below, but the accuracy in paper is recorded from the last epoch
    loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args.batch_size, model, args.gpuid, test_loader)

    acc_train_per_epoch_model = np.append(acc_train_per_epoch_model, acc_train_per_epoch)
    loss_train_per_epoch_model = np.append(loss_train_per_epoch_model, loss_train_per_epoch)
    acc_val_per_epoch_model = np.append(acc_val_per_epoch_model, acc_val_per_epoch_i)
    loss_val_per_epoch_model = np.append(loss_val_per_epoch_model, loss_per_epoch)


    if epoch == 1:
        best_acc_val = acc_val_per_epoch_i[-1]
        snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.1f_bestAccVal_%.5f' % (
            epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.r, best_acc_val)
        torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
    else:
        if acc_val_per_epoch_i[-1] > best_acc_val:
            best_acc_val = acc_val_per_epoch_i[-1]
            if cont > 0:
                try:
                    os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                    os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    # os.remove(os.path.join(exp_path, lossBest))
                except OSError:
                    pass
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.1f_bestAccVal_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.r, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

    cont += 1

    if epoch == args.num_epochs:
        torch.save(model.state_dict(), os.path.join(exp_path, 'model_last.pth'))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_last.pth'))

print('\n')
print('========== Test per Disk ==========')
accuracy, macro_f1, weighted_f1, FDR, FAR = per_disk_test(model, test_scaled)

end_time = datetime.now()
print(f"End time: {end_time.strftime('%Y/%m/%d %H:%M:%S')}")

# save corrected labels
_, corrected_labels = torch.max(criterion.soft_labels, dim=1)
correct_num = np.sum(corrected_labels.cpu().numpy() == np.array(clean_labels))
corrected_acc = correct_num / len(clean_labels)
print('Corrected accuracy = {}/{} = {}'.format(correct_num, len(clean_labels), corrected_acc))

np.save(os.path.join(exp_path, 'corrected_labels_%.4f.npy' % (corrected_acc)),
        corrected_labels.cpu().numpy())

np.save(os.path.join(exp_path, 'acc_train_per_epoch_model.npy'), acc_train_per_epoch_model)
np.save(os.path.join(exp_path, 'loss_train_per_epoch_model.npy'), loss_train_per_epoch_model)
np.save(os.path.join(exp_path, 'acc_val_per_epoch_model.npy'), acc_val_per_epoch_model)
np.save(os.path.join(exp_path, 'loss_val_per_epoch_model.npy'), loss_val_per_epoch_model)
