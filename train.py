import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import swanlab
from model import swin_cifar
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# 设置随机种子函数，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)                 # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed_all(seed)        # 设置PyTorch GPU随机种子
    np.random.seed(seed)                    # 设置NumPy随机种子
    torch.backends.cudnn.deterministic = True # 确保cudnn使用确定性算法

# 调用设置随机种子函数
set_seed()

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#fw Gpu,原：device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用设备: {device}, cuda_available: {torch.cuda.is_available()}')
if device == 'cuda':
    print(f'CUDA device count: {torch.cuda.device_count()}, current: {torch.cuda.current_device()}, name: {torch.cuda.get_device_name(0)}')
    # 打印显存使用情况（调试用）
    try:
        import torch.cuda
        print("CUDA memory allocated:", torch.cuda.memory_allocated() / (1024**2), "MB")
        print("CUDA memory reserved :", torch.cuda.memory_reserved() / (1024**2), "MB")
    except Exception:
        pass


def rand_bbox(size, lam):
    # size: (N, C, H, W)
    _, _, H, W = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # center
    cx = np.random.randint(0, W)
    cy = np.random.randint(0, H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bby1, bbx1, bby2, bbx2  # 注意返回(H1,W1,H2,W2)顺序，配合切片时使用
# 超参数设置
batch_size = 128
epochs = 400
learning_rate = 1e-4
weight_decay = 5e-4
num_classes = 10  # CIFAR-10 类别数

# 数据增强和标准化
# 训练集增强：随机裁剪、水平翻转、标准化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 测试集只进行标准化
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10数据集
print('加载CIFAR-10数据集...')
trainset = torchvision.datasets.CIFAR10(
    root='../data/cifar10', train=True, download=False, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=batch_size, shuffle=True, num_workers=4)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True if 4>0 else False)
testset = torchvision.datasets.CIFAR10(
    root='../data/cifar10', train=False, download=False, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=batch_size, shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True if 4>0 else False)


# 定义类别
classes = ('飞机', '汽车', '鸟', '猫', '鹿', 
           '狗', '青蛙', '马', '船', '卡车')

# 初始化模型
print("构建SwinTransformer模型...")
net = swin_cifar(patch_size=2, n_classes=num_classes, mlp_ratio=1).to(device)

# 计算模型参数量
def count_parameters(model):
    """计算模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(net)
print(f"模型总参数量: {total_params:,}")
print(f"参数量(MB): {total_params * 4 / (1024 * 1024):.2f} MB")  # 假设每个参数是4字节(float32)

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 'min', factor=0.94, patience=1, min_lr=0.000001)  # 动态更新学习率
#scheduler.step(test_loss)   # ReduceLROnPlateau注意这里必须传 test_loss
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
#scheduler.step()   # CosineAnnealingLR不需要传参数
# 混合精度
scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

# 初始化SwanLab进行实验追踪
run = swanlab.init(
    project="cifar10-swin",
    name="swin-transformer",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "architecture": "SwinTransformer",
        "device": device,
        "parameters": total_params,         # 记录参数量
        "parameters_MB": total_params * 4 / (1024 * 1024)  # 参数量(MB)
    }
)

# 计算Top-K准确率的辅助函数
def calculate_topk_accuracy(outputs, targets, topk=(1, 5)):
    """
    计算Top-K准确率
    Args:
        outputs: 模型输出的预测结果
        targets: 真实标签
        topk: 要计算的top-k值
    Returns:
        Top-K准确率列表
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

# 训练函数
def train(epoch):
    print(f'\n第 {epoch} 轮训练开始')
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm显示进度条
    with tqdm(trainloader, desc=f'Epoch {epoch}', ncols=100) as pbar:
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # --- CutMix 随机启用 ---
            r = np.random.rand()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):   #此行开始改动
                if r < 0.5:
                    lam = np.random.beta(1.0, 1.0)
                    rand_index = torch.randperm(inputs.size(0)).to(device)
                    target_a = targets
                    target_b = targets[rand_index]

                    y1, x1, y2, x2 = rand_bbox(inputs.size(), lam)
                    # 注意：切片顺序 (N,C,H,W) -> inputs[:, :, y1:y2, x1:x2]
                    inputs[:, :, y1:y2, x1:x2] = inputs[rand_index, :, y1:y2, x1:x2]
                    # 重新计算 lam 为实际区域比例
                    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (inputs.size(2) * inputs.size(3)))
                    outputs = net(inputs)
                    loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
                else:
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            # 可选：梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'Loss': f'{train_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.3f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        train_loss /= len(trainloader)
        train_acc = 100. * correct / total

        swanlab.log({"train/loss": train_loss, "train/accuracy": train_acc, "lr": optimizer.param_groups[0]['lr']})
        return train_loss, train_acc
    #         if r < 0.5:
    #             lam = np.random.beta(1.0, 1.0)
    #             rand_index = torch.randperm(inputs.size()[0]).to(device)
    #             target_a = targets
    #             target_b = targets[rand_index]
    #             bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    #             inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
    #             lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
    #             outputs = net(inputs)
    #             loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
    #         else:
    #             outputs = net(inputs)
    #             loss = criterion(outputs, targets)
    #
    #         # 梯度清零
    #         optimizer.zero_grad()
    #
    #         # 前向传播 + 反向传播 + 优化
    #     # outputs = net(inputs)
    #     #loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # 统计
    #         train_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()
    #
    #         # 更新进度条信息
    #         pbar.set_postfix({
    #             'Loss': f'{train_loss/(batch_idx+1):.3f}',
    #             'Acc': f'{100.*correct/total:.3f}%'
    #         })
    #
    # # 计算平均损失和准确率
    # train_loss /= len(trainloader)
    # train_accuracy = 100. * correct / total
    #
    # # 记录指标到SwanLab
    # swanlab.log({
    #     "train/loss": train_loss,
    #     "train/accuracy": train_accuracy,
    #     "lr": optimizer.param_groups[0]['lr']
    # })
    #
    # print(f'训练集损失: {train_loss:.3f} | 训练集准确率: {train_accuracy:.2f}%')
    # return train_loss, train_accuracy

# 测试函数
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    print('测试中...')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
             outputs = net(inputs)
             loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            
            # 计算Top-1和Top-5准确率
            top1_acc, top5_acc = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
            
            # 计算Top-1准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_top1 += predicted.eq(targets).sum().item()
            
            # 计算Top-5准确率 (对于CIFAR-10，只有10个类别，所以最多取min(5, 10))
            _, pred_top5 = outputs.topk(min(5, num_classes), 1, largest=True, sorted=True)
            correct_top5 += torch.eq(pred_top5, targets.view(-1, 1).expand_as(pred_top5)).sum().item()
            
            # 计算每个类别的准确率
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 计算平均损失和准确率
    test_loss /= len(testloader)
    test_acc_top1 = 100. * correct_top1 / total  # Top-1准确率
    test_acc_top5 = 100. * correct_top5 / total  # Top-5准确率
    
    # 计算错误率
    test_err_top1 = 100. - test_acc_top1    # Top-1错误率
    test_err_top5 = 100. - test_acc_top5    # Top-5错误率
    
    # 打印测试统计信息
    print(f'测试集损失: {test_loss:.3f} | 测试集Top-1准确率: {test_acc_top1:.2f}% ({correct_top1}/{total})')
    print(f'Top-5 准确率: {test_acc_top5:.2f}% (错误率: {test_err_top5:.2f}%)')
    
    # 打印每个类别的准确率
    for i in range(num_classes):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'{classes[i]} 类准确率: {class_acc:.2f}%')
    
    # 记录指标到SwanLab
    metrics = {
        "test/loss": test_loss,
        "test/accuracy_top1": test_acc_top1,
        "test/accuracy_top5": test_acc_top5,
        "test/error_top1": test_err_top1,
        "test/error_top5": test_err_top5
    }
    
    # 记录每个类别的准确率
    for i in range(num_classes):
        metrics[f"test/class_acc/{classes[i]}"] = 100 * class_correct[i] / class_total[i]
    
    swanlab.log(metrics)
    
    # 保存最佳模型 (基于Top-1准确率)
    if test_acc_top1 > best_acc:
        print('保存最佳模型...')
        state = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': test_acc_top1,
            'params_count': total_params
        }
        torch.save(state, 'cifar10_swin_best_88.85.pth')
        best_acc = test_acc_top1
    
    return test_loss, test_acc_top1, test_acc_top5, test_err_top1, test_err_top5, {classes[i]: 100 * class_correct[i] / class_total[i] for i in range(num_classes)}

# 主训练循环
if __name__ == "__main__":
    best_acc = 0  # 最佳准确率
    start_epoch = 0  # 起始轮次
    # 如果存在检查点，则从检查点恢复
    if os.path.exists('cifar10_swin_best_88.85.pth'):
        print('从检查点恢复训练...')
        checkpoint = torch.load('cifar10_swin_best_88.85.pth')
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f'从第 {start_epoch} 轮继续训练，最佳准确率: {best_acc:.2f}%')
    
    print(f'开始训练，共 {epochs} 轮...')
    start_time = time.time()
    
    # 训练轮次循环
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # 训练和测试
        train_loss, train_acc = train(epoch)
        test_loss, test_acc_top1, test_acc_top5, test_err_top1, test_err_top5, class_acc = test(epoch)
        
        # 更新学习率,根据调度器来
        scheduler.step()

        
        # 记录每轮时间
        epoch_time = time.time() - epoch_start
        swanlab.log({"time/epoch": epoch_time})
        
        print(f'第 {epoch} 轮完成，耗时 {epoch_time:.2f}秒 | '
              f'训练损失: {train_loss:.3f}, 训练准确率: {train_acc:.2f}% | '
              f'测试Top-1准确率: {test_acc_top1:.2f}%, Top-5准确率: {test_acc_top5:.2f}% | '
              f'最佳准确率: {best_acc:.2f}%')
        
        # 每10轮保存一次模型
        if (epoch + 1) % 10 == 0:
            print(f'保存第 {epoch+1} 轮模型...')
            torch.save(net.state_dict(), f'./cifar10_swin_epoch{epoch+1}.pth')
    
    # 训练结束，计算总时间
    total_time = time.time() - start_time
    print(f'训练完成，共耗时 {total_time/60:.2f} 分钟')
    print(f'最佳准确率: {best_acc:.2f}%')
    
    # 保存最终模型
    print('保存最终模型...')
    torch.save(net.state_dict(), 'SwinTransformer_cifar10_final_90.57.pth')
    
    # 测试最终模型性能
    print("使用最佳模型进行最终测试...")
    net.load_state_dict(torch.load('cifar10_swin_best_88.85.pth')['model'])
    final_test_loss, final_acc_top1, final_acc_top5, final_err_top1, final_err_top5, final_class_acc = test(-1)  # 使用-1表示最终测试
    
    print(f"最终测试结果:")
    print(f"  - Top-1 准确率: {final_acc_top1:.2f}% (错误率: {final_err_top1:.2f}%)")
    print(f"  - Top-5 准确率: {final_acc_top5:.2f}% (错误率: {final_err_top5:.2f}%)")
    print(f"  - 模型参数量: {total_params:,} ({total_params * 4 / (1024 * 1024):.2f} MB)")
    
    # 记录最终结果
    swanlab.log({
        "final/test_loss": final_test_loss,
        "final/accuracy_top1": final_acc_top1,
        "final/accuracy_top5": final_acc_top5,
        "final/error_top1": final_err_top1,
        "final/error_top5": final_err_top5,
        "model_parameters": total_params,
        "model_size_mb": total_params * 4 / (1024 * 1024)
    })
    
    # 关闭SwanLab记录
    run.finish()