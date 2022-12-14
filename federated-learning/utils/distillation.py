import torch.nn.functional as F
from torch import nn
import torch

def distilling(teach_net, stu_net,t,lr,d_epoch,ldr_train):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teach_net.eval()
    model = stu_net
    model.to(device)
    model.train()
    T = t  # 蒸馏温度8
    hard_loss = nn.CrossEntropyLoss()  # 设置蒸馏学习的损失函数
    # 设置学习的损失值的权重
    alpha = 0.3
    soft_loss = nn.KLDivLoss(reduction='batchmean')  # 使用一个soft_loss
    # 设置一个优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_loss = []
    for epoch in range(d_epoch):
        batch_loss = []
        # 训练集上训练模型的权重
        for batch_idx, (images, lables) in enumerate(ldr_train):
            images = images.to(device)
            lables = lables.to(device)
            with torch.no_grad():
                teacher_preds = teach_net(images)
                # print(teacher_preds)
                # teacher_preds = teacher_preds.numpy()
            # 学生的模型
            student_preds = model(images)
            # 计算hard_loss
            student_loss = hard_loss(student_preds, lables)
            # 计算蒸馏后的预测结果soft_loss
            ditillation_loss = soft_loss(
                F.log_softmax(student_preds / T, dim=1),
                F.softmax(teacher_preds / T, dim=1)
            )
            # 将hard_loss和soft_loss加权求和  更新loss值
            loss = alpha * student_loss + (1 - alpha) * T * T * ditillation_loss
            # 反向传播 优化权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


