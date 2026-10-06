import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import model

import shift

# 参数设置
dim = 128
Steps = 10000  # 迭代步数
LR = 0.0001  # 学习率
noise_level = 1.0 / 30

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 创建结果保存路径
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


result_save_path = './results/'
model_save_path = './models/'
weight_path = 'models/exp_step_40000_lr_0.000100_noise0.025__.pth'
mkdir(result_save_path)
mkdir(model_save_path)
model_save_path = os.path.join(model_save_path, f'exp_step_{Steps}_lr_{LR:.6f}_noise{noise_level}_学习优化.pth')

# 加载原始数据
HG_path = os.path.join("", f"HG10.npz")
measure_temp = np.load(HG_path)['image']
measure_temp = measure_temp / np.max(measure_temp)
plt.imshow(measure_temp, cmap='gray')
plt.savefig("HG.png")
measure_temp = torch.tensor(measure_temp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# 定义模型
model = model.UNet().to(device)
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path))
    print('successful load weight!')
else:
    print("not successful load weight!")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 定义学习率调整器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000, verbose=True)

# 训练过程
print('开始优化...')
m_loss = np.zeros([Steps, 1])
for step in range(Steps):
    new_rand = torch.rand((1, 1, dim, dim), device=device, dtype=torch.float32) * noise_level
    input_data = measure_temp + new_rand

    optimizer.zero_grad()
    output = model(input_data)
    out_measure = shift.phase_to_amp_ones(output)  # 物理算法把相位恢复到强度
    # gs_phase = shift.gerchberg_saxton(torch.ones_like(input_data), input_data, 10)  # 通过强度得到相位
    loss = criterion(out_measure, input_data)
    # loss002 = 0.001 * criterion(gs_phase, output)
    # loss = loss001 + loss002
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        # print(f'step: {step} loss001: {loss001.item()} loss002：{loss002.item()} loss:{loss.item()}')
        print(f'step: {step} loss:{loss.item()}')
        m_loss[step] = loss.item()

    if step % 100 == 0:
        pha_out = output.squeeze().detach().cpu().numpy()
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(input_data.squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title('input')
        plt.subplot(1, 3, 2)
        plt.imshow(pha_out, cmap='gray')
        plt.title('output')
        plt.subplot(1, 3, 3)
        pha_re = shift.phase_to_amp_ones(output).squeeze().detach().cpu().numpy()
        plt.imshow(pha_re, cmap='gray')
        plt.title('re_input')
        # plt.subplot(1, 5, 4)
        # phase = gs_phase.squeeze().detach().cpu().numpy()
        # plt.imshow(phase, cmap='gray')
        # plt.title('gs_phase')
        # plt.subplot(1, 5, 5)
        # gs_re_input = shift.phase_to_amp_ones(gs_phase).squeeze().detach().cpu().numpy()
        # plt.imshow(gs_re_input, cmap='gray')
        # plt.title('gs_re_input')
        plt.savefig(os.path.join(result_save_path, f'result_step{step}_mse{loss.item()}.png'))
        plt.close()

    if step > 0 and step % 5000 == 0:
        if loss.item() <= np.min(m_loss[:step]):
            min_saved_loss = loss.item()
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved at step {step} with loss {loss.item()}')

    # 更新学习率
    scheduler.step(loss.item())

# 保存最终模型
torch.save(model.state_dict(), model_save_path)
