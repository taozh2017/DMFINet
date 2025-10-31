import re
import matplotlib.pyplot as plt

# 定义文件路径
log_file_path = 'EN2CTtrain.log'

# 初始化列表存储 re_loss 值、D_A 值和横坐标
re_losses = []
d_a_values = []
x_labels = []

# 读取文件并提取符合特定格式的行中的 re_loss 值、D_A 值和横坐标
with open(log_file_path, 'r') as f:
    for line in f:
        # 使用正则表达式匹配符合 "102 / 193,  80 / 100" 格式的行
        match = re.search(r'97 / 386,\s+(\d+) / \d+', line)
        if match:
            step_value = int(match.group(1))
            
            # 使用正则表达式匹配 re_loss 值
            re_loss_match = re.search(r're_loss: ([0-9.]+)', line)
            # 使用正则表达式匹配 D_A 值
            d_a_match = re.search(r'D_A: ([0-9.]+)', line)

            # 确保同时匹配到 re_loss 和 D_A 才进行记录
            if re_loss_match and d_a_match:
                re_loss = float(re_loss_match.group(1))
                d_a = float(d_a_match.group(1))

                x_labels.append(step_value)
                re_losses.append(re_loss)
                d_a_values.append(d_a)

# 检查提取的数据
print(f"x_labels: {x_labels}")
print(f"re_losses: {re_losses}")
print(f"d_a_values: {d_a_values}")

# 绘制 re_loss 和 D_A 的变化曲线
plt.figure(figsize=(10, 5))
plt.plot(x_labels, re_losses, marker='o', linestyle='-', color='b', label='Reconstruction Loss')
plt.plot(x_labels, d_a_values, marker='x', linestyle='--', color='r', label='D_A Value')
plt.xlabel('Step in Epoch')
plt.ylabel('Value')
plt.title('Reconstruction Loss and D_A over Steps')
plt.legend()
plt.grid(True)
plt.savefig('loss_and_da_curve.png')
plt.show()
