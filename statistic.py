import pandas as pd
import matplotlib.pyplot as plt
import os

log_folder = 'log/south_pole/03.27'
log_path = os.path.join(log_folder, 'log.txt')
progress_path = os.path.join(log_folder, 'progress.csv')
fig_path = os.path.join(log_folder, 'progress.png')

data = pd.read_csv(progress_path)
print(data.head())  # 打印前几行数据以确认读取成功

# 绘制以 loss 为纵轴的折线图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(data['step'], data['loss'], label='Loss', color='blue')  # 绘制折线
plt.xlabel('Step')  # X 轴标签
plt.ylabel('Loss')  # Y 轴标签
plt.title('Loss over Steps')  # 图表标题
plt.legend()  # 显示图例
plt.grid()  # 添加网格
plt.savefig(fig_path)  # 保存 progress 折线图
plt.show()  # 显示图形


filtered_data = data[data['step'] % 10000 == 0]  # 筛选 step 值能整除 10000 的行
min_loss_row = filtered_data.loc[filtered_data['loss'].idxmin()]  # 找到 loss 最小的行

min_loss_step = min_loss_row['step']  # 获取对应的 step 值
min_loss_value = min_loss_row['loss']  # 获取对应的 loss 值

print(f'The step value with the minimum loss (where step is divisible by 10000) is: {min_loss_step}, with loss: {min_loss_value}')