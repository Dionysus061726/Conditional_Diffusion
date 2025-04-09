import argparse
import os

# 设置默认值
image_size = 300

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--image_size', type=int, help='Size of the image', default=image_size)

# 解析命令行参数
args = parser.parse_args()

# 如果 MODEL_FLAGS 环境变量存在，则更新 image_size
model_flags = os.getenv('MODEL_FLAGS')
if model_flags:
    # 解析环境变量中的参数
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--image_size', type=int, help='Size of the image')

    model_args, _ = model_parser.parse_known_args(model_flags.split())

    # 更新 image_size，如果在环境变量中有新的值
    if model_args.image_size is not None:
        image_size = model_args.image_size

# 最终输出 image_size
print(f'Final image size: {image_size}')


