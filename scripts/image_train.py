"""
Train a diffusion model on images.
"""

import argparse
import time
import os
import sys
from visdom import Visdom

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

# Debug时，手动设置环境变量OPENAI_LOGDIR和NCCL_P2P_DISABLE。
# OPENAI_LOGDIR：模型、日志、指标的储存路径。
# NCCL_P2P_Disable：禁用P2P（点对点）通信。
# P2P（Peer-to-Peer）通信允许GPU直接访问另一个GPU的内存，绕过CPU和系统内存，是多GPU系统中高效数据交换的另一种方式。
# 当NCCL_P2P_Disable设置为非零值（同样通常是1），它会禁用P2P通信。
# 禁用P2P后，GPU之间的直接数据传输不再可行，数据需要通过其他途径（如共享内存或网络）来传输。
# 这意味着，即使共享内存未被明确禁用，NCCL在某些情况下也不得不依赖于它，因为P2P这一更高效的直接路径已被关闭。
os.environ.setdefault('OPENAI_LOGDIR', '../log/debug/12.31')
os.environ.setdefault('NCCL_P2P_DISABLE', '1')

# visdom使用：python -m visdom.server

def main():
    start_time = time.time()  # 启动计时

    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())  # **用于解包字典
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir_sar=args.data_dir_sar,
        data_dir_opt=args.data_dir_opt,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    viz = Visdom()
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        viz=viz,
    ).run_loop()

    end_time = time.time()  # 结束计时
    elapsed_time = end_time - start_time  # 计算耗时
    print(f'训练完毕，迭代退火步数 {args.lr_anneal_steps}，训练耗时：{elapsed_time/3600}小时')  # 打印耗时（小时）


def create_argparser():
    defaults = dict(
        data_dir_sar="",  # sar影像文件夹路径
        data_dir_opt="",  # opt影像文件夹路径
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=50000,  # 退火学习步数。11.28尝试，是否是在这一步设置迭代次数，若为0或None则无限学习不自动停止，若为具体值则在达到值后停止学习
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,  # log记录间隔
        save_interval=10000,  # 模型储存间隔
        resume_checkpoint="",  # 继续训练起点
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # x = os.environ.get('OPENAI_LOGDIR')
    # print(f'xtype: {type(x)}, x={x}')
    # y = os.environ.get('NCCL_P2P_DISABLE')
    # print(f'ytype: {type(y)}, y={y}')

    main()