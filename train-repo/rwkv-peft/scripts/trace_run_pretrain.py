import argparse
import datetime
import math
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset


def main() -> None:
    sys.path.insert(0, os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", default="out_trace/L12-D768-x070/rwkv-init.pth")
    parser.add_argument("--proj_dir", default="out_trace/L12-D768-x070")
    parser.add_argument("--data_file", default="../../data/minipile")
    parser.add_argument("--vocab_size", default=65536, type=int)
    parser.add_argument("--n_layer", default=12, type=int)
    parser.add_argument("--n_embd", default=768, type=int)
    parser.add_argument("--ctx_len", default=512, type=int)
    parser.add_argument("--micro_bsz", default=16, type=int)
    parser.add_argument("--epoch_steps", default=2520, type=int)
    parser.add_argument("--magic_prime", default=2926181, type=int)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)
    args = parser.parse_args()

    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = "64"
    os.environ["RWKV_TRAIN_TYPE"] = "none"
    os.environ["WKV"] = "cuda"
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["RWKV_FLOAT_MODE"] = "bf16"
    os.environ["RWKV_JIT_ON"] = "0"

    from lightning import Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy
    from rwkvt.args_type import TrainingArgs
    from rwkvt.dataset.binidx import MMapIndexedDataset
    from rwkvt.lightning_train.light_rwkv import RWKV
    from rwkvt.lightning_train.trainer import train_callback
    from rwkvt.rwkv7.model import RWKV7

    class TraceAlignedDataset(Dataset):
        def __init__(self, train_args):
            self.args = train_args
            self.data = MMapIndexedDataset(train_args.data_file)

        def __len__(self):
            return self.args.micro_bsz

        def __getitem__(self, idx):
            ctx_len = self.args.ctx_len
            req_len = ctx_len + 1
            ii = 1 + idx
            factor = int(self.args.magic_prime * ((math.sqrt(5) - 1) / 2))
            offset = ((factor * ii * ii * ii) % self.args.magic_prime) * ctx_len
            dix = self.data.get(idx=0, offset=offset, length=req_len).astype(int)
            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)
            return x, y

    train_args = TrainingArgs()
    train_args.load_model = args.load_model
    train_args.proj_dir = args.proj_dir
    train_args.data_file = args.data_file
    train_args.data_type = "binidx"
    train_args.vocab_size = args.vocab_size
    train_args.n_layer = args.n_layer
    train_args.n_embd = args.n_embd
    train_args.ctx_len = args.ctx_len
    train_args.micro_bsz = args.micro_bsz
    train_args.epoch_steps = args.epoch_steps
    train_args.epoch_count = 1
    train_args.epoch_begin = 0
    train_args.lr_init = 6e-4
    train_args.lr_final = 6e-5
    train_args.warmup_steps = 10
    train_args.beta1 = 0.9
    train_args.beta2 = 0.99
    train_args.adam_eps = 1e-8
    train_args.grad_cp = 0
    train_args.weight_decay = 0
    train_args.weight_decay_final = -1
    train_args.layerwise_lr = 1
    train_args.ds_bucket_mb = args.ds_bucket_mb
    train_args.magic_prime = args.magic_prime
    train_args.my_testing = "x070"
    train_args.head_size_a = 64
    train_args.head_size_divisor = 8
    train_args.peft = "none"
    train_args.train_type = "none"
    train_args.dataload = "pad"
    train_args.loss_mask = "none"
    train_args.optimizer = "none"
    train_args.lr_schedule = "cos"
    train_args.num_workers = 0
    train_args.dim_att = args.n_embd
    train_args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
    train_args.precision = "bf16"
    train_args.accelerator = "gpu"
    train_args.devices = 1
    train_args.num_nodes = 1
    train_args.strategy = "deepspeed_stage_2"
    train_args.accumulate_grad_batches = 1
    train_args.enable_checkpointing = False
    train_args.logger = False
    train_args.gradient_clip_val = 1.0
    train_args.num_sanity_val_steps = 0
    train_args.check_val_every_n_epoch = int(1e20)
    train_args.log_every_n_steps = int(1e20)
    train_args.max_epochs = 1
    train_args.betas = (train_args.beta1, train_args.beta2)
    train_args.real_bsz = train_args.devices * train_args.num_nodes * train_args.micro_bsz
    train_args.run_name = f"{train_args.vocab_size} ctx{train_args.ctx_len} L{train_args.n_layer} D{train_args.n_embd}"
    train_args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

    os.makedirs(train_args.proj_dir, exist_ok=True)
    model = RWKV7(train_args)
    state_dict = torch.load(train_args.load_model, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state_dict, strict=False, assign=True)
    lightning_model = RWKV(train_args, model=model)

    bucket = train_args.ds_bucket_mb * 1000 * 1000
    strategy = DeepSpeedStrategy(stage=2, allgather_bucket_size=bucket, reduce_bucket_size=bucket)
    trainer = Trainer(
        accelerator=train_args.accelerator,
        strategy=strategy,
        devices=train_args.devices,
        num_nodes=train_args.num_nodes,
        precision=train_args.precision,
        logger=train_args.logger,
        callbacks=[train_callback(train_args)],
        max_epochs=train_args.max_epochs,
        check_val_every_n_epoch=train_args.check_val_every_n_epoch,
        num_sanity_val_steps=train_args.num_sanity_val_steps,
        log_every_n_steps=train_args.log_every_n_steps,
        enable_checkpointing=train_args.enable_checkpointing,
        accumulate_grad_batches=train_args.accumulate_grad_batches,
        gradient_clip_val=train_args.gradient_clip_val,
        limit_train_batches=1,
        use_distributed_sampler=False,
    )
    train_loader = DataLoader(
        TraceAlignedDataset(train_args),
        batch_size=train_args.micro_bsz,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    trainer.fit(lightning_model, train_loader)


if __name__ == "__main__":
    main()
