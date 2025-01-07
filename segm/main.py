import sys
import yaml
import json
from pathlib import Path
import argparse

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.utils import NativeScaler

import config as config
from utils import distributed
import utils.torch_utils as ptu
from utils.model_utils import num_params
from utils.parsing import bool_flag
from model_factory import get_model
from optim.factory import create_optimizer, create_scheduler
from data.factory import create_dataset
from utils.distributed import sync_model
from engine import train_one_epoch, evaluate


def main(args):
    # start distributed mode
    ptu.set_gpu_mode(True)
    distributed.init_process()

    # set up output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pth"

    # set up configuration
    cfg = config.load_config()
    dataset_cfg = cfg["dataset"][args.dataset]
    decoder_cfg = cfg["decoder"]["linear"]

    # if not present in args, use config values
    if not args.input_size:
        args.input_size = dataset_cfg["input_size"]
    if not args.crop_size:
        args.crop_size = dataset_cfg.get("crop_size", args.input_size)
    if not args.window_size:
        args.window_size = dataset_cfg.get("window_size", args.input_size)
    if not args.window_stride:
        args.window_stride = dataset_cfg.get("window_stride", args.input_size)
    if not args.batch_size:
        args.batch_size = dataset_cfg["batch_size"]
    if not args.epochs:
        args.epochs = dataset_cfg["epochs"]
    if not args.lr:
        args.lr = dataset_cfg["learning_rate"]
    if not args.eval_freq:
        args.eval_freq = dataset_cfg.get("eval_freq", 1)

    model_cfg = {}
    model_cfg["image_size"] = (args.crop_size, args.crop_size)
    model_cfg["dropout"] = args.dropout
    model_cfg["decoder"] = decoder_cfg
    model_cfg["nb_classes"] = args.nb_classes
    model_cfg["input_size"] = args.crop_size
    model_cfg["patch_size"] = args.patch_size
    model_cfg["global_pool"] = args.global_pool
    model_cfg["embed_dim"] = args.embed_dim
    model_cfg["depth"] = args.depth
    model_cfg["num_heads"] = args.num_heads
    model_cfg["mlp_ratio"] = args.mlp_ratio
    model_cfg["drop_path"] = args.drop_path
    model_cfg["checkpoint"] = args.checkpoint

    # Cropr config
    cropr_cfg = {"use_cropr": args.use_cropr}
    if args.use_cropr:
        cropr_cfg.update(
            {
                "pruning_rate": args.cropr_pruning_rate,
                "num_queries": args.cropr_num_queries,
                "num_heads": args.cropr_num_heads,
                "pre_attn_norm": args.cropr_pre_attn_norm,
                "q_proj": args.cropr_q_proj,
                "k_proj": args.cropr_k_proj,
                "v_proj": args.cropr_v_proj,
                "mlp": args.cropr_mlp,
                "mlp_ratio": args.cropr_mlp_ratio,
                "epochs": args.epochs,
                "training": True,
            }
        )
    args.cropr_cfg = cropr_cfg
    model_cfg["cropr_cfg"] = args.cropr_cfg

    # experiment config
    bs = args.batch_size // ptu.world_size
    variant = dict(
        world_batch_size=args.batch_size,
        version="normal",
        resume=checkpoint_path.exists(),
        dataset_kwargs=dict(
            dataset=args.dataset,
            image_size=args.input_size,
            crop_size=args.crop_size,
            batch_size=bs,
            normalization=args.normalization,
            split="train",
            num_workers=args.num_workers,
        ),
        algorithm_kwargs=dict(
            batch_size=bs,
            start_epoch=0,
            num_epochs=args.epochs,
            eval_freq=args.eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=args.optimizer,
            lr=args.lr,
            weight_decay=args.weight_decay,
            layer_decay=args.layer_decay,
            momentum=0.9,
            clip_grad=None,
            sched=args.scheduler,
            epochs=args.epochs,
            min_lr=0.0,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=args.use_fp16,
        output_dir=args.output_dir,
        inference_kwargs=dict(
            im_size=args.input_size,
            window_size=args.window_size,
            window_stride=args.window_stride,
        ),
    )

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False
    val_loader = create_dataset(val_kwargs)

    model = get_model(args)
    model.to(ptu.device)

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = int(
        len(train_loader) * optimizer_kwargs["epochs"] / args.accum_steps
    )
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, model)
    lr_scheduler = create_scheduler(opt_args, optimizer)
    loss_scaler = NativeScaler() if args.use_fp16 else None

    # resume
    if checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1
    else:
        sync_model(output_dir, model)

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant_str = yaml.dump(variant)
    print(f"Configuration:\n{variant_str}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    val_seg_gt = val_loader.dataset.get_gt_seg_maps()

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")
    print(f"Model parameters: {num_params(model_without_ddp)}")

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train_logger = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            loss_scaler,
            args,
        )

        # save checkpoint
        if ptu.dist_rank == 0:
            snapshot = dict(
                model=model_without_ddp.state_dict(),
                optimizer=optimizer.state_dict(),
                n_cls=args.nb_classes,
                lr_scheduler=lr_scheduler.state_dict(),
            )
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            snapshot["epoch"] = epoch
            torch.save(snapshot, checkpoint_path)

        # evaluate
        eval_epoch = epoch % args.eval_freq == 0 or epoch == args.epochs - 1
        if eval_epoch:
            eval_logger = evaluate(
                model,
                val_loader,
                val_seg_gt,
                args.window_size,
                args.window_stride,
                args,
            )
            print(f"Stats [{epoch}]:", eval_logger, flush=True)
            print("")

        # log stats
        if ptu.dist_rank == 0:
            train_stats = {
                k: meter.global_avg for k, meter in train_logger.meters.items()
            }
            val_stats = {}
            if eval_epoch:
                val_stats = {
                    k: meter.global_avg for k, meter in eval_logger.meters.items()
                }

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "num_updates": (epoch + 1) * len(train_loader),
            }

            with open(output_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training on ADE20k.")

    ### Model args
    parser.add_argument(
        "--dataset",
        type=str,
        default="ade20k",
    )
    parser.add_argument(
        "--use_cropr",
        type=bool_flag,
        default=True,
        help="Whether or not to apply Cropr.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Pretraining checkpoint to be loaded from timm.",
    )
    parser.add_argument(
        "--input_size",
        default=None,
        type=int,
        help="dataset resize size",
    )
    parser.add_argument(
        "--crop_size",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--window_size",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--window_stride",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="Patch size of the model.",
    )
    parser.add_argument(
        "--global_pool",
        default="avg",
        type=str,
    )
    parser.add_argument(
        "--embed_dim",
        default=1024,
        type=int,
        help="Embedding dimension of the model.",
    )
    parser.add_argument(
        "--depth",
        default=24,
        type=int,
        help="Depth of the model.",
    )
    parser.add_argument(
        "--num_heads",
        default=16,
        type=int,
        help="Number of heads in the model.",
    )
    parser.add_argument(
        "--mlp_ratio",
        default=2.6666666666666665,
        type=float,
        help="MLP ratio in the model.",
    )
    parser.add_argument(
        "--nb_classes",
        default=150,
        type=int,
        help="Number of segmentation labels.",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.2,
        help="Drop path rate.",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.9,
        help="Layer-wise learning rate decay value. Last layers will have higher learning rate than first layers.",
    )

    ### Paths
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Path to save logs and checkpoints.",
    )

    ### Training args
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of data loading workers (per GPU).",
    )
    parser.add_argument(
        "--epochs", default=64, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay value.",
    )
    parser.add_argument(
        "--use_fp16",
        type=bool_flag,
        default=True,
        help="Whether or not to use half precision for training. Improves training time and memory requirements.",
    )
    parser.add_argument(
        "--lr",
        default=0.00002,
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accum_steps",
        default=1,
        type=int,
        help="Accumulate gradients on several steps before performing a step with the optimizer.",
    )
    parser.add_argument(
        "--max_norm",
        default=None,
        type=float,
        help="Max norm for the gradients if using gradient clipping",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
    )
    parser.add_argument(
        "--scheduler",
        default="polynomial",
        type=str,
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--normalization",
        default="vit",
        type=str,
    )
    parser.add_argument(
        "--eval_freq",
        default=None,
        type=int,
    )

    ### Cropr specific arguments
    parser.add_argument(
        "--cropr_pruning_rate",
        type=int,
        help="Constant pruning rate. Reduce by this amont with each Cropr module.",
    )
    parser.add_argument(
        "--cropr_num_queries",
        default=1,
        type=int,
        help="Number of learnable queries in each Cropr module.",
    )
    parser.add_argument(
        "--cropr_num_heads",
        default=1,
        type=int,
        help="Number of heads in Cropr modules.",
    )
    parser.add_argument(
        "--cropr_pre_attn_norm",
        type=bool_flag,
        default=False,
        help="Whether to apply LayerNorm prior to cross-attention.",
    )
    parser.add_argument(
        "--cropr_q_proj",
        type=bool_flag,
        default=False,
        help="Whether to project the queries in the Cropr module.",
    )
    parser.add_argument(
        "--cropr_k_proj",
        type=bool_flag,
        default=False,
        help="Whether to project the keys in the Cropr module.",
    )
    parser.add_argument(
        "--cropr_v_proj",
        type=bool_flag,
        default=False,
        help="Whether to project the values in the Cropr module.",
    )
    parser.add_argument(
        "--cropr_mlp",
        type=bool_flag,
        default=True,
        help="Whether to use an MLP in the Cropr module.",
    )
    parser.add_argument(
        "--cropr_mlp_ratio", type=int, default=4, help="MLP ratio in Cropr module."
    )

    args, unknown = parser.parse_known_args()
    main(args)
