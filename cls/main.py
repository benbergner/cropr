import os
import json
import time
import datetime
import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import utils
from model_factory import get_model
from engine import train_one_epoch, validate_network


def main(args):
    # ============ preparing ... ============
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    cropr_cfg = {"use_cropr": args.use_cropr}
    if args.use_cropr:
        cropr_cfg.update(
            {
                "pruning_rate": args.cropr_pruning_rate,
                "llf": args.cropr_llf,
                "num_queries": args.cropr_num_queries,
                "num_heads": args.cropr_num_heads,
                "pre_attn_norm": args.cropr_pre_attn_norm,
                "q_proj": args.cropr_q_proj,
                "k_proj": args.cropr_k_proj,
                "v_proj": args.cropr_v_proj,
                "mlp": args.cropr_mlp,
                "mlp_ratio": args.cropr_mlp_ratio,
                "training": True,
            }
        )
    args.cropr_cfg = cropr_cfg

    cudnn.benchmark = True

    # ============ building network ... ============
    model = get_model(args)
    model.to("cuda")

    print(f"Model built.")

    # ============ preparing data ... ============
    val_transform = utils.build_transform(False, args)
    dataset_val = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_transform = utils.build_transform(True, args)
    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(
        f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs."
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.label_smoothing,
            num_classes=args.nb_classes,
        )

    # ============ preparing loss ... ============
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        loss_fn = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0.0:
        loss_fn = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # ============ preparing optimizer ... ============
    param_groups = utils.param_groups_lrd(
        model,
        args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size * args.accum_steps / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(train_loader) / args.accum_steps,
        warmup_epochs=args.warmup_epochs,
    )

    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "best_acc": 0.0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    if args.eval:
        test_stats = validate_network(val_loader, model, fp16_scaler)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    start_time = time.time()
    print("Starting training !")

    for epoch in range(start_epoch, args.epochs):

        # ============ training one epoch ... ============
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            lr_schedule,
            epoch,
            fp16_scaler,
            mixup_fn,
            args,
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        # ============ evaluating the model ... ============
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, fp16_scaler)
            print(
                f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
            )
            best_acc = max(best_acc, test_stats["acc1"])
            print(f"Max accuracy so far: {best_acc:.2f}%")
            log_stats = {
                **{k: v for k, v in log_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }

        # ============ writing logs ... ============
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            torch.save(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    print(
        "Training of the supervised classifier completed.\n"
        "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training on ImageNet-1k.")

    ### Model args
    parser.add_argument(
        "--use_cropr",
        type=utils.bool_flag,
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
        "--eval",
        type=utils.bool_flag,
        default=False,
        help="Only evaluate model.",
    )
    parser.add_argument(
        "--img_size",
        default=224,
        type=int,
        help="Img size of the model.",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="Patch size of the model.",
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
        default=4.0,
        type=float,
        help="MLP ratio in the model.",
    )
    parser.add_argument(
        "--global_pool",
        default="avg",
        choices=["avg", "token"],
        type=str,
        help="Which final pooling method to use.",
    )
    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="Number of classification labels.",
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
        default=0.75,
        help="Layer-wise learning rate decay value. Last layers will have higher learning rate than first layers.",
    )

    # Paths
    parser.add_argument(
        "--data_path",
        default="/path/to/imagenet/",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Path to save logs and checkpoints.",
    )

    ### Training args
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers (per GPU).",
    )
    parser.add_argument(
        "--epochs", default=50, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--warmup_epochs",
        default=5,
        type=int,
        help="Number of epochs for the linear learning-rate warm up.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay value.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing.",
    )
    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="Whether or not to use half precision for training. Improves training time and memory requirements.",
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.",
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

    ### Augmentation
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),

    ### * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount",
        type=int,
        default=1,
        help="Random erase count (default: 1)",
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    ### * Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0.",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0.",
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    ### Misc
    parser.add_argument(
        "--val_freq",
        default=1,
        type=int,
        help="Epoch frequency for validation.",
    )
    parser.add_argument(
        "--saveckp_freq",
        default=100,
        type=int,
        help="Save checkpoint every x epochs.",
    )

    ### Cropr specific arguments
    parser.add_argument(
        "--cropr_pruning_rate",
        type=int,
        help="Constant pruning rate. Reduce by this amont with each Cropr module.",
    )
    parser.add_argument(
        "--cropr_llf",
        type=utils.bool_flag,
        default=False,
        help="Whether to use Last Layer Fusion in the Cropr module.",
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
        type=utils.bool_flag,
        default=False,
        help="Whether to apply LayerNorm prior to cross-attention.",
    )
    parser.add_argument(
        "--cropr_q_proj",
        type=utils.bool_flag,
        default=False,
        help="Whether to project the queries in the Cropr module.",
    )
    parser.add_argument(
        "--cropr_k_proj",
        type=utils.bool_flag,
        default=False,
        help="Whether to project the keys in the Cropr module.",
    )
    parser.add_argument(
        "--cropr_v_proj",
        type=utils.bool_flag,
        default=False,
        help="Whether to project the values in the Cropr module.",
    )
    parser.add_argument(
        "--cropr_mlp",
        type=utils.bool_flag,
        default=True,
        help="Whether to use an MLP in the Cropr module.",
    )
    parser.add_argument(
        "--cropr_mlp_ratio", type=int, default=4, help="MLP ratio in Cropr module."
    )

    args, unknown = parser.parse_known_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
