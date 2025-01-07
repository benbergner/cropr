import sys
import math
import torch

from utils.logger import MetricLogger
from utils import model_utils
import utils.torch_utils as ptu
from metrics import gather_data, compute_metrics

from data.utils import IGNORE_LABEL


def train_one_epoch(
    model, data_loader, optimizer, lr_scheduler, epoch, loss_scaler, args
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    if args.use_cropr:
        # apply pruning rate curriculum
        model.set_pruning_rate(epoch)

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = int(epoch * len(data_loader) / args.accum_steps)
    for batch_id, batch in enumerate(logger.log_every(data_loader, print_freq, header)):
        im = batch["im"].to(ptu.device)  # B, C, H, W
        seg_gt = batch["segmentation"].long().to(ptu.device)  # B, H, W

        H, W = seg_gt.shape[-2:]

        with torch.cuda.amp.autocast(enabled=args.use_fp16):

            seg_pred = model.forward(im)

            if isinstance(seg_pred, torch.Tensor):
                loss = criterion(seg_pred, seg_gt)
                losses_itemized = {"loss": loss.item()}
            else:
                loss = 0

                # downsample seg_gt
                down_factor = H // model.patch_size
                seg_gt_down = (
                    torch.nn.functional.interpolate(
                        seg_gt.unsqueeze(1).float(),
                        size=(down_factor, down_factor),
                        mode="nearest",
                    )
                    .squeeze(1)
                    .long()
                )

                losses = {}
                for i, pred in enumerate(seg_pred):
                    loss_name = "main" if i == 0 else f"aux_{i}"
                    if i == 0:
                        losses[loss_name] = criterion(pred, seg_gt)
                    else:
                        losses[loss_name] = criterion(pred.squeeze(0), seg_gt_down)
                loss = sum(losses.values())
                losses_itemized = {"total": loss.item()}
                losses_itemized.update(
                    {f"loss_{i}": l.item() for i, l in losses.items()}
                )

        if not math.isfinite(loss.item()):
            print(
                "Loss is {}, stopping training".format(loss.item()),
                force=True,
            )
            sys.exit(1)

        need_update = True if (batch_id + 1) % args.accum_steps == 0 else False
        if loss_scaler is not None:
            loss_scaler(
                loss / args.accum_steps,
                optimizer,
                parameters=model.parameters(),
                need_update=need_update,
            )
        else:
            (loss / args.accum_steps).backward()
            if need_update:
                optimizer.step()

        if need_update:
            optimizer.zero_grad()
            num_updates += 1
            lr_scheduler.step_update(num_updates=num_updates)

        logger.update(
            **losses_itemized,
            learning_rate=optimizer.param_groups[-1]["lr"],
        )

    return logger


@torch.no_grad()
def evaluate(model, data_loader, val_seg_gt, window_size, window_stride, args):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):

        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with torch.cuda.amp.autocast(enabled=args.use_fp16):
            seg_pred = model_utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger
