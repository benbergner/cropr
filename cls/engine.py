import sys
import math
import torch
from torch import nn

import utils


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    loss_fn,
    lr_schedule,
    epoch,
    fp16_scaler,
    mixup_fn,
    args,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    grad_i = int(len(data_loader) / args.accum_steps * epoch)
    for data_i, (images, target) in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        # update learning rate according to their schedule
        if data_i % args.accum_steps == 0:
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr_schedule[grad_i] * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr_schedule[grad_i]

        # move to gpu
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if mixup_fn is not None:
            # Batch size should be even when using this
            if images.shape[0] % 2 != 0:
                images = images[: images.shape[0] - 1]
                target = target[: target.shape[0] - 1]
            images, target = mixup_fn(images, target)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # forward
            output = model(images)

            if isinstance(output, torch.Tensor):
                loss = loss_fn(output, target)
                loss_dic = {"loss": loss.item()}
            else:
                losses = {}
                for i, out in enumerate(output):
                    loss_name = "main" if i == 0 else f"aux_{i}"
                    losses[loss_name] = loss_fn(out, target)
                loss = sum(losses.values())
                loss_dic = {"total": loss.item()}
                loss_dic.update({k: v.item() for k, v in losses.items()})

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # backward
        if fp16_scaler is not None:
            fp16_scaler.scale(loss / args.accum_steps).backward()
        else:
            (loss / args.accum_steps).backward()

        # step
        if (data_i + 1) % args.accum_steps == 0:
            if fp16_scaler is not None:
                if args.max_norm is not None:
                    fp16_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            else:
                if args.max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()
            optimizer.zero_grad()

        # log
        torch.cuda.synchronize()
        metric_logger.update(**loss_dic, lr=optimizer.param_groups[-1]["lr"])

        # increase grad_i every accum_steps
        if (data_i + 1) % args.accum_steps == 0:
            grad_i += 1

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, fp16_scaler):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            output = model(inp)

        loss = criterion(output, target)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss,
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
