import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.eva import Eva, EvaBlock

from cropr import Cropr
from utils.pad import padding, unpadding


class Segmenter(Eva):
    def __init__(self, cropr_cfg, **kwargs):
        super(Segmenter, self).__init__(**kwargs)

        self.patch_size = kwargs["patch_size"]
        self.n_cls = kwargs["num_classes"]

        self.use_cropr = cropr_cfg["use_cropr"]

        dpr = [
            x.item()
            for x in torch.linspace(0, kwargs["drop_path_rate"], len(self.blocks))
        ]
        dpr[-1] = 0.0  # last block does not have drop path
        self.blocks = nn.ModuleList(
            [
                EvaBlock(
                    dim=self.embed_dim,
                    num_heads=kwargs["num_heads"],
                    mlp_ratio=kwargs["mlp_ratio"],
                    qkv_fused=False,
                    swiglu_mlp=True,
                    scale_mlp=True,
                    drop_path=dpr[i],
                )
                for i in range(len(self.blocks))
            ]
        )

        if self.use_cropr:
            n_blk = len(self.blocks)
            num_cropr_modules = n_blk - 2  # due to LLF, enabled for segmentation
            schedule = [cropr_cfg["pruning_rate"]] * num_cropr_modules
            schedule[0] += 1  # to account for CLS token, remove if no CLS token

            self.cropr = nn.ModuleList(
                [
                    Cropr(
                        pruning_rate=schedule[i],
                        num_queries=cropr_cfg["num_queries"],
                        num_classes=kwargs["num_classes"],
                        embed_dim=kwargs["embed_dim"],
                        num_heads=cropr_cfg["num_heads"],
                        pre_attn_norm=cropr_cfg["pre_attn_norm"],
                        q_proj=cropr_cfg["q_proj"],
                        k_proj=cropr_cfg["k_proj"],
                        v_proj=cropr_cfg["v_proj"],
                        mlp=cropr_cfg["mlp"],
                        mlp_ratio=cropr_cfg["mlp_ratio"],
                        training=self.training,
                    )
                    for i in range(num_cropr_modules)
                ]
            )

            print("\nUsing Cropr.")
            print("LLF active.")
            num_tokens = (kwargs["img_size"] // kwargs["patch_size"]) ** 2 + 1
            num_remaining_per_block = [
                num_tokens - sum(schedule[: i + 1]) for i in range(num_cropr_modules)
            ]
            print(f"Using Cropr with schedule: {num_remaining_per_block}\n")

            # store pruning rate curriculum, linearly increase to final rate during first half of training
            pruning_rate_curr = torch.linspace(
                0, cropr_cfg["pruning_rate"], steps=cropr_cfg["epochs"] // 2
            ).long()
            self.pruning_rate_curr = torch.cat(
                (
                    pruning_rate_curr,
                    torch.ones(cropr_cfg["epochs"] // 2).long()
                    * cropr_cfg["pruning_rate"],
                ),
                dim=0,
            )

    def set_pruning_rate(self, epoch):
        # set pruning rate for each Cropr module according to curriculum
        new_pruning_rate = self.pruning_rate_curr[epoch].item()
        print("Pruning rate: ", new_pruning_rate)
        for i in range(len(self.cropr)):
            self.cropr[i].pruning_rate = new_pruning_rate
        self.cropr[0].pruning_rate += 1  # account for CLS token

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, pos = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=pos)
        x = self.norm(x)
        return x

    def forward_cropr(self, x):
        x = self.patch_embed(x)
        x, pos = self._pos_embed(x)
        pos = pos.unsqueeze(0).expand(x.shape[0], -1, -1)

        B, M, D = x.shape

        # Init retained token indices
        idx = torch.arange(M, device=x.device).unsqueeze(0).expand(B, -1)

        # For storing of auxiliary preds, pruned tokens, idx, and pos embeddings
        preds, prnd, idx_prnd, pos_prnd = [], [], [], []
        for i, blk in enumerate(self.blocks[:-2]):
            x = blk(x, rope=pos)

            x, x_p, pos, pos_p, idx, idx_p, pred = self.cropr[i](
                x, pos, idx=idx, inference=not self.training
            )
            preds.append(pred)
            prnd.append(x_p)
            idx_prnd.append(idx_p)
            pos_prnd.append(pos_p)

        # no pruning applied to penultimate block because pruned tokens would be concatenated next anyways
        x = self.blocks[-2](x, rope=pos)

        x = torch.cat([x] + prnd, dim=1)
        pos = torch.cat([pos] + pos_prnd, dim=1)
        # Sort idx to obtain original order
        idx = torch.cat([idx] + idx_prnd, dim=1).argsort(dim=1)

        # LLF
        x = self.blocks[-1](x, rope=pos)
        x = self.norm(x)

        # Reorder
        x = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))

        return x, preds

    def forward_head(self, x, pre_logits: bool = False):
        x = x[:, self.num_prefix_tokens :]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, upsample=True):

        H_orig, W_orig = x.size(2), x.size(3)
        x = padding(x, self.patch_size)
        H, W = x.size(2), x.size(3)

        if not self.use_cropr:
            x = self.forward_features(x)
            preds = None
        else:
            x, preds = self.forward_cropr(x)
        x = self.forward_head(x)

        if not upsample:
            return x

        GS = H // self.patch_size
        masks = rearrange(x, "b (h w) c -> b c h w", h=GS)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_orig, W_orig))

        if preds is None or preds[-1] is None:
            return masks

        preds = torch.stack(preds, dim=0)
        num_preds = preds.size(0)

        aux_masks = rearrange(preds, "n b (h w) c -> n b c h w", h=GS)

        # create tuple of masks
        aux_masks = torch.chunk(aux_masks, num_preds, dim=0)

        return (masks,) + aux_masks
