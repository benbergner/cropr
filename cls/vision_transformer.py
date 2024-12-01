from functools import partial
import torch
import torch.nn as nn
import timm

from cropr import Cropr


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(
        self,
        cropr_cfg,
        **kwargs,
    ):
        super(VisionTransformer, self).__init__(**kwargs)

        self.use_cropr = cropr_cfg["use_cropr"]
        self.llf = cropr_cfg["llf"] if self.use_cropr else False

        dpr = [
            x.item()
            for x in torch.linspace(0, kwargs["drop_path_rate"], len(self.blocks))
        ]
        dpr[-1] = 0.0 if cropr_cfg["use_cropr"] else dpr[-1]
        for blk_idx in range(len(self.blocks)):
            self.blocks[blk_idx] = timm.models.vision_transformer.Block(
                dim=self.embed_dim,
                num_heads=kwargs["num_heads"],
                qkv_bias=True,
                drop_path=dpr[blk_idx],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )

        # Create Cropr modules
        if self.use_cropr:
            n_blk = len(self.blocks)
            num_cropr_modules = n_blk - 2 if self.llf else n_blk - 1
            schedule = [cropr_cfg["pruning_rate"]] * num_cropr_modules
            schedule[0] += 1

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
            print(f"LLF active: {self.llf}.")
            num_tokens = (kwargs["img_size"] // kwargs["patch_size"]) ** 2 + 1
            num_remaining_per_block = [
                num_tokens - sum(schedule[: i + 1]) for i in range(num_cropr_modules)
            ]
            print(f"Using Cropr with schedule: {num_remaining_per_block}\n")

        self.init_weights()

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x.mean(dim=1)
        else:
            x = x[:, 0]
        x = self.fc_norm(x)
        return self.head(x)

    def forward_cropr_wo_llf(self, x):
        preds = []
        for i, blk in enumerate(self.blocks[:-1]):
            x = blk(x)

            x, _, pred = self.cropr[i](x, inference=not self.training)
            preds.append(pred)

        x = self.blocks[-1](x)
        x = self.norm(x)

        return x, preds

    def forward_cropr(self, x):
        preds, prnd = [], []
        for i, blk in enumerate(self.blocks[:-2]):
            x = blk(x)

            x, x_p, pred = self.cropr[i](x, inference=not self.training)
            preds.append(pred)
            prnd.append(x_p)

        # no pruning applied to penultimate block because pruned tokens would be concatenated next anyways
        x = self.blocks[-2](x)
        x = torch.cat([x] + prnd, dim=1)
        # LLF
        x = self.blocks[-1](x)
        x = self.norm(x)

        return x, preds

    def forward_features(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_embed(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        return x

    def forward(self, x):

        x = self.forward_embed(x)
        # decide whether to use default ViT or apply Cropr (with or w/o LLF)
        if not self.use_cropr:
            x = self.forward_features(x)
            preds = None
        elif self.llf:
            x, preds = self.forward_cropr(x)
        else:
            x, preds = self.forward_cropr_wo_llf(x)
        x = self.forward_head(x)

        if preds is None or preds[-1] is None:
            return x
        return [x] + preds
