This code uses:
- Python 3.10
- PyTorch 2.3.0
- timm 1.0.3

To fine-tune an MAE-pretrained ViT-L without Cropr (no-pruning baseline) on ImageNet-1k, run:
```
python main.py \
--data_path ${DATA_DIR} \
--output_dir ${OUT_DIR} \
--checkpoint "vit_large_patch16_224.mae" \
--use_cropr "false" \
--batch_size 256 \
--accum_steps 4 \
```
This should result in a top-1 val accuracy of 85.8 ([logs](../logs/cls_unpruned.txt))

To fine-tune an MAE-pretrained ViT-L with Cropr and LLF on ImageNet-1k, run:
```
python main.py \
--data_path ${DATA_DIR} \
--output_dir ${OUT_DIR} \
--checkpoint "vit_large_patch16_224.mae" \
--use_cropr "true" \
--batch_size 512 \
--accum_steps 2 \
--cropr_pruning_rate 8 \
--cropr_llf true \
```
This should result in a top-1 val accuracy of 85.3 ([logs](../logs/cls_cropr_llf.txt))