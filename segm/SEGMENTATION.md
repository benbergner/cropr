This code is based on [Segmenter](https://github.com/rstrudel/segmenter/blob/master/README.md). Please see their install instructions. Furthermore, we use:
- Python 3.10
- PyTorch 2.3.0
- timm 1.0.3

To fine-tune an EVA-pretrained ViT-L without Cropr (no-pruning baseline) on ADE20k, run:
```
python main.py \
--output_dir ${OUT_DIR} \
--checkpoint "eva02_large_patch14_224.mim_m38m" \
--use_cropr "false" \
--batch_size 8 \
--accum_steps 1 \
```
This should result in an mIoU of x.x ([logs](../logs/segm_unpruned.txt))

To fine-tune an EVA-pretrained ViT-L with Cropr and LLF on ADE20k, run:
```
python main.py \
--output_dir ${OUT_DIR} \
--checkpoint "eva02_large_patch14_224.mim_m38m" \
--use_cropr "true" \
--batch_size 8 \
--accum_steps 1 \
--cropr_pruning_rate 40 \
--cropr_num_queries 1024 \
```
This should result in an mIoU of x.x ([logs](../logs/segm_cropr_llf.txt))