import timm

from vision_transformer import VisionTransformer
from utils import count_parameters


def get_model(args):

    model_kwargs = {
        "num_classes": args.nb_classes,
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "global_pool": args.global_pool,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "depth": args.depth,
        "mlp_ratio": args.mlp_ratio,
        "drop_path_rate": args.drop_path,
        "class_token": True,
    }

    model = VisionTransformer(args.cropr_cfg, **model_kwargs)

    if args.checkpoint is not None:
        # Load pretrained weights
        print("Loading pretrained weights: ", args.checkpoint)

        model_kwargs["num_classes"] = 0
        model_kwargs.pop("global_pool")

        checkpoint = timm.create_model(
            args.checkpoint,
            pretrained=True,
            **model_kwargs,
        )
        state_dict = {
            k.replace("base_model.", ""): v for k, v in checkpoint.state_dict().items()
        }
        msg = model.load_state_dict(state_dict, strict=False)

        del checkpoint
        # print("Loaded checkpoint with msg: {}".format(msg))

    print("Number of parameters: ", count_parameters(model))

    return model
