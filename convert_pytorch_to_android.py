import torch
import torch.nn as nn
import transform_layers_jit as TL
from models.resnet_imagenet_infer import resnet18
import pickle
import argparse
import os


def get_shift_classifer(model, K_shift):
    model.shift_cls_layer = nn.Linear(model.last_dim, K_shift)

    return model


def get_simclr_augmentation():
    image_size = 224
    resize_factor = 0.54
    # parameter for resizecrop
    resize_scale = (resize_factor, resize_factor)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=(image_size, image_size, 3))

    transform = nn.Sequential(
        color_jitter,
        color_gray,
        resize_crop,
    )

    return transform


def main(args):
    device = 'cpu'
    K_shift = 4
    axis = pickle.load(open(args.axis_path, "rb"))
    for i in range(4):
        axis[i] = axis[i].to(device)
    hflip = TL.HorizontalFlipLayer().to(device)
    shift_transform = TL.Rotation()
    simclr_aug = get_simclr_augmentation().to(device)

    model = resnet18(num_classes=2)
    model = get_shift_classifer(model, K_shift).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.set_transforms(hflip, shift_transform, simclr_aug, axis)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    example = torch.rand(1, 3, 224, 224)
    ret = model(example)
    print(ret, ret.shape)
    # sys.exit()
    traced_script_module = torch.jit.trace(model, example)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    traced_script_module.save(args.output_path)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--axis_path', type=str)

    parser.add_argument('--output_path', type=int, default=2)
    main(parser.parse_args())
