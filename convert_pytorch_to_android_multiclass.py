import torch
import torch.nn as nn
from models.resnet_imagenet_multiclass_infer import resnet18
import argparse
import os


def get_shift_classifer(model, K_shift):
    model.shift_cls_layer = nn.Linear(model.last_dim, K_shift)

    return model


def main(args):
    device = 'cpu'
    K_shift = 4

    model = resnet18(num_classes=args.num_classes)
    model = get_shift_classifer(model, K_shift).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    example = torch.rand(1, 3, 224, 224)
    ret = model(example)
    print(ret, ret.shape)
    traced_script_module = torch.jit.trace(model, example)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    traced_script_module.save(args.output_path)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        default='/home/irelin/resource/afp/skin_anomaly_detection/pretrained_models/multi_class_skin_pytorch.pth')

    parser.add_argument('--output_path', type=str, default='./multiclass_skin_model.pt')
    parser.add_argument('--num_classes', type=int, default=3)
    main(parser.parse_args())
