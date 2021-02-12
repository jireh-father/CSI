import models.classifier as C
from common.common import parse_args
import torch
import torch.nn as nn
import argparse
from torchvision import transforms
import models.transform_layers as TL
from PIL import Image
import glob
import os
import pickle


def _get_features(P, model, img, simclr_aug=None, layers=('simclr', 'shift'), hflip=None, device=None):
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list

    x = torch.unsqueeze(img, 0)
    x = x.to(device)  # gpu tensor

    # compute features in one batch
    feats_batch = {layer: [] for layer in layers}  # initialize: empty list

    if P.K_shift > 1:
        x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
    else:
        x_t = x  # No shifting: SimCLR
    x_t = simclr_aug(x_t)

    # compute augmented features
    with torch.no_grad():
        kwargs = {layer: True for layer in layers}  # only forward selected layers
        _, output_aux = model(x_t, **kwargs)
    # add features in one batch
    for layer in layers:
        feats = output_aux[layer].cpu()
        feats_batch[layer] += [feats]  # (B, d) cpu tensor

    # concatenate features in one batch
    for key, val in feats_batch.items():
        feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)

    # add features in full dataset
    for layer in layers:
        feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    return feats_all


def get_features(P, model, img, simclr_aug=None, layers=('simclr', 'shift'), hflip=None, device=None):
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    return _get_features(P, model, img, simclr_aug, layers=layers, hflip=hflip, device=device)


def get_simclr_augmentation():
    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)

    # Transform define #
    transform = nn.Sequential(
        color_jitter,
        color_gray,
    )

    return transform


def get_scores(P, feats_dict, device):
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(device)
    feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)

    # compute scores
    scores = []
    for f_sim, f_shi in zip(feats_sim, feats_shi):
        f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim.chunk(P.K_shift)]  # list of (1, d)
        f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(P.K_shift)]  # list of (1, 4)
        score = 0
        for shi in range(P.K_shift):
            score += (f_sim[shi] * P.axis[shi]).sum(dim=1).max().item() * P.weight_sim[shi]
            score += f_shi[shi][:, shi].item() * P.weight_shi[shi]
        score = score / P.K_shift
        scores.append(score)
    scores = torch.tensor(scores)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def main(args):
    P = parse_args()

    P.load_path = ''
    P.no_strict = False

    P.shift_trans_type = 'rotation'
    P.mode = 'ood_pre'
    P.n_classes = 2
    P.model = 'resnet18_imagenet'
    P.image_size = (224, 224, 3)

    P.resize_factor = 0.54
    P.resize_fix = True
    P.layers = ['simclr', 'shift']

    device = torch.device(f"cuda" if args.use_cuda else "cpu")

    P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)

    P.axis = pickle.load(open(args.axis_path, "rb"))
    P.weight_sim = pickle.load(open(args.weight_sim_path, "rb"))
    P.weight_shi = pickle.load(open(args.weight_shi_path, "rb"))

    hflip = TL.HorizontalFlipLayer().to(device)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    simclr_aug = get_simclr_augmentation().to(device)

    model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
    model = C.get_shift_classifer(model, P.K_shift).to(device)

    checkpoint = torch.load(P.load_path)
    model.load_state_dict(checkpoint, strict=not P.no_strict)

    kwargs = {
        'simclr_aug': simclr_aug,
        'layers': P.layers,
        'hflip': hflip,
        'device': device
    }

    image_files = glob.glob(os.path.join(args.image_dir, "*"))
    total_scores = []
    for image_file in image_files:
        img = Image.open(image_file).convert("RGB")
        img = test_transform(img)
        features = get_features(P, model, img, **kwargs)
        scores = get_scores(P, features, device).numpy()
        total_scores += list(scores)
    accuracy = (total_scores < args.score_thres).sum() / len(total_scores)
    print("accuracy", accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--axis_path', type=str, default=None)
    parser.add_argument('--weight_sim_path', type=str, default=None)
    parser.add_argument('--weight_shi_path', type=str, default=None)
    parser.add_argument('--score_thres', type=float, default=0.5)
    main(parser.parse_args())
