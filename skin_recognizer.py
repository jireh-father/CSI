import torch
import torch.nn as nn
import argparse
from torchvision import transforms
import transform_layers as TL
from PIL import Image
import glob
import os
import pickle
import time
import cv2
import torch.nn.functional as F


class SkinRecognizer(object):
    def __init__(self, model_path, axis_path=None, score_thres=0.86, image_size=224, model='resnet18_imagenet',
                 shift_trans_type='rotation',
                 resize_factor=0.54, resize_fix=True, layers=['simclr', 'shift'], use_cuda=False,
                 weight_sim=[0.007519226599080519, 0.007939391391667395, 0.008598049328054363, 0.015014530319964874],
                 weight_shi=[0.04909334419285857, 0.052858438675397496, 0.05840793893796496, 0.11790745570891596],
                 n_classes=2, is_multi_class=False):
        class DumpClass(object):
            pass

        P = DumpClass()
        device = torch.device(f"cuda" if use_cuda else "cpu")
        self.device = device
        P.n_classes = n_classes
        P.model = model
        P.image_size = (image_size, image_size, 3)

        if not is_multi_class:
            P.shift_trans_type = shift_trans_type
            P.resize_factor = resize_factor
            P.resize_fix = resize_fix
            P.layers = layers
            P.shift_trans, P.K_shift = self.get_shift_module(shift_trans_type)
            P.axis = pickle.load(open(axis_path, "rb"))
            P.weight_sim = weight_sim
            P.weight_shi = weight_shi
            self.hflip = TL.HorizontalFlipLayer().to(device)
            self.params = P
            self.layers = P.layers
            self.simclr_aug = self.get_simclr_augmentation(P.image_size).to(device)

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(224),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

        model = self.get_classifier().to(device)
        model = self.get_shift_classifer(model).to(device)

        if use_cuda:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        self.model = model

        self.score_thres = score_thres

    def get_shift_module(self, shift_trans_type):
        if shift_trans_type == 'rotation':
            shift_transform = TL.Rotation()
            K_shift = 4
        elif shift_trans_type == 'cutperm':
            shift_transform = TL.CutPerm()
            K_shift = 4
        else:
            shift_transform = nn.Identity()
            K_shift = 1

        return shift_transform, K_shift

    def get_shift_classifer(self, model):

        model.shift_cls_layer = nn.Linear(model.last_dim, self.params.K_shift)

        return model

    def get_classifier(self):
        if self.params.model == 'resnet18':
            from models.resnet import ResNet18
            classifier = ResNet18(num_classes=self.params.n_classes)
        elif self.params.model == 'resnet34':
            from models.resnet import ResNet34
            classifier = ResNet34(num_classes=self.params.n_classes)
        elif self.params.model == 'resnet50':
            from models.resnet import ResNet50
            classifier = ResNet50(num_classes=self.params.n_classes)
        elif self.params.model == 'resnet18_imagenet':
            from models.resnet_imagenet import resnet18
            classifier = resnet18(num_classes=self.params.n_classes)
        elif self.params.model == 'resnet50_imagenet':
            from models.resnet_imagenet import resnet50
            classifier = resnet50(num_classes=self.params.n_classes)
        else:
            raise NotImplementedError()

        return classifier

    def _get_features(self, img):
        # check if arguments are valid
        assert self.simclr_aug is not None

        # compute features in full dataset
        self.model.eval()

        x = torch.unsqueeze(img, 0)
        x = x.to(self.device)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in self.layers}  # initialize: empty list
        if self.params.K_shift > 1:
            # x_t = torch.cat([self.params.shift_trans(self.hflip(x), k) for k in range(self.params.K_shift)])
            x_t = torch.cat([self.params.shift_trans(x, k) for k in range(self.params.K_shift)])
        else:
            x_t = x  # No shifting: SimCLR
        # x_t = self.simclr_aug(x_t)

        # compute augmented features
        with torch.no_grad():
            kwargs = {layer: True for layer in self.layers}  # only forward selected layers
            _, output_aux = self.model(x_t, **kwargs)
        # add features in one batch
        for layer in self.layers:
            feats = output_aux[layer].cpu()
            feats = torch.unsqueeze(feats, 0)
            feats_batch[layer] = feats
            # feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        # for key, val in feats_batch.items():
        #     feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
        #     print(key, feats_batch[key].shape)
        # add features in full dataset
        return feats_batch

    def get_features(self, img):
        return self._get_features(img)

    def get_scores(self, feats_dict):
        # convert to gpu tensor
        feats_sim = feats_dict['simclr'].to(self.device)
        feats_shi = feats_dict['shift'].to(self.device)
        N = feats_sim.size(0)
        # compute scores
        scores = []
        for f_sim, f_shi in zip(feats_sim, feats_shi):
            f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim.chunk(self.params.K_shift)]  # list of (1, d)
            f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(self.params.K_shift)]  # list of (1, 4)
            score = 0
            for shi in range(self.params.K_shift):
                # print(f_sim[shi].is_cuda())
                tmp_axis = self.params.axis[shi].to(self.device)
                score += (f_sim[shi] * tmp_axis).sum(dim=1).max().item() * self.params.weight_sim[shi]
                score += f_shi[shi][:, shi].item() * self.params.weight_shi[shi]
            score = score / self.params.K_shift
            scores.append(score)
        scores = torch.tensor(scores)

        assert scores.dim() == 1 and scores.size(0) == N  # (N)
        return scores.cpu()

    def get_simclr_augmentation(self, image_size):
        # parameter for resizecrop
        resize_scale = (self.params.resize_factor, 1.0)  # resize scaling factor
        if self.params.resize_fix:  # if resize_fix is True, use same scale
            resize_scale = (self.params.resize_factor, self.params.resize_factor)

        # Align augmentation
        color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
        color_gray = TL.RandomColorGrayLayer(p=0.2)
        resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

        transform = nn.Sequential(
            color_jitter,
            color_gray,
            resize_crop,
        )

        return transform

    def is_skin(self, img):
        img = Image.fromarray(img)
        img = self.test_transform(img)
        features = self.get_features(img)
        scores = self.get_scores(features).numpy()
        print(scores)
        return scores[0] >= self.score_thres

    def is_skin_and_what_class(self, img, num_rotation=2, classes=['armpit_belly', 'ear', 'foot']):
        if num_rotation > 4:
            num_rotation = 4
        if num_rotation < 1:
            num_rotation = 1

        n_classes = len(classes)

        img = self.test_transform(img)
        outputs = 0
        for i in range(num_rotation):
            rot_images = torch.rot90(img, i, (2, 3))
            _, outputs_aux = self.model(rot_images, joint=True)
            outputs += outputs_aux['joint'][:, n_classes * i: n_classes * (i + 1)] / float(num_rotation)

        _, preds = torch.max(outputs, 1)
        result_class = classes[preds.cpu().numpy()[0]]

        scores = F.softmax(outputs, dim=1).max(dim=1)[0]
        score = scores.detach().cpu().numpy()[0]
        return result_class, score >= self.score_thres


def main(P):
    if P.w_sim_path:
        weight_sim = pickle.load(open(P.w_sim_path, "rb"))
        weight_shi = pickle.load(open(P.w_shi_path, "rb"))
        sr = SkinRecognizer(P.load_path, P.axis_path, use_cuda=P.use_cuda, score_thres=P.score_thres,
                            weight_sim=weight_sim, weight_shi=weight_shi)
    else:
        sr = SkinRecognizer(P.load_path, P.axis_path, use_cuda=P.use_cuda, score_thres=P.score_thres,
                            is_multi_class=P.is_multi_class)

    image_files = glob.glob(os.path.join(P.image_dir, "*"))
    is_skins = 0
    classes = []
    for i, image_file in enumerate(image_files):
        start = time.time()
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if P.is_multi_class:
            cls, is_skin = sr.is_skin_and_what_class(img)
            is_skins += is_skin
            classes.append(cls)
            print(is_skin, classes)
        else:
            is_skins += sr.is_skin(img)
            print(is_skins)

    if P.is_positive:
        print('true accuracy thres', P.score_thres, is_skins / len(image_files))
    else:
        print('false accuracy thres', P.score_thres, 1. - (is_skins / len(image_files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str,
                        default='/home/irelin/resource/afp/skin_anomaly_detection/resnet18_224_last.model')
    parser.add_argument('--image_dir', type=str,
                        default='/home/irelin/resource/afp/skin_anomaly_detection/real_test_images')
    parser.add_argument('--axis_path', type=str, default='/home/irelin/resource/afp/skin_anomaly_detection/axis.pth')

    parser.add_argument('--w_sim_path', type=str, default=None)
    parser.add_argument('--w_shi_path', type=str, default=None)
    parser.add_argument('--score_thres', type=float, default=0.4)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--is_positive', action='store_true', default=False)
    parser.add_argument('--is_multi_class', action='store_true', default=False)

    main(parser.parse_args())
