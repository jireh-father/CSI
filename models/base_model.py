from abc import *
import torch.nn as nn
import torch


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10, simclr_dim=128):
        super(BaseModel, self).__init__()
        self.linear = nn.Linear(last_dim, num_classes)
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )
        self.shift_cls_layer = nn.Linear(last_dim, 2)
        self.joint_distribution_layer = nn.Linear(last_dim, 4 * num_classes)

    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def forward(self, inputs, penultimate=False, simclr=False, shift=False, joint=False):
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)

        output = self.linear(features)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            _aux['simclr'] = self.simclr_layer(features)

        if shift:
            _return_aux = True
            _aux['shift'] = self.shift_cls_layer(features)

        if joint:
            _return_aux = True
            _aux['joint'] = self.joint_distribution_layer(features)

        if _return_aux:
            return output, _aux

        return output

    def forward_(self, inputs):
        print(inputs.shape)
        inputs = torch.cat([self.shift_trans(self.hflip(inputs), k) for k in range(4)])
        print(inputs.shape)
        inputs = self.simclr_aug(inputs)
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)

        _aux['simclr'] = self.simclr_layer(features)

        _aux['shift'] = self.shift_cls_layer(features)

        return _aux['simclr']

    def set_transforms(self, hflip, shift_transform, simclr_aug):
        self.hflip = hflip
        self.shift_trans = shift_transform
        self.simclr_aug = simclr_aug
