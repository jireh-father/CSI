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

    def forward_(self, inputs, penultimate=False, simclr=False, shift=False, joint=False):
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

    def forward(self, inputs):
        with torch.no_grad():
            inputs = torch.cat([self.shift_trans(self.hflip(inputs), k) for k in range(4)])
            # inputs = torch.cat([self.shift_trans(inputs, k) for k in range(4)])
            inputs = self.simclr_aug(inputs)
            features = self.penultimate(inputs)
            f_sim = self.simclr_layer(features)
            f_shi = self.shift_cls_layer(features)
            # print(f_sim.chunk(4))
            # print(self.axis.shape)
            # print(self.weight_sim.shape)
            # self.score += (f_sim * self.axis).sum(dim=1).max() * self.weight_sim
            # self.score += f_shi[:] * self.weight_shi
            f_sim = f_sim.unsqueeze(dim=1)
            # print((f_sim * self.axis).sum(dim=2).shape)
            # print((f_sim * self.axis).sum(dim=2).max().shape)
            # print((f_sim * self.axis).sum(dim=2).max(), (f_sim * self.axis).sum(dim=2).max(dim=1))

            return torch.cat([(f_sim * self.axis).sum(dim=2).flatten(), f_shi.flatten()])
            # return (f_sim * self.axis).sum(dim=2).max(dim=1)
            # score = ((f_sim * self.axis).sum(dim=2).max(dim=1).values * self.weight_sim).sum()
            # score = f_shi[0][0]
            # score = f_shi[0][0] * self.weight_shi[0]
            # score += f_shi[1][1] * self.weight_shi[1]
            # score += f_shi[2][2] * self.weight_shi[2]
            # score += f_shi[3][3] * self.weight_shi[3]
            # print(f_shi.shape)
            # print(len(f_shi.chunk(4)))
            # print(f_shi.chunk(4)[0].shape)
            # f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim.chunk(4)]  # list of (1, d)
            # f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(4)]  # list of (1, 4)
            # for shi in range(4):
            #     # print(f_sim[shi].is_cuda())
            #     tmp_axis = self.axis[shi]
            #     print('f_shi[shi]', f_shi[shi].shape)
            #     print('f_shi[shi][:]', f_shi[shi][:].shape)
            #     print('f_shi[shi][:, shi]', f_shi[shi][:, shi].shape)
            #     print('f_shi[shi][:, shi][0]', f_shi[shi][:, shi][0])
            #     sys.exit()
            #     self.score += (f_sim[shi] * tmp_axis).sum(dim=1).max() * self.weight_sim[shi]
            #     self.score += f_shi[shi][:, shi][0] * self.weight_shi[shi]
            # return score / 4

    def set_transforms(self, hflip, shift_transform, simclr_aug, axis):
        self.hflip = hflip
        self.shift_trans = shift_transform
        self.simclr_aug = simclr_aug
        self.axis = torch.cat([torch.unsqueeze(axis[0], 0),torch.unsqueeze(axis[1], 0),torch.unsqueeze(axis[2], 0),torch.unsqueeze(axis[3], 0)], axis=0)
