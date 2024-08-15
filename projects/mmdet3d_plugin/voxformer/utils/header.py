

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from projects.mmdet3d_plugin.voxformer.modules.up_downsample import (Upsample,Downsample)
from projects.mmdet3d_plugin.voxformer.utils.atten3D import Attention3d

class Header(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        feature,
    ):
        super(Header, self).__init__()
        self.feature = feature
        self.class_num = class_num
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )

        self.atten_1_4 = Attention3d(512)
        self.atten_1_2 = Attention3d(256)


        self.downsample_1_2 = Downsample(feature=128, norm_layer=torch.nn.modules.batchnorm.BatchNorm3d, bn_momentum=0.1)

        self.downsample_1_4 = Downsample(feature=256, norm_layer=torch.nn.modules.batchnorm.BatchNorm3d, bn_momentum=0.1)

        self.upsample_1_1 = Upsample(in_channels=128, out_channels=128,
                        norm_layer=torch.nn.modules.batchnorm.BatchNorm3d, bn_momentum=0.1)

        self.upsample_d_1_2 = Upsample(in_channels=256, out_channels=128,
                        norm_layer=torch.nn.modules.batchnorm.BatchNorm3d, bn_momentum=0.1)

        self.upsample_d_1_4 = Upsample(in_channels=512, out_channels=256,
                                       norm_layer=torch.nn.modules.batchnorm.BatchNorm3d, bn_momentum=0.1)


        #self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict["x3d"] # [1, 128, 128, 128, 16]

        #x3d_up_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]

        x3d_1_2_d = self.downsample_1_2(x3d_l1)

        x3d_1_4_d = self.downsample_1_4(x3d_1_2_d)

        x3d_1_4_d =self.atten_1_4(x3d_1_4_d)

        x3d_1_2 = self.upsample_d_1_4(x3d_1_4_d)+x3d_1_2_d

        x3d_1_2 = self.atten_1_2(x3d_1_2)

        x3d_1_1 = self.upsample_d_1_2(x3d_1_2)+x3d_l1

        x3d_1_u = self.upsample_1_1(x3d_1_1)

        x3d_up_l1 = x3d_1_u

        _, feat_dim, w, l, h  = x3d_up_l1.shape

        x3d_up_l1 = x3d_up_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)

        ssc_logit_full = self.mlp_head(x3d_up_l1)

        res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)

        return res
