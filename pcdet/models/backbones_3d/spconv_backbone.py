from functools import partial

import torch
import torch.nn as nn
import numpy as np

import time
import os

from ...utils.spconv_utils import replace_feature, spconv
from tools.visual_utils.vis_feature_maps import entropyOfFmapsSparse


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None, act_fn=nn.ReLU):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        act_fn(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None, act_fn=nn.ReLU):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = act_fn()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


# this class is copied partly from SparseKD
class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        if model_cfg.get('NUM_FILTERS', None):
            num_filters = model_cfg.NUM_FILTERS
        else:
            num_filters = [16, 16, 32, 64, 64, 128]

        #if model_cfg.get('WIDTH', None):
            #num_filters = (np.array(num_filters, dtype=np.int32) * model_cfg.WIDTH).astype(int)
        num_filters = (np.array(num_filters, dtype=np.int32) * 0.75).astype(int)

        if model_cfg.get('LAYER_NUMS', None):
            layer_nums = model_cfg.LAYER_NUMS
        else:
            layer_nums = [1, 1, 3, 3, 3, 1]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )
        block = post_act_block

        conv1_list = [block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, indice_key='subm1')]
        for k in range(layer_nums[1] - 1):
            conv1_list.append(block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, indice_key='subm1'))
        self.conv1 = spconv.SparseSequential(*conv1_list)

        conv2_list = [
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        ]
        for k in range(layer_nums[2] - 1):
            conv2_list.append(block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, indice_key='subm2'))
        self.conv2 = spconv.SparseSequential(*conv2_list)

        # conv3
        conv3_list = [
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
        ]
        for k in range(layer_nums[3] - 1):
            conv3_list.append(block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, indice_key='subm3'))
        self.conv3 = spconv.SparseSequential(*conv3_list)

        # conv4
        conv4_list = [
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[3], num_filters[4], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        ]
        for k in range(layer_nums[4] - 1):
            conv4_list.append(block(num_filters[4], num_filters[4], 3, norm_fn=norm_fn, indice_key='subm4'))
        self.conv4 = spconv.SparseSequential(*conv4_list)

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[4], num_filters[5], (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(num_filters[5]),
            nn.ReLU(),
        )
        self.num_point_features = num_filters[5]
        self.backbone_channels = {
            'x_conv1': num_filters[1],
            'x_conv2': num_filters[2],
            'x_conv3': num_filters[3],
            'x_conv4': num_filters[4]
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelBackBone8x_old(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # currently not used
        # act_fn = get_act_layer(self.model_cfg.get('ACT_FN', 'ReLU'))
        act_fn = nn.ReLU

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        if model_cfg.get('NUM_FILTERS', None):
            num_filters = model_cfg.NUM_FILTERS
        else:
            num_filters = [16, 16, 32, 64, 128, 128]

        if model_cfg.get('WIDTH', None):
            num_filters = (np.array(num_filters, dtype=np.int32) * model_cfg.WIDTH).astype(int)

        if model_cfg.get('LAYER_NUMS', None):
            layer_nums = model_cfg.LAYER_NUMS
        else:
            layer_nums = [1, 2, 3, 3, 3, 1]

        if model_cfg.get('FEAT_ADAPT_SINGLE', False):
            use_feat_adapt_single = model_cfg.FEAT_ADAPT_SINGLE
        else:
            use_feat_adapt_single = False
            
        if model_cfg.get('FEAT_ADAPT_AUTOENCODER', False):
            use_feat_adapt_autoencoder = model_cfg.FEAT_ADAPT_AUTOENCODER
        else:
            use_feat_adapt_autoencoder = False

        if model_cfg.get('FULL_AUTOENCODER', False):
            use_full_autoencoder = model_cfg.FULL_AUTOENCODER
        else:
            use_full_autoencoder = False

        if model_cfg.get('TOP_PERCENTAGE', False):
            self.top_percentage = model_cfg.TOP_PERCENTAGE

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            act_fn(),
        )
        block = partial(post_act_block, act_fn=act_fn)

        # conv1
        conv1_list = [SparseBasicBlock(num_filters[0], num_filters[1], norm_fn=norm_fn, act_fn=act_fn, indice_key='res1')]
        for k in range(layer_nums[1] - 1):
            conv1_list.append(SparseBasicBlock(num_filters[1], num_filters[1], norm_fn=norm_fn, act_fn=act_fn, indice_key='res1'))
        self.conv1 = spconv.SparseSequential(*conv1_list)

        # conv2
        conv2_list = [
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        ]
        for k in range(layer_nums[2] - 1):
            conv2_list.append(SparseBasicBlock(num_filters[2], num_filters[2], norm_fn=norm_fn, act_fn=act_fn, indice_key='res2'))
        self.conv2 = spconv.SparseSequential(*conv2_list)

        # conv3
        conv3_list = [
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        ]
        for k in range(layer_nums[3] - 1):
            conv3_list.append(SparseBasicBlock(
                num_filters[3], num_filters[3], norm_fn=norm_fn, act_fn=act_fn, indice_key='res3'
            ))
        self.conv3 = spconv.SparseSequential(*conv3_list)

        # conv4
        conv4_list = [
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[3], num_filters[4], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        ]
        for k in range(layer_nums[4] - 1):
            conv4_list.append(SparseBasicBlock(
                num_filters[4], num_filters[4], norm_fn=norm_fn, act_fn=act_fn, indice_key='res4'
            ))
        self.conv4 = spconv.SparseSequential(*conv4_list)

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[4], num_filters[5], (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(num_filters[5]))

        if use_feat_adapt_single is True:
            # the feature adaptation layer is used to adapt the feature dimension of the student to the teacher
            self.feat_adapt_single = spconv.SparseSequential(
                spconv.SparseConv3d(num_filters[5], 128, 1, stride=1, padding=0),
                nn.ReLU())
            self.feat_adapt_single[0].bias.requires_grad = True
            self.feat_adapt_single[0].weight.requires_grad = True
            nn.init.zeros_(self.feat_adapt_single[0].bias.data)
            nn.init.kaiming_normal_(self.feat_adapt_single[0].weight.data)
            # self.feat_adapt_single[0].bias.data.fill_(0.)
            # self.feat_adapt_single[0].weight.data.fill_(1.)
            
        if use_feat_adapt_autoencoder is True:
            # currently only used for teacher to student adaptation
            self.feat_adapt_autoencoder = spconv.SparseSequential(
                spconv.SparseConv3d(num_filters[5], 112, 1, stride=1, padding=0),
                nn.ReLU(),
                spconv.SparseConv3d(112, 96, 1, stride=1, padding=0),
                nn.ReLU())

        if use_full_autoencoder is True:
            # currently used for teacher only training
            self.full_autoencoder = spconv.SparseSequential(
                spconv.SparseConv3d(num_filters[5], 112, 1, stride=1, padding=0),
                nn.ReLU(),
                spconv.SparseConv3d(112, 96, 1, stride=1, padding=0),
                nn.ReLU(),
                spconv.SparseConv3d(96, 112, 1, stride=1, padding=0),
                nn.ReLU(),
                spconv.SparseConv3d(112, num_filters[5], 1, stride=1, padding=0),
                nn.ReLU())

        self.final_act = act_fn()

        self.num_point_features = num_filters[5]
        self.backbone_channels = {
            'x_conv1': num_filters[1],
            'x_conv2': num_filters[2],
            'x_conv3': num_filters[3],
            'x_conv4': num_filters[4]
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # if getattr(self, 'top_percentage', None) or self.top_percentage < 1.0:
        #     topn_indices = torch.topk(entropyOfFmapsSparse(x_conv4.features),
        #                               int(x_conv4.features.shape[0] * self.top_percentage),
        #                               dim=0,
        #                               largest=True,
        #                               sorted=False).indices
        #     x_topn = spconv.SparseConvTensor(
        #         features=x_conv4.features[topn_indices],
        #         indices=x_conv4.indices[topn_indices],
        #         spatial_shape=x_conv4.spatial_shape,
        #         batch_size=x_conv4.batch_size
        #     )
        #     out = self.conv_out(x_topn)
        # else:
        #     # for detection head
        #     # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        # if student, insert feature adaptation layer
        if getattr(self, 'feat_adapt_single', None):
            self.feat_adapt_single(self.conv_out[0](x_conv4))
        if getattr(self, 'feat_adapt_autoencoder', None):
            self.feat_adapt_autoencoder(self.conv_out[0](x_conv4))
        if getattr(self, 'full_autoencoder', None):
            self.full_autoencoder(self.conv_out[0](x_conv4))

        # TODO: commented because clone_sp_tensor need modification of spconv_utils.py
        # if getattr(self, 'is_teacher', None):
        #     pre_act_encoded_spconv_tensor = clone_sp_tensor(out, batch_size)
        # out = replace_feature(out, self.final_act(out.features))

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
        })

        # comment to save memory
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        # TODO: Teacher stuff not used yet
        # if getattr(self, 'is_teacher', None):
        #     batch_dict.update({
        #         'encoded_spconv_tensor_pre-act': pre_act_encoded_spconv_tensor
        #     })

        return batch_dict


# Note: When classes are added to backbone_3d.py they must be added to pcdet/models/backbones_3d/__init__.py aswell
class VoxelResBackBone8xEntropy1(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # currently not used
        # act_fn = get_act_layer(self.model_cfg.get('ACT_FN', 'ReLU'))
        act_fn = nn.ReLU

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        if model_cfg.get('NUM_FILTERS', None):
            num_filters = model_cfg.NUM_FILTERS
        else:
            num_filters = [16, 16, 32, 64, 128, 128]

        if model_cfg.get('WIDTH', None):
            num_filters = (np.array(num_filters, dtype=np.int32) * model_cfg.WIDTH).astype(int)

        if model_cfg.get('LAYER_NUMS', None):
            layer_nums = model_cfg.LAYER_NUMS
        else:
            layer_nums = [1, 2, 3, 3, 3, 1]

        if model_cfg.get('FEAT_ADAPT_SINGLE', False):
            use_feat_adapt_single = model_cfg.FEAT_ADAPT_SINGLE
        else:
            use_feat_adapt_single = False

        if model_cfg.get('FEAT_ADAPT_AUTOENCODER', False):
            use_feat_adapt_autoencoder = model_cfg.FEAT_ADAPT_AUTOENCODER
        else:
            use_feat_adapt_autoencoder = False

        if model_cfg.get('FULL_AUTOENCODER', False):
            use_full_autoencoder = model_cfg.FULL_AUTOENCODER
        else:
            use_full_autoencoder = False

        if model_cfg.get('TOP_PERCENTAGE', False):
            self.top_percentage = model_cfg.TOP_PERCENTAGE

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            act_fn(),
        )
        block = partial(post_act_block, act_fn=act_fn)

        # conv1
        conv1_list = [
            SparseBasicBlock(num_filters[0], num_filters[1], norm_fn=norm_fn, act_fn=act_fn, indice_key='res1')]
        for k in range(layer_nums[1] - 1):
            conv1_list.append(
                SparseBasicBlock(num_filters[1], num_filters[1], norm_fn=norm_fn, act_fn=act_fn, indice_key='res1'))
        self.conv1 = spconv.SparseSequential(*conv1_list)

        # conv2
        conv2_list = [
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
        ]
        for k in range(layer_nums[2] - 1):
            conv2_list.append(
                SparseBasicBlock(num_filters[2], num_filters[2], norm_fn=norm_fn, act_fn=act_fn, indice_key='res2'))
        self.conv2 = spconv.SparseSequential(*conv2_list)

        # conv3
        conv3_list = [
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
        ]
        for k in range(layer_nums[3] - 1):
            conv3_list.append(SparseBasicBlock(
                num_filters[3], num_filters[3], norm_fn=norm_fn, act_fn=act_fn, indice_key='res3'
            ))
        self.conv3 = spconv.SparseSequential(*conv3_list)

        # conv4
        conv4_list = [
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[3], num_filters[4], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
                  conv_type='spconv'),
        ]
        for k in range(layer_nums[4] - 1):
            conv4_list.append(SparseBasicBlock(
                num_filters[4], num_filters[4], norm_fn=norm_fn, act_fn=act_fn, indice_key='res4'
            ))
        self.conv4 = spconv.SparseSequential(*conv4_list)

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[4], num_filters[5], (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(num_filters[5]))

        if use_feat_adapt_single is True:
            # the feature adaptation layer is used to adapt the feature dimension of the student to the teacher
            self.feat_adapt_single = spconv.SparseSequential(
                spconv.SparseConv3d(num_filters[5], 128, 1, stride=1, padding=0),
                nn.ReLU())
            self.feat_adapt_single[0].bias.requires_grad = True
            self.feat_adapt_single[0].weight.requires_grad = True
            nn.init.zeros_(self.feat_adapt_single[0].bias.data)
            nn.init.kaiming_normal_(self.feat_adapt_single[0].weight.data)
            # self.feat_adapt_single[0].bias.data.fill_(0.)
            # self.feat_adapt_single[0].weight.data.fill_(1.)

        if use_feat_adapt_autoencoder is True:
            # currently only used for teacher to student adaptation
            self.feat_adapt_autoencoder = spconv.SparseSequential(
                spconv.SparseConv3d(num_filters[5], 112, 1, stride=1, padding=0),
                nn.ReLU(),
                spconv.SparseConv3d(112, 96, 1, stride=1, padding=0),
                nn.ReLU())

        if use_full_autoencoder is True:
            # currently used for teacher only training
            self.full_autoencoder = spconv.SparseSequential(
                spconv.SparseConv3d(num_filters[5], 112, 1, stride=1, padding=0),
                nn.ReLU(),
                spconv.SparseConv3d(112, 96, 1, stride=1, padding=0),
                nn.ReLU(),
                spconv.SparseConv3d(96, 112, 1, stride=1, padding=0),
                nn.ReLU(),
                spconv.SparseConv3d(112, num_filters[5], 1, stride=1, padding=0),
                nn.ReLU())

        self.final_act = act_fn()

        self.num_point_features = num_filters[5]
        self.backbone_channels = {
            'x_conv1': num_filters[1],
            'x_conv2': num_filters[2],
            'x_conv3': num_filters[3],
            'x_conv4': num_filters[4]
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        """
        :: WITHOUT ENTROPY FOR TIME CALCS
        """
        start_time_wo_ent = time.time()
        input_sp_tensor_wo_ent = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_wo_ent = self.conv_input(input_sp_tensor_wo_ent)

        x_conv1_wo_ent = self.conv1(x_wo_ent)
        x_conv2_wo_ent = self.conv2(x_conv1_wo_ent)
        x_conv3_wo_ent = self.conv3(x_conv2_wo_ent)
        x_conv4_wo_ent = self.conv4(x_conv3_wo_ent)
        out_wo_ent = self.conv_out(x_conv4_wo_ent)

        # Observation Memory Consumption:
        mem_time = time.time()
        memory_con_wo_ent = (
            input_sp_tensor_wo_ent.features.element_size() * input_sp_tensor_wo_ent.features.nelement() +
            input_sp_tensor_wo_ent.indices.element_size() * input_sp_tensor_wo_ent.indices.nelement() +
            x_wo_ent.features.element_size() * x_wo_ent.features.nelement() +
            x_wo_ent.indices.element_size() * x_wo_ent.indices.nelement() +
            x_conv1_wo_ent.features.element_size() * x_conv1_wo_ent.features.nelement() +
            x_conv1_wo_ent.indices.element_size() * x_conv1_wo_ent.indices.nelement() +
            x_conv2_wo_ent.features.element_size() * x_conv2_wo_ent.features.nelement() +
            x_conv2_wo_ent.indices.element_size() * x_conv2_wo_ent.indices.nelement() +
            x_conv3_wo_ent.features.element_size() * x_conv3_wo_ent.features.nelement() +
            x_conv3_wo_ent.indices.element_size() * x_conv3_wo_ent.indices.nelement() +
            x_conv4_wo_ent.features.element_size() * x_conv4_wo_ent.features.nelement() +
            x_conv4_wo_ent.indices.element_size() * x_conv4_wo_ent.indices.nelement()) * 1e-6
        end_mem_time = time.time() - mem_time

        end_time_wo_ent = time.time() - start_time_wo_ent - end_mem_time

        """
        :: WITH ENTROPY FOR TIME CALCS
        """
        start_time_entropy = time.time()
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)

        topn_indices = torch.topk(entropyOfFmapsSparse(x_conv1.features),
                                  int(x_conv1.features.shape[0] * self.top_percentage)).indices

        x_topn = spconv.SparseConvTensor(
            features=x_conv1.features[topn_indices],
            indices=x_conv1.indices[topn_indices],
            spatial_shape=x_conv1.spatial_shape,
            batch_size=x_conv1.batch_size
        )

        x_conv2 = self.conv2(x_topn)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)

        # Observation Memory Consumption:
        mem_time = time.time()
        memory_con = (
             input_sp_tensor.features.element_size() * input_sp_tensor.features.nelement() +
             input_sp_tensor.indices.element_size() * input_sp_tensor.indices.nelement() +
             x.features.element_size() * x.features.nelement() +
             x.indices.element_size() * x.indices.nelement() +
             x_conv1.features.element_size() * x_conv1.features.nelement() +
             x_conv1.indices.element_size() * x_conv1.indices.nelement() +
             x_conv2.features.element_size() * x_conv2.features.nelement() +
             x_conv2.indices.element_size() * x_conv2.indices.nelement() +
             x_conv3.features.element_size() * x_conv3.features.nelement() +
             x_conv3.indices.element_size() * x_conv3.indices.nelement() +
             x_conv4.features.element_size() * x_conv4.features.nelement() +
             x_conv4.indices.element_size() * x_conv4.indices.nelement()) * 1e-6

        batch_dict.update({
            'multi_scale_3d_features': {
                'input_': input_sp_tensor,
                'x_input': x,
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }})
        end_mem_time = time.time() - mem_time

        end_time_entropy = time.time() - start_time_entropy - end_mem_time

        # if student, insert feature adaptation layer
        if getattr(self, 'feat_adapt_single', None):
            self.feat_adapt_single(self.conv_out[0](x_conv4_wo_ent))
        if getattr(self, 'feat_adapt_autoencoder', None):
            self.feat_adapt_autoencoder(self.conv_out[0](x_conv4_wo_ent))
        if getattr(self, 'full_autoencoder', None):
            self.full_autoencoder(self.conv_out[0](x_conv4_wo_ent))

        # TODO: commented because clone_sp_tensor need modification of spconv_utils.py
        # if getattr(self, 'is_teacher', None):
        #     pre_act_encoded_spconv_tensor = clone_sp_tensor(out, batch_size)
        # out = replace_feature(out, self.final_act(out.features))

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
        })

        # comment to save memory
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        # Time & Memory Observation:
        batch_dict.update({
            'time_wo_ent': end_time_wo_ent,
            'time_entropy': end_time_entropy,
            'memory_con_wo_ent': memory_con_wo_ent,
            'memory_con': memory_con,
        })

        # TODO: Teacher stuff not used yet
        # if getattr(self, 'is_teacher', None):
        #     batch_dict.update({
        #         'encoded_spconv_tensor_pre-act': pre_act_encoded_spconv_tensor
        #     })

        return batch_dict


# Note: When classes are added to backbone_3d.py they must be added to pcdet/models/backbones_3d/__init__.py aswell
class VoxelResBackBone8xEntropy2(VoxelResBackBone8xEntropy1):
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        """
        :: WITHOUT ENTROPY FOR TIME CALCS
        """
        start_time_wo_ent = time.time()
        input_sp_tensor_wo_ent = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_wo_ent = self.conv_input(input_sp_tensor_wo_ent)

        x_conv1_wo_ent = self.conv1(x_wo_ent)
        x_conv2_wo_ent = self.conv2(x_conv1_wo_ent)
        x_conv3_wo_ent = self.conv3(x_conv2_wo_ent)
        x_conv4_wo_ent = self.conv4(x_conv3_wo_ent)
        out_wo_ent = self.conv_out(x_conv4_wo_ent)

        # Observation Memory Consumption:
        mem_time = time.time()
        memory_con_wo_ent = (
                                    input_sp_tensor_wo_ent.features.element_size() * input_sp_tensor_wo_ent.features.nelement() +
                                    input_sp_tensor_wo_ent.indices.element_size() * input_sp_tensor_wo_ent.indices.nelement() +
                                    x_wo_ent.features.element_size() * x_wo_ent.features.nelement() +
                                    x_wo_ent.indices.element_size() * x_wo_ent.indices.nelement() +
                                    x_conv1_wo_ent.features.element_size() * x_conv1_wo_ent.features.nelement() +
                                    x_conv1_wo_ent.indices.element_size() * x_conv1_wo_ent.indices.nelement() +
                                    x_conv2_wo_ent.features.element_size() * x_conv2_wo_ent.features.nelement() +
                                    x_conv2_wo_ent.indices.element_size() * x_conv2_wo_ent.indices.nelement() +
                                    x_conv3_wo_ent.features.element_size() * x_conv3_wo_ent.features.nelement() +
                                    x_conv3_wo_ent.indices.element_size() * x_conv3_wo_ent.indices.nelement() +
                                    x_conv4_wo_ent.features.element_size() * x_conv4_wo_ent.features.nelement() +
                                    x_conv4_wo_ent.indices.element_size() * x_conv4_wo_ent.indices.nelement()) * 1e-6
        end_mem_time = time.time() - mem_time

        end_time_wo_ent = time.time() - start_time_wo_ent - end_mem_time

        """
        :: WITH ENTROPY FOR TIME CALCS
        """
        start_time_entropy = time.time()
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)

        topn_indices = torch.topk(entropyOfFmapsSparse(x_conv2.features),
                                  int(x_conv1.features.shape[0] * self.top_percentage)).indices

        x_topn = spconv.SparseConvTensor(
            features=x_conv2.features[topn_indices],
            indices=x_conv2.indices[topn_indices],
            spatial_shape=x_conv2.spatial_shape,
            batch_size=x_conv2.batch_size
        )
        x_conv3 = self.conv3(x_topn)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)

        # Observation Memory Consumption:
        mem_time = time.time()
        memory_con = (
                             input_sp_tensor.features.element_size() * input_sp_tensor.features.nelement() +
                             input_sp_tensor.indices.element_size() * input_sp_tensor.indices.nelement() +
                             x.features.element_size() * x.features.nelement() +
                             x.indices.element_size() * x.indices.nelement() +
                             x_conv1.features.element_size() * x_conv1.features.nelement() +
                             x_conv1.indices.element_size() * x_conv1.indices.nelement() +
                             x_conv2.features.element_size() * x_conv2.features.nelement() +
                             x_conv2.indices.element_size() * x_conv2.indices.nelement() +
                             x_conv3.features.element_size() * x_conv3.features.nelement() +
                             x_conv3.indices.element_size() * x_conv3.indices.nelement() +
                             x_conv4.features.element_size() * x_conv4.features.nelement() +
                             x_conv4.indices.element_size() * x_conv4.indices.nelement()) * 1e-6

        batch_dict.update({
            'multi_scale_3d_features': {
                'input_': input_sp_tensor,
                'x_input': x,
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }})
        end_mem_time = time.time() - mem_time

        end_time_entropy = time.time() - start_time_entropy - end_mem_time

        # if student, insert feature adaptation layer
        if getattr(self, 'feat_adapt_single', None):
            self.feat_adapt_single(self.conv_out[0](x_conv4_wo_ent))
        if getattr(self, 'feat_adapt_autoencoder', None):
            self.feat_adapt_autoencoder(self.conv_out[0](x_conv4_wo_ent))
        if getattr(self, 'full_autoencoder', None):
            self.full_autoencoder(self.conv_out[0](x_conv4_wo_ent))

        # TODO: commented because clone_sp_tensor need modification of spconv_utils.py
        # if getattr(self, 'is_teacher', None):
        #     pre_act_encoded_spconv_tensor = clone_sp_tensor(out, batch_size)
        # out = replace_feature(out, self.final_act(out.features))

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
        })

        # comment to save memory
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        # Time & Memory Observation:
        batch_dict.update({
            'time_wo_ent': end_time_wo_ent,
            'time_entropy': end_time_entropy,
            'memory_con_wo_ent': memory_con_wo_ent,
            'memory_con': memory_con,
        })

        # TODO: Teacher stuff not used yet
        # if getattr(self, 'is_teacher', None):
        #     batch_dict.update({
        #         'encoded_spconv_tensor_pre-act': pre_act_encoded_spconv_tensor
        #     })

        return batch_dict


# Note: When classes are added to backbone_3d.py they must be added to pcdet/models/backbones_3d/__init__.py aswell
class VoxelResBackBone8xEntropy3(VoxelResBackBone8xEntropy1):
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        """
        :: WITHOUT ENTROPY FOR TIME CALCS
        """
        start_time_wo_ent = time.time()
        input_sp_tensor_wo_ent = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_wo_ent = self.conv_input(input_sp_tensor_wo_ent)

        x_conv1_wo_ent = self.conv1(x_wo_ent)
        x_conv2_wo_ent = self.conv2(x_conv1_wo_ent)
        x_conv3_wo_ent = self.conv3(x_conv2_wo_ent)
        x_conv4_wo_ent = self.conv4(x_conv3_wo_ent)
        out_wo_ent = self.conv_out(x_conv4_wo_ent)

        # Observation Memory Consumption:
        mem_time = time.time()
        memory_con_wo_ent = (
                                    input_sp_tensor_wo_ent.features.element_size() * input_sp_tensor_wo_ent.features.nelement() +
                                    input_sp_tensor_wo_ent.indices.element_size() * input_sp_tensor_wo_ent.indices.nelement() +
                                    x_wo_ent.features.element_size() * x_wo_ent.features.nelement() +
                                    x_wo_ent.indices.element_size() * x_wo_ent.indices.nelement() +
                                    x_conv1_wo_ent.features.element_size() * x_conv1_wo_ent.features.nelement() +
                                    x_conv1_wo_ent.indices.element_size() * x_conv1_wo_ent.indices.nelement() +
                                    x_conv2_wo_ent.features.element_size() * x_conv2_wo_ent.features.nelement() +
                                    x_conv2_wo_ent.indices.element_size() * x_conv2_wo_ent.indices.nelement() +
                                    x_conv3_wo_ent.features.element_size() * x_conv3_wo_ent.features.nelement() +
                                    x_conv3_wo_ent.indices.element_size() * x_conv3_wo_ent.indices.nelement() +
                                    x_conv4_wo_ent.features.element_size() * x_conv4_wo_ent.features.nelement() +
                                    x_conv4_wo_ent.indices.element_size() * x_conv4_wo_ent.indices.nelement()) * 1e-6
        end_mem_time = time.time() - mem_time

        end_time_wo_ent = time.time() - start_time_wo_ent - end_mem_time

        """
        :: WITH ENTROPY FOR TIME CALCS
        """
        start_time_entropy = time.time()
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        topn_indices = torch.topk(entropyOfFmapsSparse(x_conv3.features),
                                  int(x_conv3.features.shape[0] * self.top_percentage)).indices

        x_topn = spconv.SparseConvTensor(
            features=x_conv3.features[topn_indices],
            indices=x_conv3.indices[topn_indices],
            spatial_shape=x_conv3.spatial_shape,
            batch_size=x_conv3.batch_size
        )
        x_conv4 = self.conv4(x_topn)
        out = self.conv_out(x_conv4)

        # Observation Memory Consumption:
        mem_time = time.time()
        memory_con = (
                             input_sp_tensor.features.element_size() * input_sp_tensor.features.nelement() +
                             input_sp_tensor.indices.element_size() * input_sp_tensor.indices.nelement() +
                             x.features.element_size() * x.features.nelement() +
                             x.indices.element_size() * x.indices.nelement() +
                             x_conv1.features.element_size() * x_conv1.features.nelement() +
                             x_conv1.indices.element_size() * x_conv1.indices.nelement() +
                             x_conv2.features.element_size() * x_conv2.features.nelement() +
                             x_conv2.indices.element_size() * x_conv2.indices.nelement() +
                             x_conv3.features.element_size() * x_conv3.features.nelement() +
                             x_conv3.indices.element_size() * x_conv3.indices.nelement() +
                             x_conv4.features.element_size() * x_conv4.features.nelement() +
                             x_conv4.indices.element_size() * x_conv4.indices.nelement()) * 1e-6

        batch_dict.update({
            'multi_scale_3d_features': {
                'input_': input_sp_tensor,
                'x_input': x,
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }})
        end_mem_time = time.time() - mem_time

        end_time_entropy = time.time() - start_time_entropy - end_mem_time

        # if student, insert feature adaptation layer
        if getattr(self, 'feat_adapt_single', None):
            self.feat_adapt_single(self.conv_out[0](x_conv4_wo_ent))
        if getattr(self, 'feat_adapt_autoencoder', None):
            self.feat_adapt_autoencoder(self.conv_out[0](x_conv4_wo_ent))
        if getattr(self, 'full_autoencoder', None):
            self.full_autoencoder(self.conv_out[0](x_conv4_wo_ent))

        # TODO: commented because clone_sp_tensor need modification of spconv_utils.py
        # if getattr(self, 'is_teacher', None):
        #     pre_act_encoded_spconv_tensor = clone_sp_tensor(out, batch_size)
        # out = replace_feature(out, self.final_act(out.features))

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
        })

        # comment to save memory
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        # Time & Memory Observation:
        batch_dict.update({
            'time_wo_ent': end_time_wo_ent,
            'time_entropy': end_time_entropy,
            'memory_con_wo_ent': memory_con_wo_ent,
            'memory_con': memory_con,
        })

        # TODO: Teacher stuff not used yet
        # if getattr(self, 'is_teacher', None):
        #     batch_dict.update({
        #         'encoded_spconv_tensor_pre-act': pre_act_encoded_spconv_tensor
        #     })

        return batch_dict


# Note: When classes are added to backbone_3d.py they must be added to pcdet/models/backbones_3d/__init__.py aswell
class VoxelResBackBone8xEntropy4(VoxelResBackBone8xEntropy1):
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        """
        :: WITHOUT ENTROPY FOR TIME CALCS
        """
        start_time_wo_ent = time.time()
        input_sp_tensor_wo_ent = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_wo_ent = self.conv_input(input_sp_tensor_wo_ent)

        x_conv1_wo_ent = self.conv1(x_wo_ent)
        x_conv2_wo_ent = self.conv2(x_conv1_wo_ent)
        x_conv3_wo_ent = self.conv3(x_conv2_wo_ent)
        x_conv4_wo_ent = self.conv4(x_conv3_wo_ent)
        out_wo_ent = self.conv_out(x_conv4_wo_ent)

        # Observation Memory Consumption:
        mem_time = time.time()
        memory_con_wo_ent = (
                                    input_sp_tensor_wo_ent.features.element_size() * input_sp_tensor_wo_ent.features.nelement() +
                                    input_sp_tensor_wo_ent.indices.element_size() * input_sp_tensor_wo_ent.indices.nelement() +
                                    x_wo_ent.features.element_size() * x_wo_ent.features.nelement() +
                                    x_wo_ent.indices.element_size() * x_wo_ent.indices.nelement() +
                                    x_conv1_wo_ent.features.element_size() * x_conv1_wo_ent.features.nelement() +
                                    x_conv1_wo_ent.indices.element_size() * x_conv1_wo_ent.indices.nelement() +
                                    x_conv2_wo_ent.features.element_size() * x_conv2_wo_ent.features.nelement() +
                                    x_conv2_wo_ent.indices.element_size() * x_conv2_wo_ent.indices.nelement() +
                                    x_conv3_wo_ent.features.element_size() * x_conv3_wo_ent.features.nelement() +
                                    x_conv3_wo_ent.indices.element_size() * x_conv3_wo_ent.indices.nelement() +
                                    x_conv4_wo_ent.features.element_size() * x_conv4_wo_ent.features.nelement() +
                                    x_conv4_wo_ent.indices.element_size() * x_conv4_wo_ent.indices.nelement()) * 1e-6
        end_mem_time = time.time() - mem_time

        end_time_wo_ent = time.time() - start_time_wo_ent - end_mem_time

        """
        :: WITH ENTROPY FOR TIME CALCS
        """
        start_time_entropy = time.time()
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        topn_indices = torch.topk(entropyOfFmapsSparse(x_conv4.features),
                                  int(x_conv4.features.shape[0] * self.top_percentage)).indices

        x_topn = spconv.SparseConvTensor(
            features=x_conv4.features[topn_indices],
            indices=x_conv4.indices[topn_indices],
            spatial_shape=x_conv4.spatial_shape,
            batch_size=x_conv4.batch_size
        )
        out = self.conv_out(x_topn)

        # Observation Memory Consumption:
        mem_time = time.time()
        memory_con = (
                             input_sp_tensor.features.element_size() * input_sp_tensor.features.nelement() +
                             input_sp_tensor.indices.element_size() * input_sp_tensor.indices.nelement() +
                             x.features.element_size() * x.features.nelement() +
                             x.indices.element_size() * x.indices.nelement() +
                             x_conv1.features.element_size() * x_conv1.features.nelement() +
                             x_conv1.indices.element_size() * x_conv1.indices.nelement() +
                             x_conv2.features.element_size() * x_conv2.features.nelement() +
                             x_conv2.indices.element_size() * x_conv2.indices.nelement() +
                             x_conv3.features.element_size() * x_conv3.features.nelement() +
                             x_conv3.indices.element_size() * x_conv3.indices.nelement() +
                             x_conv4.features.element_size() * x_conv4.features.nelement() +
                             x_conv4.indices.element_size() * x_conv4.indices.nelement()) * 1e-6

        batch_dict.update({
            'multi_scale_3d_features': {
                'input_': input_sp_tensor,
                'x_input': x,
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }})
        end_mem_time = time.time() - mem_time

        end_time_entropy = time.time() - start_time_entropy - end_mem_time

        # if student, insert feature adaptation layer
        if getattr(self, 'feat_adapt_single', None):
            self.feat_adapt_single(self.conv_out[0](x_conv4_wo_ent))
        if getattr(self, 'feat_adapt_autoencoder', None):
            self.feat_adapt_autoencoder(self.conv_out[0](x_conv4_wo_ent))
        if getattr(self, 'full_autoencoder', None):
            self.full_autoencoder(self.conv_out[0](x_conv4_wo_ent))

        # TODO: commented because clone_sp_tensor need modification of spconv_utils.py
        # if getattr(self, 'is_teacher', None):
        #     pre_act_encoded_spconv_tensor = clone_sp_tensor(out, batch_size)
        # out = replace_feature(out, self.final_act(out.features))

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
        })

        # comment to save memory
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        # Time & Memory Observation:
        batch_dict.update({
            'time_wo_ent': end_time_wo_ent,
            'time_entropy': end_time_entropy,
            'memory_con_wo_ent': memory_con_wo_ent,
            'memory_con': memory_con,
        })

        # TODO: Teacher stuff not used yet
        # if getattr(self, 'is_teacher', None):
        #     batch_dict.update({
        #         'encoded_spconv_tensor_pre-act': pre_act_encoded_spconv_tensor
        #     })

        return batch_dict


# Note: When classes are added to backbone_3d.py they must be added to pcdet/models/backbones_3d/__init__.py aswell
class VoxelResBackBone8xImpNotImpGen(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # currently not used
        # act_fn = get_act_layer(self.model_cfg.get('ACT_FN', 'ReLU'))
        act_fn = nn.ReLU

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        if model_cfg.get('NUM_FILTERS', None):
            num_filters = model_cfg.NUM_FILTERS
        else:
            num_filters = [16, 16, 32, 64, 128, 128]

        if model_cfg.get('WIDTH', None):
            num_filters = (np.array(num_filters, dtype=np.int32) * model_cfg.WIDTH).astype(int)

        if model_cfg.get('LAYER_NUMS', None):
            layer_nums = model_cfg.LAYER_NUMS
        else:
            layer_nums = [1, 2, 3, 3, 3, 1]

        if model_cfg.get('FEAT_ADAPT_SINGLE', False):
            use_feat_adapt_single = model_cfg.FEAT_ADAPT_SINGLE
        else:
            use_feat_adapt_single = False

        if model_cfg.get('TOP_PERCENTAGE', False):
            self.top_percentage = model_cfg.TOP_PERCENTAGE

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            act_fn(),
        )
        block = partial(post_act_block, act_fn=act_fn)

        # conv1
        conv1_list = [
            SparseBasicBlock(num_filters[0], num_filters[1], norm_fn=norm_fn, act_fn=act_fn, indice_key='res1')]
        for k in range(layer_nums[1] - 1):
            conv1_list.append(
                SparseBasicBlock(num_filters[1], num_filters[1], norm_fn=norm_fn, act_fn=act_fn, indice_key='res1'))
        self.conv1 = spconv.SparseSequential(*conv1_list)

        # conv2
        conv2_list = [
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
        ]
        for k in range(layer_nums[2] - 1):
            conv2_list.append(
                SparseBasicBlock(num_filters[2], num_filters[2], norm_fn=norm_fn, act_fn=act_fn, indice_key='res2'))
        self.conv2 = spconv.SparseSequential(*conv2_list)

        # conv3
        conv3_list = [
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
        ]
        for k in range(layer_nums[3] - 1):
            conv3_list.append(SparseBasicBlock(
                num_filters[3], num_filters[3], norm_fn=norm_fn, act_fn=act_fn, indice_key='res3'
            ))
        self.conv3 = spconv.SparseSequential(*conv3_list)

        # conv4
        conv4_list = [
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[3], num_filters[4], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
                  conv_type='spconv'),
        ]
        for k in range(layer_nums[4] - 1):
            conv4_list.append(SparseBasicBlock(
                num_filters[4], num_filters[4], norm_fn=norm_fn, act_fn=act_fn, indice_key='res4'
            ))
        self.conv4 = spconv.SparseSequential(*conv4_list)

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[4], num_filters[5], (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(num_filters[5]))

        if use_feat_adapt_single is True:
            # the feature adaptation layer is used to adapt the feature dimension of the student to the teacher
            self.feat_adapt_single = spconv.SparseSequential(
                spconv.SparseConv3d(num_filters[5], 128, 1, stride=1, padding=0),
                nn.ReLU())
            self.feat_adapt_single[0].bias.requires_grad = True
            self.feat_adapt_single[0].weight.requires_grad = True
            nn.init.zeros_(self.feat_adapt_single[0].bias.data)
            nn.init.kaiming_normal_(self.feat_adapt_single[0].weight.data)
            # self.feat_adapt_single[0].bias.data.fill_(0.)
            # self.feat_adapt_single[0].weight.data.fill_(1.)

        self.final_act = act_fn()

        self.num_point_features = num_filters[5]
        self.backbone_channels = {
            'x_conv1': num_filters[1],
            'x_conv2': num_filters[2],
            'x_conv3': num_filters[3],
            'x_conv4': num_filters[4]
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        def mask_points_by_range(points, limit_range_min, limit_range_max):
            mask = (points[:, 0] >= limit_range_min[0]) & (points[:, 0] <= limit_range_max[0]) \
                   & (points[:, 1] >= limit_range_min[1]) & (points[:, 1] <= limit_range_max[1]) \
                   & (points[:, 2] >= limit_range_min[2]) & (points[:, 2] <= limit_range_max[2])
            return mask

        # Fixed defined:
        fm_stride = {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
        }
        point_cloud_range = torch.Tensor([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        voxel_size = torch.Tensor([0.1, 0.1, 0.2])

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        points = batch_dict['points'][:, 1:4]
        gtboxes = batch_dict['gt_boxes'][0]

        shape_iter = 0
        voxel_mask_list = []
        if gtboxes.shape[0] >= 1:
            for i in range(gtboxes.shape[0]):
                range_max_x = gtboxes[i, 0] + gtboxes[i, 3] / 2
                range_max_y = gtboxes[i, 1] + gtboxes[i, 4] / 2
                range_max_z = gtboxes[i, 2] + gtboxes[i, 5] / 2
                range_min_x = gtboxes[i, 0] - gtboxes[i, 3] / 2
                range_min_y = gtboxes[i, 1] - gtboxes[i, 4] / 2
                range_min_z = gtboxes[i, 2] - gtboxes[i, 5] / 2
                limit_range_min = torch.Tensor([range_min_x, range_min_y, range_min_z])
                limit_range_max = torch.Tensor([range_max_x, range_max_y, range_max_z])
                limit_range_max += 1
                limit_range_min -= 1
                mask = mask_points_by_range(points, limit_range_min, limit_range_max)

                pts_imp = points[mask]
                coords_imp_ = torch.floor(
                    (pts_imp - point_cloud_range[0:3].to('cuda:0')) / voxel_size.to('cuda:0'))  # / feature_map_stride!!
                coords_imp = torch.ones(coords_imp_.shape)
                coords_imp[:, 0] = coords_imp_[:, 2]
                coords_imp[:, 1] = coords_imp_[:, 1]
                coords_imp[:, 2] = coords_imp_[:, 0]

                # Here weiter - get the corresponding voxel coordinates:
                voxel_coords_ = voxel_coords[:, 1:]
                voxel_mask = (voxel_coords_[:, None] == coords_imp.to('cuda:0')).all(-1).any(-1).nonzero().flatten()
                voxel_mask_list.append(voxel_mask)
                imp_features = voxel_features[voxel_mask]
                shape_iter += imp_features.shape[0]

                if imp_features.shape[0] >= 1:
                    fn = ('/home/niko/fm_observation_Second/layer_0/imp/' +
                          batch_dict['metadata'][0]['token'] + '_{}'.format(i))
                    torch.save(imp_features, fn)

            bool_mask = torch.ones(voxel_features.shape[0]).to(torch.bool)
            mask_imp = torch.cat(voxel_mask_list)
            bool_mask[mask_imp] = False
            fea_no_imp = voxel_features[bool_mask]

            fn = ('/home/niko/fm_observation_Second/layer_0/noimp/' +
                  batch_dict['metadata'][0]['token'])
            torch.save(fea_no_imp, fn)

        # Reminder: Loading Files: list_ = glob.glob("/home/nleuze/promotion/code/ieee_sensors24/VoxelNeXt/fm_observation/layer0/imp/{}*".format(batch_dict['metadata'][0]['token']))

        if self.entropy_thresh is None:
            self.entropy_thresh = 1.0

        """
        :: WITH ENTROPY
        """
        start_time_entropy = time.time()
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)

        """
        :: Distribution Observation -- Feature Map 1
        """
        shape_iter = 0
        voxel_mask_list = []
        voxel_features_conv1 = x_conv1.features
        if gtboxes.shape[0] >= 1:
            for i in range(gtboxes.shape[0]):
                range_max_x = gtboxes[i, 0] + gtboxes[i, 3] / 2
                range_max_y = gtboxes[i, 1] + gtboxes[i, 4] / 2
                range_max_z = gtboxes[i, 2] + gtboxes[i, 5] / 2
                range_min_x = gtboxes[i, 0] - gtboxes[i, 3] / 2
                range_min_y = gtboxes[i, 1] - gtboxes[i, 4] / 2
                range_min_z = gtboxes[i, 2] - gtboxes[i, 5] / 2
                limit_range_min = torch.Tensor([range_min_x, range_min_y, range_min_z])
                limit_range_max = torch.Tensor([range_max_x, range_max_y, range_max_z])
                limit_range_max += 1
                limit_range_min -= 1
                mask = mask_points_by_range(points, limit_range_min, limit_range_max)

                pts_imp = points[mask]
                coords_imp_ = torch.floor(
                    (pts_imp - point_cloud_range[0:3].to('cuda:0')) / voxel_size.to('cuda:0') / fm_stride[
                        'x_conv1'])  # / feature_map_stride!!
                coords_imp = torch.ones(coords_imp_.shape)
                coords_imp[:, 0] = coords_imp_[:, 2]
                coords_imp[:, 1] = coords_imp_[:, 1]
                coords_imp[:, 2] = coords_imp_[:, 0]

                # Here weiter - get the corresponding voxel coordinates:
                voxel_coords_ = x_conv1.indices[:, 1:]
                voxel_mask = (voxel_coords_[:, None] == coords_imp.to('cuda:0')).all(-1).any(-1).nonzero().flatten()
                voxel_mask_list.append(voxel_mask)
                imp_features = voxel_features_conv1[
                    voxel_mask]  # @Nico Reminder: possible optimization when only using features with huge number of points - more closer analysis of the scene
                shape_iter += imp_features.shape[0]

                if imp_features.shape[0] >= 1:
                    dir = '/home/niko/fm_observation_Second/layer_1/imp/'
                    fn = f"{batch_dict['metadata'][0]['token'] + '_{}'.format(i)}"
                    os.makedirs(dir, exist_ok=True)
                    torch.save(imp_features, os.path.join(dir, fn))

            bool_mask = torch.ones(voxel_features_conv1.shape[0]).to(torch.bool)
            mask_imp = torch.cat(voxel_mask_list)
            bool_mask[mask_imp] = False
            fea_no_imp = voxel_features_conv1[bool_mask]
            dir = '/home/niko/fm_observation_Second/layer_1/noimp/'
            fn = f"{batch_dict['metadata'][0]['token']}"
            os.makedirs(dir, exist_ok=True)
            torch.save(fea_no_imp, os.path.join(dir, fn))

        x_conv2 = self.conv2(x_conv1)

        """
        :: Distribution Observation -- Feature Map 2
        """
        shape_iter = 0
        voxel_mask_list = []
        voxel_features_conv2 = x_conv2.features
        if gtboxes.shape[0] >= 1:
            for i in range(gtboxes.shape[0]):
                range_max_x = gtboxes[i, 0] + gtboxes[i, 3] / 2
                range_max_y = gtboxes[i, 1] + gtboxes[i, 4] / 2
                range_max_z = gtboxes[i, 2] + gtboxes[i, 5] / 2
                range_min_x = gtboxes[i, 0] - gtboxes[i, 3] / 2
                range_min_y = gtboxes[i, 1] - gtboxes[i, 4] / 2
                range_min_z = gtboxes[i, 2] - gtboxes[i, 5] / 2
                limit_range_min = torch.Tensor([range_min_x, range_min_y, range_min_z])
                limit_range_max = torch.Tensor([range_max_x, range_max_y, range_max_z])
                limit_range_max += 1
                limit_range_min -= 1
                mask = mask_points_by_range(points, limit_range_min, limit_range_max)

                pts_imp = points[mask]
                coords_imp_ = torch.floor(
                    (pts_imp - point_cloud_range[0:3].to('cuda:0')) / voxel_size.to('cuda:0') / fm_stride['x_conv2'])
                coords_imp = torch.ones(coords_imp_.shape)
                coords_imp[:, 0] = coords_imp_[:, 2]
                coords_imp[:, 1] = coords_imp_[:, 1]
                coords_imp[:, 2] = coords_imp_[:, 0]

                # Here weiter - get the corresponding voxel coordinates:
                voxel_coords_ = x_conv2.indices[:, 1:]
                voxel_mask = (voxel_coords_[:, None] == coords_imp.to('cuda:0')).all(-1).any(-1).nonzero().flatten()
                voxel_mask_list.append(voxel_mask)
                imp_features = voxel_features_conv2[voxel_mask]
                shape_iter += imp_features.shape[0]

                if imp_features.shape[0] >= 1:
                    dir = '/home/niko/fm_observation_Second/layer_2/imp/'
                    fn = f"{batch_dict['metadata'][0]['token'] + '_{}'.format(i)}"
                    os.makedirs(dir, exist_ok=True)
                    torch.save(imp_features, os.path.join(dir, fn))

            bool_mask = torch.ones(voxel_features_conv2.shape[0]).to(torch.bool)
            mask_imp = torch.cat(voxel_mask_list)
            bool_mask[mask_imp] = False
            fea_no_imp = voxel_features_conv2[bool_mask]
            dir = '/home/niko/fm_observation_Second/layer_2/noimp/'
            fn = f"{batch_dict['metadata'][0]['token']}"
            os.makedirs(dir, exist_ok=True)
            torch.save(fea_no_imp, os.path.join(dir, fn))

        x_conv3 = self.conv3(x_conv2)

        """
        :: Distribution Observation -- Feature Map 3
        """
        shape_iter = 0
        voxel_mask_list = []
        voxel_features_conv3 = x_conv3.features
        if gtboxes.shape[0] >= 1:
            for i in range(gtboxes.shape[0]):
                range_max_x = gtboxes[i, 0] + gtboxes[i, 3] / 2
                range_max_y = gtboxes[i, 1] + gtboxes[i, 4] / 2
                range_max_z = gtboxes[i, 2] + gtboxes[i, 5] / 2
                range_min_x = gtboxes[i, 0] - gtboxes[i, 3] / 2
                range_min_y = gtboxes[i, 1] - gtboxes[i, 4] / 2
                range_min_z = gtboxes[i, 2] - gtboxes[i, 5] / 2
                limit_range_min = torch.Tensor([range_min_x, range_min_y, range_min_z])
                limit_range_max = torch.Tensor([range_max_x, range_max_y, range_max_z])
                limit_range_max += 1
                limit_range_min -= 1
                mask = mask_points_by_range(points, limit_range_min, limit_range_max)

                pts_imp = points[mask]
                coords_imp_ = torch.floor(
                    (pts_imp - point_cloud_range[0:3].to('cuda:0')) / voxel_size.to('cuda:0') / fm_stride['x_conv3'])
                coords_imp = torch.ones(coords_imp_.shape)
                coords_imp[:, 0] = coords_imp_[:, 2]
                coords_imp[:, 1] = coords_imp_[:, 1]
                coords_imp[:, 2] = coords_imp_[:, 0]

                # Here weiter - get the corresponding voxel coordinates:
                voxel_coords_ = x_conv3.indices[:, 1:]
                voxel_mask = (voxel_coords_[:, None] == coords_imp.to('cuda:0')).all(-1).any(-1).nonzero().flatten()
                voxel_mask_list.append(voxel_mask)
                imp_features = voxel_features_conv3[voxel_mask]
                shape_iter += imp_features.shape[0]

                if imp_features.shape[0] >= 1:
                    dir = '/home/niko/fm_observation_Second/layer_3/imp/'
                    fn = f"{batch_dict['metadata'][0]['token'] + '_{}'.format(i)}"
                    os.makedirs(dir, exist_ok=True)
                    torch.save(imp_features, os.path.join(dir, fn))

            bool_mask = torch.ones(voxel_features_conv3.shape[0]).to(torch.bool)
            mask_imp = torch.cat(voxel_mask_list)
            bool_mask[mask_imp] = False
            fea_no_imp = voxel_features_conv3[bool_mask]
            dir = '/home/niko/fm_observation_Second/layer_3/noimp/'
            fn = f"{batch_dict['metadata'][0]['token']}"
            os.makedirs(dir, exist_ok=True)
            torch.save(fea_no_imp, os.path.join(dir, fn))

        x_conv4 = self.conv4(x_conv3)

        """
        :: Distribution Observation -- Feature Map 4
        """
        shape_iter = 0
        voxel_mask_list = []
        voxel_features_conv4 = x_conv4.features
        if gtboxes.shape[0] >= 1:
            for i in range(gtboxes.shape[0]):
                range_max_x = gtboxes[i, 0] + gtboxes[i, 3] / 2
                range_max_y = gtboxes[i, 1] + gtboxes[i, 4] / 2
                range_max_z = gtboxes[i, 2] + gtboxes[i, 5] / 2
                range_min_x = gtboxes[i, 0] - gtboxes[i, 3] / 2
                range_min_y = gtboxes[i, 1] - gtboxes[i, 4] / 2
                range_min_z = gtboxes[i, 2] - gtboxes[i, 5] / 2
                limit_range_min = torch.Tensor([range_min_x, range_min_y, range_min_z])
                limit_range_max = torch.Tensor([range_max_x, range_max_y, range_max_z])
                limit_range_max += 1
                limit_range_min -= 1
                mask = mask_points_by_range(points, limit_range_min, limit_range_max)

                pts_imp = points[mask]
                coords_imp_ = torch.floor(
                    (pts_imp - point_cloud_range[0:3].to('cuda:0')) / voxel_size.to('cuda:0') / fm_stride['x_conv4'])
                coords_imp = torch.ones(coords_imp_.shape)
                coords_imp[:, 0] = coords_imp_[:, 2]
                coords_imp[:, 1] = coords_imp_[:, 1]
                coords_imp[:, 2] = coords_imp_[:, 0]

                # Here weiter - get the corresponding voxel coordinates:
                voxel_coords_ = x_conv4.indices[:, 1:]
                voxel_mask = (voxel_coords_[:, None] == coords_imp.to('cuda:0')).all(-1).any(-1).nonzero().flatten()
                voxel_mask_list.append(voxel_mask)
                imp_features = voxel_features_conv4[voxel_mask]
                shape_iter += imp_features.shape[0]

                dir = '/home/niko/fm_observation_Second/layer_4/imp/'
                fn = f"{batch_dict['metadata'][0]['token'] + '_{}'.format(i)}"
                os.makedirs(dir, exist_ok=True)
                torch.save(imp_features, os.path.join(dir, fn))

            bool_mask = torch.ones(voxel_features_conv4.shape[0]).to(torch.bool)
            mask_imp = torch.cat(voxel_mask_list)
            bool_mask[mask_imp] = False
            fea_no_imp = voxel_features_conv4[bool_mask]
            dir = '/home/niko/fm_observation_Second/layer_4/noimp/'
            fn = f"{batch_dict['metadata'][0]['token']}"
            os.makedirs(dir, exist_ok=True)
            torch.save(fea_no_imp, os.path.join(dir, fn))
        else:
            self.check_gts += 1
            print('noGT ' + '+ ' * 15, self.check_gts)

        out = self.conv_out(x_conv4)

        end_time_entropy = time.time() - start_time_entropy

        # if student, insert feature adaptation layer
        if getattr(self, 'feat_adapt_single', None):
            self.feat_adapt_single(self.conv_out[0](x_conv4))

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
        })

        # comment to save memory
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x_old(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict
