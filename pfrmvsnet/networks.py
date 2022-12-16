from audioop import bias
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from pointmvsnet.functions.gather_knn import gather_knn
from pointmvsnet.nn.conv import *

import scipy.io as io

def CloudWalk(feature, knn_inds):
    batch_size, channels, num_points = feature.shape     
    k = knn_inds.shape[2] 
    new_group = []
    curve_length = 16
    center_feature = feature.unsqueeze(-1).permute(0, 2, 3, 1) 
    neighbour_feature = gather_knn(feature, knn_inds) 
        
    NN_similarity = torch.matmul(center_feature, neighbour_feature.permute(0, 2, 1, 3)) 
    NN_similarity = NN_similarity.squeeze(2) 
    NN_attention = F.softmax(NN_similarity, -1)
    _, top3_knn = torch.topk(NN_attention, 4, dim=-1, largest=True, sorted=True) 

    tmp_idx = top3_knn.view(batch_size * num_points, -1) 

    for step in range(curve_length):
        if step == 0:
            # top-2
            next_point_index = top3_knn[:, :, 1].unsqueeze(-1) # (b, n, 1)
            new_group.append(next_point_index)

        else:
            next_point_index = tmp_idx[next_point_index.view(-1), 1]
            next_point_index = next_point_index.view(-1, 1)
            next_point_index = next_point_index.view(batch_size, num_points, 1)
            new_group.append(next_point_index)

    NN_new_group = torch.cat(new_group, dim=-1) 
    return NN_new_group
        
            
# structure
class Structure_Information(nn.Module):
    def __init__(self, channels):
        super(Structure_Information, self).__init__()
        # SFA stage
        self.Q = nn.Linear(channels, channels)
        self.K = nn.Linear(channels, channels)
        self.V = nn.Linear(channels, channels)

        self.BN = nn.BatchNorm1d(channels)
        self.struc_Action = nn.ReLU(inplace=True)

    def forward(self, feature, knn_inds):
        batch_size, channels, num_points = feature.shape       
        k = knn_inds.shape[2] # (B, num_points, k)

        if feature.is_cuda:
            # custom improved gather
            neighbour_feature = gather_knn(feature, knn_inds)      
        else:
            # pytorch gather
            knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_points, k)
            edge_feature_expand = feature.unsqueeze(2).expand(batch_size, -1, num_points, num_points)
            neighbour_feature = torch.gather(edge_feature_expand, 3, knn_inds_expand) 

        center_feature = feature.unsqueeze(-1).expand(-1, -1, -1, k) 
        structure_feature = (center_feature - neighbour_feature).permute(0, 2, 3, 1) 

        # weighted sum
        structure_feature_Q = self.Q(structure_feature) 
        structure_feature_K = self.K(structure_feature).permute(0, 1, 3, 2) 
        structure_feature_V = self.V(neighbour_feature.permute(0, 2, 3, 1)) 
        structure_feature_similarity1 = torch.matmul(structure_feature_Q, structure_feature_K) 
        structure_feature_attention1 = F.softmax(structure_feature_similarity1, dim=-1)
        structure_feature1 = torch.einsum('bnkk, bnkc->bcn', structure_feature_attention1, structure_feature_V)

        mean_Q = torch.mean(structure_feature_Q, dim=-1, keepdim=True).expand(-1,-1, -1, channels) 
        mean_K = torch.mean(structure_feature_K, dim=2, keepdim=True).expand(-1,-1, channels, -1) 
        structure_feature_similarity2 = (torch.matmul((structure_feature_Q - mean_Q), (structure_feature_K - mean_K))) / torch.matmul(torch.norm((structure_feature_Q - mean_Q), dim=-1, keepdim=True), torch.norm((structure_feature_K - mean_K), dim=2, keepdim=True)) # (b, n, k, k)
        structure_feature_attention2 = F.softmax(structure_feature_similarity2, dim=-1)
        structure_feature2 = torch.einsum('bnkk, bnkc->bcn', structure_feature_attention2, structure_feature_V)
        structure_feature_weighted = 0.6 * structure_feature1 + 0.4 * structure_feature2
        structure_feature_weighted= self.struc_Action(self.BN(structure_feature_weighted))
        

        return  structure_feature_weighted
           
class Self_Attention_Layer(nn.Module):
    def __init__(self, channels):
        super(Self_Attention_Layer, self).__init__()
        self.Q = nn.Linear(channels, channels)
        self.K = nn.Linear(channels, channels)
        self.V = nn.Linear(channels, channels)

        self.BN = nn.BatchNorm1d(channels)
        self.Action = nn.ReLU(inplace=True)
    
    def forward(self, feature, knn_inds):

        batch_size, channels, num_points = feature.shape       
        k = knn_inds.shape[2] # (B, num_points, k)
        
        if feature.is_cuda:
            # custom improved gather
            walk_feature = gather_knn(feature, knn_inds)               
        else:
            # pytorch gather
            knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_points, k)
            edge_feature_expand = feature.unsqueeze(2).expand(batch_size, -1, num_points, num_points)
            walk_feature = torch.gather(edge_feature_expand, 3, knn_inds_expand) 

        center_feature = feature.unsqueeze(-1).permute(0, 2, 3, 1) 
        walk_feature = torch.cat((walk_feature.permute(0, 2, 3, 1), center_feature), dim=2) 
        
        KNN_Q = self.Q(walk_feature)
        KNN_K = self.K(walk_feature).permute(0, 1, 3, 2) 
        KNN_V = self.V(walk_feature)
        similarity = torch.matmul(KNN_Q, KNN_K) 
        attention = F.softmax(similarity, dim=-1) 
        x_V_Attenion = torch.einsum('bnkk, bnkc->bnc', attention, KNN_V) 
        x_V_Attenion = x_V_Attenion.permute(0, 2, 1)
        x_V_Attenion = self.Action(self.BN(x_V_Attenion))
        feature = feature + x_V_Attenion
        return feature

class PointMVS_Transformer(nn.Module):
    def __init__(self, channels):
        super(PointMVS_Transformer, self).__init__()
        self.SAL_x = Self_Attention_Layer(channels)
        self.SAL_y = Self_Attention_Layer(channels)
        self.SAL_z = Self_Attention_Layer(channels)
    
    def forward(self, feature, knn_idx):
        # select point with feature 
        walk_index = CloudWalk(feature, knn_idx)
        
        head1 = self.SAL_x(feature,  walk_index)         
        head2 = self.SAL_y(head1, walk_index)           
        head3 = self.SAL_z(head2,  walk_index)

        all_feature = torch.cat((head1, head2, head3), dim=1)
        
        return  all_feature

# class PositionalEncoding(nn.Module):
    
#     def __init__(self, d_hid, n_position):
#         super(PositionalEncoding, self).__init__()

#         # Not a parameter
#         self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

#     def _get_sinusoid_encoding_table(self, n_position, d_hid):
#         ''' Sinusoid position encoding table '''
#         # TODO: make it with torch instead of numpy

#         def get_position_angle_vec(position):
#             position2 = [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
#             return position2

#         sinusoid_table = torch.cat(get_position_angle_vec(n_position), dim=2)
#         sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
#         sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1
#         return sinusoid_table

#     def forward(self, x):
#         # x(B,N,d)
#         return self.pos_table[:, :x.size(1)].clone().detach()


class ImageConv(nn.Module):
    def __init__(self, base_channels):
        super(ImageConv, self).__init__()
        self.base_channels = base_channels
        self.out_channels = 8 * base_channels
        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1, bias=False)
        )


    def forward(self, imgs):
        out_dict = {}

        conv0 = self.conv0(imgs)
        out_dict["conv0"] = conv0
        conv1 = self.conv1(conv0)
        out_dict["conv1"] = conv1 
        conv2 = self.conv2(conv1)
        out_dict["conv2"] = conv2        
        conv3 = self.conv3(conv2)
        out_dict["conv3"] = conv3

        return out_dict


class VolumeConv(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(VolumeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = base_channels * 8
        self.base_channels = base_channels
        self.conv1_0 = Conv3d(in_channels, base_channels * 2, 3, stride=2, padding=1)
        self.conv2_0 = Conv3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv3_0 = Conv3d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)

        self.conv0_1 = Conv3d(in_channels, base_channels, 3, 1, padding=1)

        self.conv1_1 = Conv3d(base_channels * 2, base_channels * 2, 3, 1, padding=1)
        self.conv2_1 = Conv3d(base_channels * 4, base_channels * 4, 3, 1, padding=1)

        self.conv3_1 = Conv3d(base_channels * 8, base_channels * 8, 3, 1, padding=1)
        self.conv4_0 = Deconv3d(base_channels * 8, base_channels * 4, 3, 2, padding=1, output_padding=1)
        self.conv5_0 = Deconv3d(base_channels * 4, base_channels * 2, 3, 2, padding=1, output_padding=1)
        self.conv6_0 = Deconv3d(base_channels * 2, base_channels, 3, 2, padding=1, output_padding=1)

        self.conv6_2 = nn.Conv3d(base_channels, 1, 3, padding=1, bias=False)

    def forward(self, x):
        conv0_1 = self.conv0_1(x)

        conv1_0 = self.conv1_0(x)
        conv2_0 = self.conv2_0(conv1_0)
        conv3_0 = self.conv3_0(conv2_0)

        conv1_1 = self.conv1_1(conv1_0)
        conv2_1 = self.conv2_1(conv2_0)
        conv3_1 = self.conv3_1(conv3_0)

        conv4_0 = self.conv4_0(conv3_1)

        conv5_0 = self.conv5_0(conv4_0 + conv2_1)
        conv6_0 = self.conv6_0(conv5_0 + conv1_1)

        conv6_2 = self.conv6_2(conv6_0 + conv0_1)

        return conv6_2


class MAELoss(nn.Module):
    def forward(self, pred_depth_image, gt_depth_image, depth_interval):
        """non zero mean absolute loss for one batch"""
        depth_interval = depth_interval.view(-1)
        mask_valid = (~torch.eq(gt_depth_image, 0.0)).type(torch.float)
        denom = torch.sum(mask_valid, dim=(1, 2, 3)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(pred_depth_image - gt_depth_image)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2, 3))
        masked_mae = torch.sum((masked_mae / depth_interval) / denom)

        return masked_mae


class Valid_MAELoss(nn.Module):
    def __init__(self, valid_threshold=2.0):
        super(Valid_MAELoss, self).__init__()
        self.valid_threshold = valid_threshold

    def forward(self, pred_depth_image, gt_depth_image, depth_interval, before_depth_image):
        """non zero mean absolute loss for one batch"""
        pred_height = pred_depth_image.size(2)
        pred_width = pred_depth_image.size(3)
        depth_interval = depth_interval.view(-1)
        mask_true = (~torch.eq(gt_depth_image, 0.0)).type(torch.float)
        before_hight = before_depth_image.size(2)
        if before_hight != pred_height:
            before_depth_image = F.interpolate(before_depth_image, (pred_height, pred_width))
        diff = torch.abs(gt_depth_image - before_depth_image) / depth_interval.view(-1, 1, 1, 1)
        mask_valid = (diff < self.valid_threshold).type(torch.float)
        mask_valid = mask_true * mask_valid
        denom = torch.sum(mask_valid, dim=(1, 2, 3)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(pred_depth_image - gt_depth_image)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2, 3))
        masked_mae = torch.sum((masked_mae / depth_interval) / denom)

        return masked_mae
