import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# ResNet50のベースモデル
class BasicConvClassifier(nn.Module):
    def __init__(self, num_classes, seq_lens, in_channels, hid_dim:int =128):
        super(BasicConvClassifier, self).__init__()

        # conv1
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding="same")
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3)

        # conv2_x
        self.conv2_x = self._make_layer(block, 3, res_block_in_channels=64, first_conv_out_channels=64)

        # conv3_x以降
        self.conv3_x = self._make_layer(block, 4, res_block_in_channels=256,  first_conv_out_channels=128)
        self.conv4_x = self._make_layer(block, 6, res_block_in_channels=512,  first_conv_out_channels=256)
        self.conv5_x = self._make_layer(block, 3, res_block_in_channels=1024, first_conv_out_channels=512)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self,x):

        x = self.conv1(x)   
        x = self.bn1(x)     
        x = self.relu(x)    
        x = self.maxpool(x) 

        x = self.conv2_x(x)  
        x = self.conv3_x(x)  
        x = self.conv4_x(x)  
        x = self.conv5_x(x)  
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride=1):
        layers = []

        identity_conv = nn.Conv1d(res_block_in_channels, first_conv_out_channels*4, kernel_size=1,stride=stride)
        layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

        in_channels = first_conv_out_channels*4

        for i in range(num_res_blocks - 1):
            layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

        return nn.Sequential(*layers)

#２層目以降のResNetブロック
class block(nn.Module):
    def __init__(self, first_conv_in_channels, first_conv_out_channels, identity_conv=None, stride=1):
    
        super(block, self).__init__()

        # 1番目のconv層
        self.conv1 = nn.Conv1d(
            first_conv_in_channels, first_conv_out_channels, kernel_size=1, stride=1, padding="same")
        self.bn1 = nn.BatchNorm1d(first_conv_out_channels)

        # 2番目のconv層
        self.conv2 = nn.Conv1d(
            first_conv_out_channels, first_conv_out_channels, kernel_size=3, stride=stride, padding="same")
        self.bn2 = nn.BatchNorm1d(first_conv_out_channels)

        # 3番目のconv層
        self.conv3 = nn.Conv1d(
            first_conv_out_channels, first_conv_out_channels*4, kernel_size=1, stride=1, padding="same")
        self.bn3 = nn.BatchNorm1d(first_conv_out_channels*4)
        self.relu = nn.ReLU()

        self.identity_conv = identity_conv

    def forward(self, x):

        identity = x.clone()  

        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x) 
        x = self.bn3(x)

        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity

        x = self.relu(x)

        return x
