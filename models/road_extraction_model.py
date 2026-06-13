import torch
import torch.nn as nn
import torchvision.models as models
import timm


class RoadExtractionModel(nn.Module):
    def __init__(
        self,
        encoder='swin_tiny',
        num_classes=1,
        input_size=224,
        swin_img_size=None,
        orient_num_bins=0,
    ):
        super(RoadExtractionModel, self).__init__()
        self.encoder_name = encoder
        self.orient_num_bins = int(orient_num_bins)
        # timm Swin 的 img_size（默认同 input_size，用于 City-Scale 512 等非 224 输入）
        if swin_img_size is None:
            if isinstance(input_size, (tuple, list)):
                self._swin_img_size = int(input_size[0])
            else:
                self._swin_img_size = int(input_size)
        else:
            self._swin_img_size = int(swin_img_size)
        
        # 编码器
        if encoder == 'resnet50':
            self.encoder = models.resnet50(pretrained=True)
            # 移除最后的全连接层
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
            self.encoder_channels = [256, 512, 1024, 2048]  # ResNet-50的特征通道数
        elif encoder == 'swin_tiny':
            # 使用Swin-Tiny作为编码器（img_size 可配置以适配 512 等裁剪块）
            self.encoder = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=True,
                features_only=True,
                img_size=self._swin_img_size,
            )
            # 与 forward 中 reverse 后的顺序一致：从最深层到最浅层
            self.encoder_channels = [768, 384, 192, 96]  # Swin-Tiny特征通道（deep->shallow）
        else:
            raise NotImplementedError(f"Encoder {encoder} not implemented")
        
        # 解码器 - 使用FPN结构
        self.decoder = FPN(self.encoder_channels)
        
        # 三个任务头
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        self.intersection_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        _oc = self.orient_num_bins if self.orient_num_bins > 0 else 3
        self.orientation_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, _oc, kernel_size=1),
        )
    
    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        # 编码器前向传播
        if self.encoder_name == 'swin_tiny':
            # Swin Transformer 输出格式
            features = self.encoder(x)
            # 只取前四个特征图
            features = features[:4]
            # timm 的 Swin features_only 返回 BHWC，这里转为 BCHW
            normalized_features = []
            for feat in features:
                if feat.ndim == 4 and feat.shape[-1] in (96, 192, 384, 768):
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                normalized_features.append(feat)
            features = normalized_features
        else:
            # ResNet 输出格式
            features = []
            # 第一阶段
            x = self.encoder[0](x)  # conv1
            x = self.encoder[1](x)  # bn1
            x = self.encoder[2](x)  # relu
            x = self.encoder[3](x)  # maxpool
            
            # 第二阶段
            x = self.encoder[4](x)  # layer1
            features.append(x)  # C2
            
            # 第三阶段
            x = self.encoder[5](x)  # layer2
            features.append(x)  # C3
            
            # 第四阶段
            x = self.encoder[6](x)  # layer3
            features.append(x)  # C4
            
            # 第五阶段
            x = self.encoder[7](x)  # layer4
            features.append(x)  # C5
        
        features = features[::-1]  # 从最深层到最浅层
        
        # 解码器前向传播
        x = self.decoder(features)
        
        # 三个任务头前向传播
        segmentation = self.segmentation_head(x)
        intersection = self.intersection_head(x)
        orientation = self.orientation_head(x)

        # 与标签分辨率对齐（恢复到输入图像尺寸）
        segmentation = nn.functional.interpolate(segmentation, size=(input_h, input_w), mode='bilinear', align_corners=False)
        intersection = nn.functional.interpolate(intersection, size=(input_h, input_w), mode='bilinear', align_corners=False)
        orientation = nn.functional.interpolate(orientation, size=(input_h, input_w), mode='bilinear', align_corners=False)
        
        return {
            'segmentation': segmentation,
            'intersection': intersection,
            'orientation': orientation
        }

class FPN(nn.Module):
    def __init__(self, encoder_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        # 横向连接卷积
        for i, channels in enumerate(encoder_channels):
            self.lateral_convs.append(
                nn.Conv2d(channels, 256, kernel_size=1)
            )
        
        # 输出卷积
        for i in range(len(encoder_channels)):
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU()
                )
            )
    
    def forward(self, features):
        # 从最深层开始
        x = self.lateral_convs[0](features[0])
        outputs = [self.output_convs[0](x)]
        
        for i in range(1, len(features)):
            # 上采样
            x = nn.functional.interpolate(x, size=features[i].shape[2:], mode='nearest')
            # 横向连接
            x = x + self.lateral_convs[i](features[i])
            # 输出卷积
            outputs.append(self.output_convs[i](x))
        
        # 返回最后一个输出（最浅层）
        return outputs[-1]
