import torchvision
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 0: -2.05
# 1: -2.00
# 2: -1.96
# 3: -1.91
# 4: -1.87
# 5: -1.83
# 6: -1.78
# 7: -1.74
# 8: -1.69
# 9: -1.65
# 10: -1.61
# 11: -1.56
# 12: -1.52
# 13: -1.47
# 14: -1.43
# 15: -1.39
# 16: -1.34
# 17: -1.30
# 18: -1.25
# 19: -1.21
# 20: -1.7

class DarknessDetectionLayer(nn.Module):
    def __init__(self, num_classes, darkness_threshold=-1.9):
        super(DarknessDetectionLayer, self).__init__()
        self.darkness_threshold = darkness_threshold
        self.num_classes = num_classes
        
    def forward(self, x):
        batch_size = x.size(0)
        brightness = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        avg_brightness = torch.mean(brightness, dim=[1, 2])
        dark_mask = (avg_brightness < self.darkness_threshold)
        self.dark_mask = dark_mask
        return x, dark_mask

class ResNet18WithAuxiliary(nn.Module):
    def __init__(self, num_classes=1000, aux_weight=0.4, use_grayscale=False, 
                 darkness_threshold=-1.9, dropout_rate=0.5, 
                 use_pretrained=True):
        super(ResNet18WithAuxiliary, self).__init__()
        
        self.use_grayscale = use_grayscale
        self.num_classes = num_classes
        
        if(use_pretrained == True) :
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        else :
            self.resnet = resnet18(num_classes=num_classes)
        # 원본 conv1 저장
        self.original_conv1 = self.resnet.conv1
        
        # 어두운 이미지 감지 레이어
        self.darkness_layer = DarknessDetectionLayer(num_classes, darkness_threshold)
        
        # ResNet18의 첫 번째 레이어 수정
        self.resnet.conv1 = nn.Identity()
        
        # 원래 FC 층 제거
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # 중간 층 특성 추출을 위한 훅 등록
        self.aux_features = None
        
        # 보조 분류기 1: layer2 다음에 배치 (128 채널)
        self.auxiliary1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 보조 분류기 2: layer3 다음에 배치 (256 채널)
        self.auxiliary2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 최종 분류기
        self.fc = nn.Linear(in_features, num_classes)

        # 보조 손실 가중치
        self.aux_weight = aux_weight
        
        # layer2와 layer3 이후 훅 등록
        self.hooks = []
        self.layer2_output = None
        self.layer3_output = None
        
        self.hooks.append(self.resnet.layer2.register_forward_hook(self._hook_layer2))
        self.hooks.append(self.resnet.layer3.register_forward_hook(self._hook_layer3))
    
    def _hook_layer2(self, module, input, output):
        self.layer2_output = output
    
    def _hook_layer3(self, module, input, output):
        self.layer3_output = output
    
    def _convert_to_grayscale(self, x):
        x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x_gray = x_gray.unsqueeze(1)
        x_gray = x_gray.expand(-1, 3, -1, -1)
        return x_gray
    
    def _freeze_backbone(self):
        """백본 네트워크를 동결"""
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """백본 네트워크 동결 해제"""
        for param in self.resnet.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        batch_size = x.size(0)
        
        if self.use_grayscale:
            x = self._convert_to_grayscale(x)
        
        # DarknessDetectionLayer 단계에서 백본 동결
        self._freeze_backbone()
        x, dark_mask = self.darkness_layer(x)
        
        # 이후 단계에서 백본 동결 해제
        self._unfreeze_backbone()
        
        feature_h, feature_w = 112, 224
        processed_features = torch.zeros(batch_size, 64, feature_h, feature_w, device=x.device)
        
        # 경고가 발생하는 조건문 수정
        # if not torch.all(dark_mask): 대신 텐서 연산 사용
        conv_output = self.original_conv1(x)
        
        # 마스크 차원 조정 (batch_size, 1, 1, 1)로 변환
        # mask_expanded = (~dark_mask).view(batch_size, 1, 1, 1).float()
        mask_expanded = (1 - dark_mask.float()).view(batch_size, 1, 1, 1)
        
        # 마스크를 사용하여 요소별 곱셈 수행
        processed_features = conv_output * mask_expanded
        
        x = processed_features
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)  # hook 작동
        x = self.resnet.layer3(x)  # hook 작동
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        main_output = self.fc(x)
        
        uniform_prob = torch.ones(batch_size, self.num_classes, device=x.device) / self.num_classes
        
        # 경고가 발생하는 조건문 수정
        # 각 샘플에 대해 조건부 처리 대신 마스크 연산 사용
        dark_mask_expanded = dark_mask.view(batch_size, 1).float()
        
        # dark_mask가 1이면 uniform_prob를 사용, 0이면 main_output 사용
        main_output = main_output * (1 - dark_mask_expanded) + uniform_prob * dark_mask_expanded
        
        if self.training:
            aux1_output = self.auxiliary1(self.layer2_output)
            aux2_output = self.auxiliary2(self.layer3_output)
            
            # 보조 출력도 같은 방식으로 수정
            aux1_output = aux1_output * (1 - dark_mask_expanded) + uniform_prob * dark_mask_expanded
            aux2_output = aux2_output * (1 - dark_mask_expanded) + uniform_prob * dark_mask_expanded
            
            return main_output, aux1_output, aux2_output
        else:
            return main_output
    
    def calculate_loss(self, criterion, outputs, targets):
        if self.training:
            main_output, aux1_output, aux2_output = outputs
            main_loss = criterion(main_output, targets)
            aux1_loss = criterion(aux1_output, targets)
            aux2_loss = criterion(aux2_output, targets)
            total_loss = main_loss + self.aux_weight * (aux1_loss + aux2_loss)
            return total_loss
        else:
            return criterion(outputs, targets)
