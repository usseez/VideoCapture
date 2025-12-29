# check my image through GradCAM input: image, output : save image

import torch
from torchvision import models, transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.nn.functional import softmax
import torch.nn as nn 
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import os
import cv2
import pandas as pd
import sys  
from io import StringIO
import torchvision
from modify_resnet18 import ResNet18WithAuxiliary

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import List, Dict
import numpy as np
import timm
import time
import torchvision

# 테스트 이미지 경로
num_classes = 15  # 저장된 모델에 따라 클래스 수를 설정
model_date = '2025-08-14-18-50-50'
test_dir = '/home/ubuntu/Dataset/99.dataset_train/Jeonju'
model_path = f'/home/ubuntu/networkbuild/pre_train_weights/weights/{model_date}/ResNet18WithAuxiliary_epoch_172_best_loss_0.13_best_acc_98.09.pth'
# 클래스 이름 맵핑
# class_names = ['day/agood', 'day/block', 'day/bubble', 'day/mud', 'day/raindrop',
#                 'night/agood', 'night/block', 'night/bubble', 'night/mud', 'night/raindrop',
#                 'terminal/agood', 'terminal/block', 'terminal/bubble', 'terminal/mud', 'terminal/raindrop']
class_names = ['day/agood', 'day/agood_t', 'day/block', 'day/block_t', 'day/bubble', 'day/bubble_t', 'day/mud', 'day/mud_t', 'day/raindrop', 'day/raindrop_t',
                'night/agood', 'night/agood_t', 'night/block', 'night/block_t', 'night/bubble', 'night/bubble_t', 'night/mud', 'night/mud_t', 'night/raindrop', 'night/raindrop_t',
                'terminal/agood', 'terminal/agood_t', 'terminal/block', 'terminal/block_t', 'terminal/bubble', 'terminal/bubble_t', 'terminal/mud', 'terminal/mud_t', 'terminal/raindrop', 'terminal/raindrop_t']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def map_location(storage, loc):
    return storage.cuda(0)  # 모든 텐서를 cuda:0으로 매핑


#모델 로드
def load_model(model_path, num_classes):
    model = ResNet18WithAuxiliary(num_classes=num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location= device), strict=False)
    model = model.to(device)
    model.eval()  # 추론 모드로 전환
    return model


#이미지 전처리
def preprocess_image(image_path):
    transform = transforms.Compose([    #다양한 data augmenatation을 한꺼번에 손쉽개 해주는 기능
        transforms.Resize((224, 448)),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 데이터셋의 평균
                                std=[0.229, 0.224, 0.225])  # ImageNet 데이터셋의 표준편차

    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    
    return image





#inference and calculate probabilities using softmax
def predict(model, test_dir):
    probabilities = None
    
    # with torch.no_grad():  # 추론 시에는 Gradient 계산을 비활성화

    # 클래스별 폴더 경로 탐색 및 이미지 예측 수행
    
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)  #class_dir : test/dust, test/haze ...
        if not os.path.isdir(class_dir):
            print('no dir', class_dir)
            continue  # 해당 디렉토리가 없으면 넘어감
        
        print(class_dir) 
        #이미지 리스트 생성
        test_images = os.listdir(class_dir) #directory 내의 모든 이미지 파일 이름을 리스트로 저장
        
        
        
        
        ##GRADCAM 참고 : https://www.kaggle.com/code/antwerp/where-is-the-model-looking-for-gradcam-pytorch
        #이미지 경로 생성 및 예측 수행/image_name
        for i, img_name in enumerate(test_images):
            img_path = os.path.join(class_dir, img_name)    #class directory/imagename.png 형태로 묶어줌
            image = cv2.imread(img_path)
            background_image = (image - np.min(image)) / (np.max(image) - np.min(image))
            
            processed_image = preprocess_image(img_path).float()  # tensor([[[[-2.1179, -2.1179, ..., 
            input_tensor = processed_image.to(device)
            # processed_image = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).float()
            output = model(input_tensor)  # 예측 수행
            _, predicted_class = torch.max(output, 1)  # 텐서에서 최대값 구하기

            probabilities = softmax(output, dim=1)
            
            
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()    #tensor형태 인덱스
            #print(predicted_class_idx)
            predicted_class_name = class_names[predicted_class_idx]# 클래스 이름으로 변환
            
            prob_str = ", ".join([f"{class_names[i]}: {prob:.4f}" for i, prob in enumerate(probabilities[0])])  #probabilities and class format into str
            probability = ", ".join([f"{prob:.4f}" for i, prob in enumerate(probabilities[0])])
            blockage_prob = probabilities[0][2] + probabilities[0][3] + probabilities[0][4] + probabilities[0][5] + probabilities[0][6] + probabilities[0][7] + 
                            probabilities[0][12] + probabilities[0][13] + probabilities[0][14] + probabilities[0][15] + probabilities[0][16] + probabilities[0][17] + 
                            probabilities[0][22] + probabilities[0][23] + probabilities[0][24] + probabilities[0][25] + probabilities[0][26] + probabilities[0][27]            
            target_layers = [model.resnet.layer4[-1]]
            cam = GradCAM(model=model, target_layers = target_layers)
            
            grayscale_cam = cam(input_tensor = input_tensor)
            grayscale_cam = grayscale_cam[0, :]
            # print(grayscale_cam)#float (0~1)
            # print(background_image)#float (0~1)
            # print(grayscale_cam.shape)#float (0~1)
            # print(background_image.shape)#float (0~1)
            visualization = show_cam_on_image(background_image, grayscale_cam, use_rgb=True)
            
            text_pred = f"Prediction result : {predicted_class_name}"
            text_block = f"Blockage probability : {blockage_prob}"

            cv2.putText(visualization, text_pred, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(visualization, text_block, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            gradcam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

            processed_np = processed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            processed_np = (processed_np - processed_np.min()) / (processed_np.max() - processed_np.min())
            processed_np = (processed_np * 255).astype(np.uint8)
            processed_np = cv2.cvtColor(processed_np, cv2.COLOR_RGB2BGR)


            merged_image = cv2.hconcat([gradcam_image, processed_np])
            
            grad_img_name = os.path.splitext(img_name)


            save_dir = f'{test_dir}/gradcam/{model_date}'
            os.makedirs(save_dir, exist_ok=True)

            save_path = f'{save_dir}/{grad_img_name[0]}_cam.png'
            cv2.imwrite(save_path, merged_image)

            #Image.fromarray(visualization, 'RGB')
            
            # cv2.imwrite(f'./gradcam_result/{img_name}_cam.jpg', visualization)

            # #save the inference result in .CSV files
            # SAVE_PATH = '/mnt/d/BLTN_3.0/DLD-terminal/predict.csv'
            # if(class_names != 'terminal_agood'):
            #     class2 = 1
            # else:
            #     class2 = 0
            # if(class_name == predicted_class_name):
            #     class15 = 1
            # else:
            #     class15 = 0
                
            
            # new_data = ", ".join([f"{i + 1},{class_name},{img_name},{predicted_class_name},{class2},{class15},{probability}\n"])
            # print((new_data))   #<class 'str'>  1, terminal_dust, 20230316_071602_23.png, terminal_dust, 1, 1, 0.7057, 0.0000, 0.2943, 0.0000, 0.0000
            
            # f = open(SAVE_PATH, "a")
            # f. writelines(new_data)
    return probabilities[0], prob_str   #probabilities : 각 class의 확률리스트

    
    
    
    

#이미지를 수정해 텍스트 추가(overlay로 해도 되고,, )
def draw_predictions_on_image(image, probabilities):
    
    #for image_name in test_images: 외부에서 image_name 경로 다 지정해서 for 돌리니까 def내부에서 for 지정할 필요 업슴
    #while True: 
    #find the top probability class in prediction
    top_prob, top_class_idx = torch.max(probabilities, 0)   #top_class_idx : tensor형태
    top_class_name = class_names[top_class_idx] #'dust', 'haze', 'mud', 'raindrop', 'agood'
    
    #write the result of the prediction
    #text = f"Prediction : {top_class_name} ({top_prob:.4f})"
    #cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    #클래스별 확률을 표시
    y0, dy = 30, 25
    max_prob = max(probabilities)
    
    for i, prob in enumerate(probabilities):
        prob_str = f"{class_names[i]}: {prob:.4f}"
        y = y0 + i * dy
        
        if(prob == max_prob):
            cv2.putText(image, prob_str, (10, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)  #result should be yellow
            
        else:
            cv2.putText(image, prob_str, (10, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            
        
        #print(image_path)
      
#show the result of inference 
    # while True:
        
    #     cv2.imshow('Classification Result', image)
        
    #     key = cv2.waitKey(1000) & 0xFF
        
    #     if key == 27:  #esc 누르면 창 닫기
    #         cv2.destroyAllWindows()
    #         exit()

    #     cv2.destroyAllWindows()
    #     break  # 'while True' 루프를 빠져나와서 다음 이미지로 넘어갑니다
    
    #     #if cv2.waitKey(500) & 0xFF == ord('q'):
    #         #cv2.destroyAllWindows()    



if __name__ == '__main__':
    
    model = load_model(model_path, num_classes=len(class_names))        
    probabilities, prob_str = predict(model, test_dir)  #inference 수행

# # 전체 클래스에 대한 확률을 콘솔에 출력
#print(f"Image: {img_name} -> Predicted Class: {prob_str}")
#(f"Probabilities: {prob_list}")


    # draw_predictions_on_image(image, probabilities) #put the probabilities on the image
        
#        cv2.imshow("result_image", result_image)

        
#classified image가 show되게 출력 (class, probability 포함)





        

    
