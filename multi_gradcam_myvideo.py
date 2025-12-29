#gradcam for 6class
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
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import List, Dict
import numpy as np
from loguru import logger
import time
import torchvision
import argparse
from modify_resnet18 import ResNet18WithAuxiliary



def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--class_num", default=15, type=int, help="number of classes"
    )
    parser.add_argument(
        "-i", "--input_video", default='/home/ubuntu/Dataset/08_BLTN_GEN3/test', type=str, help="input path"
    )
    parser.add_argument(
        "-m", "--model", default='../pre_train_weights/weights/2025-08-14-18-50-50/ResNet18WithAuxiliary_epoch_172_best_loss_0.13_best_acc_98.09.pth', type=str, help="model path"
    )
    parser.add_argument(
        "-s", "--view_class", default = 1, type = int, help = "number of classes that want to watch" #block, clean, black
    )
    return parser

args = make_parser().parse_args()
logger.info("args value: {}".format(args))

# 클래스 이름 맵핑
if (args.class_num == 2):
    class_names = ['clean', 'blockage']
    
elif (args.class_num == 6):
    class_names = ['day_clean', 'day_block', 'night_clean', 'night_block', 'terminal_clean', 'terminal_block']
    
elif (args.class_num == 15):
    class_names = ['day_agood', 'day_block', 'day_bubble', 'day_mud', 'day_raindrop',
                   'terminal_agood', 'terminal_block', 'terminal_bubble','terminal_mud', 'terminal_raindrop',
                'night_agood', 'night_block', 'night_bubble', 'night_mud', 'night_raindrop']
    
elif (args.class_num == 16):
    class_names = ['day_agood', 'day_block', 'day_bubble', 'day_mud', 'day_raindrop',
                'night_agood', 'night_block', 'night_bubble', 'night_mud', 'night_raindrop',
                'terminal_agood', 'terminal_block', 'terminal_bubble','terminal_mud', 'terminal_raindrop',
                'low_brightness']
else:
    raise ValueError(f"Unsupported class number: {args.class_num}")


def map_location(storage, loc):
    return storage.cuda(0)  # 모든 텐서를 cuda:0으로 매핑

def load_model(model_path, class_num):
    try:
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model architecture
        model = ResNet18WithAuxiliary(num_classes=class_num, darkness_threshold = -1.9)

        model.fc = nn.Linear(model.fc.in_features, class_num)
        
        # Load model weights
        try:
            # model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
            model.load_state_dict(torch.load(model_path), strict = False)
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Move model to appropriate device
        
        logger.info(f"Using device: {device}")
        model = model.to(device)
        
        # Verify model architecture
        logger.info(f"Model loaded with {class_num} output classes")
        
        # Simple test forward pass to verify model works
        dummy_input = torch.randn(1, 3, 224, 448).to(device)
        try:
            with torch.no_grad():
                _ = model(dummy_input)
            logger.info("Model test forward pass successful")
        except Exception as e:
            raise RuntimeError(f"Model forward pass failed: {str(e)}")
            
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        sys.exit(1)  # Exit with error code

def preprocess_image(frame):
    transform = transforms.Compose([    #다양한 data augmenatation을 한꺼번에 손쉽개 해주는 기능
        transforms.Resize((224, 448)),      #2차원이니까 괄호 2개 써중기...
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 데이터셋의 평균
                                std=[0.229, 0.224, 0.225])  # ImageNet 데이터셋의 표준편차

    ])
    
    pil_image=Image.fromarray(frame)
    image = transform(pil_image)
    image = image.unsqueeze(0)
    return image

#inference and calculate probabilities using softmax
def predict(model, test_dir):

    prev_time = 0
    frameRate = 1
    new_size = (448, 224)
    video_names = -1
    combined_probs = [0.0, 0.0]
    class_names2 = ['clean', 'block']
    # test video directory
    sub_dirs = [d for d in os.listdir(args.input_video) if os.path.isdir(os.path.join(test_dir, d))] #sub_dir : ['aslan', 'OWD']
    if not sub_dirs:
        print("No subdirectories found in:", test_dir)
        return None, ""
    
    current_dir_index = 0
    current_video_index = 0
    is_playing = True  # play/pause status
    sub_dir = sub_dirs[current_dir_index]
    video_dir = os.path.join(args.input_video, sub_dir)     #/dataset_video/aslan, /dataset_video/OWD
    video_names = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    while(current_dir_index < len(sub_dirs)):        #test내부 sub 디렉토리 접근
        if not video_names :
            current_dir_index += 1
            continue 
        
        while current_video_index < len(video_names):    
            # if('R' in img_name or img_name[0] == '4' or img_name == '3') : 
            # if('.mp4' in video_name) : 
            
            #get the video path
            video_name = video_names[current_video_index]
            video_path = os.path.join(video_dir, video_name)    #class directory/imagename.mp4 형태로 묶어줌
 
            video = cv2.VideoCapture(video_path)
            print(f"Playing:{sub_dir}/{video_name}, ({current_video_index+1}/{len(video_names)})")
            

            while True:
                if is_playing:
                    retval, frame = video.read()
                    if not(retval):
                        break
                    
                    #image전처리
                    background_image = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
                    processed_image = preprocess_image(frame)  # tensor([[[[-2.1179, -2.1179, ..., 
                    processed_image = processed_image.to(device) #cuda


                    B = processed_image[:, 0, :, :] * 0.29899999499320984
                    G = processed_image[:, 1, :, :] * 0.5870000123977661
                    R = processed_image[:, 2, :, :] * 0.11400000005960464
                    Y = B + G + R

                    Y_mean = Y.mean()

                    # inference 수행
                    
                    output = model(processed_image)  
                    _, predicted_class = torch.max(output, 1)  # 텐서에서 최대값 구하기
                    probabilities = softmax(output, dim=1)  #softmax 적용 tensor([[0.0661, 0.0077, 0.3107, 0.0036, 0.5994, 0.0125]], grad_fn=<SoftmaxBackward0>) 


                    predicted_class_idx = torch.argmax(probabilities, dim=1).item()    #predicted class의 인덱스
                    predicted_class_name = class_names[predicted_class_idx]# 클래스 이름으로 변환

                    prob_str = ", ".join([f"{class_names[i]}: {prob:.4f}" for i, prob in enumerate(probabilities[0])])  #probabilities and class format into str
                    probability = ", ".join([f"{prob:.4f}" for i, prob in enumerate(probabilities[0])])
        
                    #gradcam 생성
                    target_layers = [model.resnet.layer4[-1]]
                    cam = GradCAM(model=model, target_layers = target_layers)
                    grayscale_cam = cam(input_tensor = processed_image)
                    grayscale_cam = grayscale_cam[0, :]

                    #영상 resize
                    background_image = cv2.resize(background_image, new_size)
                    original_image = cv2.resize(frame, new_size)
                    grayscale_cam = cv2.resize(grayscale_cam, new_size)
                    
                    #영상 overlay
                    visualization = show_cam_on_image(background_image, grayscale_cam, use_rgb=True)
                    visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
                    
                    
                    #클래스별 확률을 표시
                    #6class로 표시
                    
                    y0, dy = 30, 25
                    text_color_yellow = (0, 255, 255)    
                    text_color_white = (255, 255, 255) 
                    if args.view_class == 6: 
                        for i, prob in enumerate(probabilities):
                            for j in range(args.class_num):
                                y = y0 + j * dy
                                if(prob[j] == prob[predicted_class_idx]):
                                    text_color = (0, 255, 255)    
                                else:
                                    text_color = (255, 255, 255) 
                                class1 = f"{class_names[j]}, {prob[j]:.4f}"
                                cv2.putText(original_image, class1, (10, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color, 1)
                    # 2class로 표시
                    elif args.view_class == 2: 
                        combined_probs[0] = probabilities[0][0].item() + probabilities[0][2].item() + probabilities[0][4].item()
                        combined_probs[1] = probabilities[0][1].item() + probabilities[0][3].item() + probabilities[0][5].item()
                                
                        for i in range(2):  # 2개의 클래스 그룹만 표시
                            y = y0 + i * dy
                            text_color = (255, 255, 255)
                            
                            # 가장 높은 확률을 가진 클래스 그룹을 하이라이트
                            if i == (0 if combined_probs[0] > combined_probs[1] else 1):
                                text_color = (0, 255, 255)
                                
                            class1 = f"{class_names2[i]}: {combined_probs[i]:.4f}"    
                            cv2.putText(original_image, class1, (10, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color, 1)
                    #3class : classified class
                    else: 
                        # combined_probs[0] = probabilities[0][0].item() + probabilities[0][2].item() + probabilities[0][4].item()
                        
                        # for i in range(2):  # 2개의 클래스 그룹만 표시
                        #     y = y0 + i * dy
                        #     text_color = (255, 255, 255)
                            
                        #     # 가장 높은 확률을 가진 클래스 그룹을 하이라이트
                        #     if i == (0 if combined_probs[0] > combined_probs[1] else 1):
                        #         text_color = (0, 255, 255)
                        y = y0
                        text_color_yellow = (0, 255, 255)    
                        text_color_white = (255, 255, 255) 

                        #top3 class index
                        topk = torch.topk(probabilities[0], k = 3)

                        topk_probs = topk.values.detach().cpu().numpy() #torch의 tensor가 requires_grad=True일땐 .numpy를 직접호출할 수 없어 detach를 먼저 해야함
                        topk_indices = topk.indices.detach().cpu().numpy()

                        for rank, (idx, prob) in enumerate(zip(topk_indices, topk_probs)):
                            text_color = text_color_yellow if rank == 0 else text_color_white
                            class_text = f"{class_names[idx]}: {prob:.4f}"  
                            y_text = f"y : {Y_mean}"
                            cv2.putText(original_image, class_text, (10, y0 + rank * dy), cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color, 1)

                        

                        
                        
                    # 영상 출력
                    FPS = 100
                    current_time = time.time() - prev_time
                    if current_time > 0.01/ FPS:
                        prev_time = time.time()
                        result_video = np.concatenate((visualization, original_image), axis = 1)
                        cv2.imshow('result_video', result_video)

                # 키 입력 처리
                key = cv2.waitKey(frameRate)
                
                if key == 27:   #esc-> terminate the video
                    video.release()
                    cv2.destroyAllWindows()
                    return probabilities[0], ""
                
                elif key == 32:  # 스페이스바 - play/pause
                    is_playing = not is_playing
                    
                elif key == 81 :  # left key
                    video.release()
                    if current_video_index > 0:# same directory
                        current_video_index = current_video_index - 1
                    else:        #go to previous directory
                        if current_dir_index > 0:
                            current_dir_index -= 1
                            sub_dir = sub_dirs[current_dir_index]  # 현재 디렉토리 변수 업데이트
                            video_dir = os.path.join(args.input_video, sub_dir)  # 경로 업데이트
                            video_names = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
                            current_video_index = len(video_names) - 1 if video_dir else 0
                    break
                        
                elif key == 83:  # right key
                    video.release()          
                    if current_video_index < len(video_names) - 1:
                        current_video_index = current_video_index + 1
                    else:
                        if current_dir_index < len(sub_dirs) - 1:#len(sub_dirs) : 2
                            current_dir_index += 1
                            sub_dir = sub_dirs[current_dir_index]  # 현재 디렉토리 변수 업데이트
                            video_dir = os.path.join(args.input_video, sub_dir)  # 경로 업데이트
                            video_names = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
                            current_video_index = 0          
                    
                    break
            if key == 27:  # ESC - 종료
                return probabilities[0], ""   
            # 비디오 끝까지 재생되고 방향키를 누르지 않은 경우 다음 비디오로
            if retval == False and key != 81 and key != 83 and key != 2424832 and key != 2555904:
                # 비디오 끝까지 재생되고 방향키를 누르지 않은 경우 다음 비디오로
                current_video_index += 1

    return probabilities[0], prob_str    #probabilities : 각 class의 확률리스트


if __name__ == '__main__':
    # 테스트 이미지 경로
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = make_parser().parse_args()
    logger.info(f"Arguments: {args}")
    model = load_model(args.model, class_num=len(class_names))        
    probabilities, prob_str = predict(model, args.input_video)  #inference 수행








        

    
