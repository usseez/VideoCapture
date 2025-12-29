#using 15class modelfile, classify and capture video
#input : video folder, modelfile
#output : captured image based on model
import os
import cv2
import torch
import torchvision.transforms as transforms
from modify_resnet18 import ResNet18WithAuxiliary
from torchvision.models import resnet18, resnet34, resnet50, wide_resnet50_2, resnet101
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import numpy as np
from glob import glob
class_names = ['day_agood', 'day_block', 'day_bubble', 'day_mud', 'day_raindrop',
                'night_agood', 'night_block', 'night_bubble', 'night_mud', 'night_raindrop',
                'terminal_agood', 'terminal_block', 'terminal_bubble', 'terminal_mud', 'terminal_raindrop']

input_dir = '/home/ubuntu/Dataset/08_BLTN_GEN3/19_OWD_250716_Rain_Drive'
model_path = '/home/ubuntu/networkbuild/pre_train_weights/2025/2025-05-27-18-45-49_epoch156/ResNet18WithAuxiliary_epoch_156_best_loss_0.12_best_acc_97.63.pth'
output_dir = '../../Dataset/99.dataset_train/19_OWD_250716_Rain_Drive/'
num_classes = 15
threshold = 5.0


# try:
#     checkpoint = torch.load(model_path)
#     state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint['state_dict']
 
#     # 키 매핑
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         new_key = key.replace('original_conv1', 'resnet.conv1')  # original_conv1 -> resnet.conv1
#         new_state_dict[new_key] = value
 
#     # 모델에 state_dict 로드
#     model.load_state_dict(new_state_dict, strict=False)
#     # model.load_state_dict(torch.load(os.path.join(model_path, model_file), map_location=torch.device('cpu')))
#     print(f"모델이 성공적으로 로드되었습니다: {os.path.join(model_path, model_file)}\n")
# except Exception as e:
#     print(f"모델 로드 중 오류 발생: {e}\n")




#load model 
def load_model(model_path, num_classes, device):
    model = ResNet18WithAuxiliary(num_classes=num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model

#preprocess image
transform = transforms.Compose([    #다양한 data augmenatation을 한꺼번에 손쉽개 해주는 기능
    transforms.Resize((224, 448)),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 데이터셋의 평균
                        std=[0.229, 0.224, 0.225])  # ImageNet 데이터셋의 표준편차

])


#inference frame
def classify_frame(model, frame, device):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #PIL to numpy array
    tensor = transform(image).unsqueeze(0).to(device) # numpy to tensor
    with torch.no_grad():   #deactivate gradient tracking
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return class_names[pred], probs[0].cpu().numpy()    #probs : gpu상의 torch.tensor를 cpu로 이동하여(gpu에선 numpy로 바꿀 수 x) numpy배열로 변환(opencv로 그릴땐 numpy배열)



def create_info_panel(predicted_class, current_video, progress, probs, size = (480, 600)):
    #write video info
    h, w = size
    cell_w = 180
    cell_h = 40
    cols = 3
    panel = 255 * np.ones((h,w,3), dtype=np.uint8)
    y = 20
    cv2.putText(panel, '# Current Video:', (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
    y+=20
    cv2.putText(panel, os.path.basename(current_video), (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(50,50,50),1)
    y += 30

    #write progress info
    cv2.putText(panel, '# Progress:',(10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
    y+=20
    cv2.rectangle(panel,(10,y),(510, y+10), (200,200,200), -1)
    cv2.rectangle(panel,(10,y),(int(500 * progress)+10,y+10),(0,0,255),-1)  #progress toolbar
    y += 30

    #write predicted class
    cv2.putText(panel, '# Predicted Class:',(10,y), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),1)
    y+=20
    if predicted_class in class_names:
        text = f"->{predicted_class}"
        (text_width, text_height), baseline = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)  #text를 출력하기 위한 사각형 크기 저장
        cv2.rectangle(panel,(5,y-text_height),(5+text_width + 10, y + baseline), (128,128,128), -1)
        cv2.putText(panel, text, (10,y), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    
    else:
        cv2.putText(panel, f"{predicted_class}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(90,90,90),1)
    y += 40

    cv2.putText(panel, "# Confidence Matrix:",(10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
    y += 20

    
    for i, cls in enumerate(class_names):
        row = i // cols
        col = i % cols
        x = 10 + col * (cell_w + 10)
        row_y = y + row * (cell_h + 2)
        prob = probs[i] if i < len(probs) else 0.0
        gray = int(255 * (1 - prob))
        color = (gray, gray, gray)
        cv2.rectangle(panel, (x, row_y), (x + cell_w, row_y + cell_h), color, -1)
        text = f"{cls} : {prob:.4f}"
        text_color = (0,0,0) if gray > 128 else (255,255,255)
        cv2.putText(panel, text, (x + 5, row_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    return panel


def process_video(input_dir, model_path, output_dir, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, num_classes, device)
    video_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('_SL.mp4', '_SR.mp4', 'R.mp4'))])
    selected_video_idx = [0]
    
    cv2.namedWindow("Inference Viewer")

    while True:
        video_path = video_paths[selected_video_idx[0]]
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()    #첫번째 프레임 읽기

        if not ret:
            selected_video_idx[0] = (selected_video_idx[0] + 1)
            continue


        VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        paused = False

        #첫 번째 프레임 inference 및 저장
        prev_frame_resized = cv2.resize(prev_frame, (448,224))
        current_frame_number = 0
        last_label, probs = classify_frame(model, prev_frame_resized, device)
        prev_frame_prob = max(probs)
        img_number = 0
        if prev_frame_prob > 0.8:
            label_path = last_label.replace('_', '/')
            prev_save_path = os.path.join(output_dir, label_path, f"{os.path.basename(video_path)}_{current_frame_number:04d}.png")
            os.makedirs(os.path.dirname(prev_save_path), exist_ok = True)
            cv2.imwrite(prev_save_path, prev_frame_resized)
            img_number += 1


        while cap.isOpened():
            if not paused:
                ret,frame = cap.read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (448,224))
                label, probs = last_label, np.zeros(num_classes)
                diff = cv2.absdiff(cv2.cvtColor(prev_frame_resized,cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY))
                diff_percentage = (np.mean(diff) / 255.0) * 100.0
                
                if diff_percentage > threshold:
                    label, probs = classify_frame(model, frame_resized, device)
                    prob = max(probs)
                    if prob >= 0.8:
                        label_path = label.replace('_', '/')
                        save_path = os.path.join(output_dir, label_path, f"{os.path.basename(video_path)}_{img_number:04d}.png")
                        os.makedirs(os.path.dirname(save_path),exist_ok = True)
                        cv2.imwrite(save_path, frame_resized)
                        img_number+=1
                        
                        prev_frame_resized= frame_resized.copy()
                else:
                    label, probs = last_label, np.zeros(num_classes)

                progress = current_frame_number / total_frames
                panel = create_info_panel(label, video_path, progress, probs)
                display = np.hstack((cv2.resize(frame, (960,480)), panel))
                cv2.imshow("Inference Viewer", display)

            key = cv2.waitKey(0 if paused else 30) & 0xFF
            if key == ord(' '):
                paused = not paused
            elif key == ord('q') or key == 27:  #esc
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == 83:   # right key : go to next frame
                cap.release() 
                selected_video_idx[0] = (selected_video_idx[0] + 1)

            elif key == 81:
                cap.release() 
                selected_video_idx[0] = (selected_video_idx[0] - 1)

            current_frame_number += 1

        if ret == False and key != 81 and key != 83:
            selected_video_idx[0] = (selected_video_idx[0] + 1)








if __name__ == "__main__":
    process_video(input_dir, model_path, output_dir, threshold)