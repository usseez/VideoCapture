import cv2
import os
import numpy as np
import torch
from datetime import timedelta
import shutil
from pathlib import Path
from modify_resnet18 import ResNet18WithAuxiliary
import torch.nn as nn
import argparse
from loguru import logger
from torchvision import models, transforms
import torchvision.transforms as transforms
from PIL import Image

model_path = '/home/ubuntu/networkbuild/pre_train_weights/2025/2025-08-14-18-50-50/ResNet18WithAuxiliary_epoch_116_best_loss_0.13_best_acc_97.87.pth'
# INPUT_DIR = '/home/ubuntu/Dataset/08_BLTN_GEN3/22_OWD_250827'
# OUTPUT_DIR = '../../Dataset/00.scenario_video_classify/20250827_side,cloudy,front/'
num_classes = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_dir", default = '/home/ubuntu/Dataset/08_BLTN_GEN3/22_OWD_250827', type = str, help = 'input path'
    )
    parser.add_argument(
        "-o", "--output_dir", default = '../../Dataset/00.scenario_video_classify/20250827_side,cloudy,front/', type = str, help = 'output path'
    )
    return parser   



#load model 
def load_model(model_path, num_classes, device):
    model = ResNet18WithAuxiliary(num_classes=num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model



class VideoClassifierSplitter:
    def __init__(self, model, input_dir, output_dir, frame_interval=30):
        """
        Args:
            model: 15클래스 분류를 위한 딥러닝 모델
            input_dir: 입력 영상들이 있는 디렉토리
            output_dir: 분류된 영상들을 저장할 디렉토리
            frame_interval: 몇 프레임마다 분류를 수행할지 (기본값: 30프레임)
        """
        self.model = model
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        
        # 클래스 리스트
        self.classes = [
            'day_agood', 'day_block', 'day_bubble', 'day_mud', 'day_raindrop',
            'night_agood', 'night_block', 'night_bubble', 'night_mud', 'night_raindrop',
            'terminal_agood', 'terminal_block', 'terminal_bubble', 'terminal_mud', 'terminal_raindrop'
        ]
        
        # 출력 디렉토리 생성
        self.create_output_directories()
        #preprocess image
        self.transform = transforms.Compose([    #다양한 data augmenatation을 한꺼번에 손쉽개 해주는 기능
            
            transforms.Resize((224, 448)),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 데이터셋의 평균
                                std=[0.229, 0.224, 0.225])  # ImageNet 데이터셋의 표준편차

        ])

    def create_output_directories(self):
        """각 클래스별 출력 디렉토리 생성"""
        for class_name in self.classes:
            class_dir = self.output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # def preprocess_frame(self, frame):
    #     """
    #     프레임을 모델 입력에 맞게 전처리
    #     (실제 모델에 맞게 수정 필요)
    #     """
    #     # 예시: 224x224로 리사이즈하고 정규화
    #     frame_resized = cv2.resize(frame, (224, 448))
    #     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    #     frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
    #     # 배치 차원 추가 및 텐서로 변환
    #     frame_tensor = torch.FloatTensor(frame_normalized).permute(2, 0, 1).unsqueeze(0)
    #     return frame_tensor
    
    def predict_frame_class(self, frame):
        """프레임의 클래스를 예측"""
        with torch.no_grad():
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = self.transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                _, predicted = torch.max(outputs, 1)
                class_idx = predicted.item()
            
        return self.classes[class_idx]
    
    def format_time(self, seconds):
        """초를 mm:ss 형식으로 변환"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def find_class_changes(self, video_path):
        """영상에서 클래스가 바뀌는 지점들을 찾음"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        class_changes = []  # (frame_number, timestamp, class_name)
        current_class = None
        frame_count = 0
        
        print(f"Analyzing video: {video_path.name}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 지정된 간격마다만 분류 수행
            if frame_count % self.frame_interval == 0:
                frame_resized = cv2.resize(frame, (448,224))
                predicted_class = self.predict_frame_class(frame_resized)
                timestamp = frame_count / fps
                
                # 클래스가 바뀌었을 때
                if current_class is None:
                    current_class = predicted_class
                    class_changes.append((frame_count, timestamp, predicted_class))
                    print(f"Initial class: {predicted_class} at {self.format_time(timestamp)}")
                    
                elif predicted_class != current_class:
                    current_class = predicted_class
                    class_changes.append((frame_count, timestamp, predicted_class))
                    print(f"Class changed to: {predicted_class} at {self.format_time(timestamp)}")
            
            frame_count += 1
            
            # 진행률 표시
            if frame_count % (total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        return class_changes, fps
    
    def split_and_save_video(self, video_path, class_changes, fps):
        """클래스 변경 지점을 기준으로 영상을 분할하여 저장"""
        if not class_changes:
            print(f"No valid classes found in {video_path.name}")
            return
        # 각 클래스 등장 횟수 계산
        class_counts = {}
        for _, _, class_name in class_changes:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 상위 2클래스 찾기
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        top_classes = sorted_classes[:2]

        video_name = video_path.stem
        video_ext = video_path.suffix
        
        for rank, (class_name, count) in enumerate(top_classes, 1):
            output_filename = f"{video_name}{video_ext}"
            output_path = self.output_dir / class_name / output_filename

            try:
                shutil.copy2(video_path, output_path)
                print(f"Saved video (rank {rank}) : {class_name}/{output_filename} (count: {count})")
            except Exception as e:
                print(f"Error copying video to {class_name} {e}")
        print(f"Class distribution:{class_counts}")
    
    
    # def extract_video_segment(self, input_path, output_path, start_time, end_time=None):
    #     """FFmpeg를 사용하여 영상의 특정 구간을 추출"""
    #     import subprocess
        
    #     cmd = [
    #         'ffmpeg', '-y',  # -y: 파일 덮어쓰기
    #         '-i', str(input_path),
    #         '-ss', str(start_time),  # 시작 시간
    #     ]
        
    #     if end_time is not None:
    #         duration = end_time - start_time
    #         cmd.extend(['-t', str(duration)])  # 지속 시간
        
    #     cmd.extend([
    #         '-c', 'copy',  # 재인코딩 없이 복사 (빠름)
    #         str(output_path)
    #     ])
        
    #     try:
    #         subprocess.run(cmd, check=True, capture_output=True)
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error extracting video segment: {e}")
    #         # FFmpeg 실패시 OpenCV로 대체
    #         self.extract_video_segment_opencv(input_path, output_path, start_time, end_time)
    
    def extract_video_segment_opencv(self, input_path, output_path, start_time, end_time=None):
        """OpenCV를 사용하여 영상의 특정 구간을 추출 (백업 방법)"""
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 비디오 코덱 및 writer 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        # 시작 프레임으로 이동
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 종료 프레임 계산
        if end_time is not None:
            end_frame = int(end_time * fps)
        else:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            out.write(frame)
            current_frame += 1
        
        cap.release()
        out.release()
    
    def process_all_videos(self):
        """디렉토리 내의 모든 영상을 처리"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        video_files = [f for f in self.input_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in video_extensions]
        
        if not video_files:
            print("No video files found in the input directory.")
            return
        
        print(f"Found {len(video_files)} video files to process.")
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'='*50}")
            print(f"Processing video {i}/{len(video_files)}: {video_path.name}")
            print(f"{'='*50}")
            
            try:
                # 클래스 변경 지점 찾기
                class_changes, fps = self.find_class_changes(video_path)
                
                # 영상 분할 및 저장
                self.split_and_save_video(video_path, class_changes, fps)
                
            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
                continue
        
        print(f"\nAll videos processed! Results saved in: {self.output_dir}")

# 사용 예시
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    # 모델 로드 
    model = load_model(model_path, num_classes, device)
    
    # 영상 분류기 생성 및 실행
    classifier = VideoClassifierSplitter(
        model=model,
        input_dir = args.input_dir,  # 입력 영상 디렉토리
        output_dir = args.output_dir,  # 출력 디렉토리
        frame_interval=2  # 30프레임마다 분류 수행
    )
    
    # 모든 영상 처리
    classifier.process_all_videos()

if __name__ == "__main__":
    main()