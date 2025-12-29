"""root dir 안의 각 sub디렉토리를 순회하면서 디렉토리 내 .png파일 중 10번째 이미지를 show해줘
내가 이미지를 보고, 이상이 있으면, key를 누를거야(e) 해당 sub디렉토리 이름 앞에 99_를 붙여줘
그리고 방향키로 다음 sub디렉토리로 넘어가게 해줘
방향키로 이전 sub디렉토리로도 돌아갈 수 있게 해줘
"""

import os
import cv2
from glob import glob
import shutil

# ➜ 여기만 바꿔서 사용하세요
ROOT_DIR = "../../Dataset/08_BLTN_GEN3/99_SCENARIO_SET_IMAGE_448x224_ERROR"  # 루트 디렉토리 경로
ERROR_DIR_NAME = "00.error"

def get_subdirs(root):
    """root 바로 아래 sub 디렉토리 목록 반환 (이름 순 정렬)"""
    subdirs = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith('00')
    ]
    subdirs.sort()
    return subdirs

def get_10th_png(subdir_path):
    """sub 디렉토리 내 .png 목록 중 10번째 파일 경로 반환 (0-based index 9)"""
    pngs = sorted(glob(os.path.join(subdir_path, "*.png")))
    if len(pngs) < 10:
        return None
    return pngs[9]  # 10번째(인덱스 9)

def ensure_error_dir(root):
    """ROOT_DIR 아래 ./error/ 디렉토리 생성 (없으면)"""
    error_dir = os.path.join(root, ERROR_DIR_NAME)
    if not os.path.exists(error_dir):
        os.makedirs(error_dir, exist_ok=True)
    return error_dir

def move_to_error(root, subdir_name):
    """
    현재 sub 디렉토리를 ROOT_DIR/error/ 밑으로 이동.
    이미 error 디렉토리 안이면 이동하지 않음.
    """
    error_root = ensure_error_dir(root)
    src_path = os.path.join(root, subdir_name)

    # 이미 error 안으로 이동된 경우 (방어적 체크)
    if os.path.commonpath([src_path, error_root]) == os.path.abspath(error_root):
        print(f"[INFO] 이미 error 디렉토리 안에 있는 것 같습니다: {src_path}")
        return False

    dst_path = os.path.join(error_root, subdir_name)

    if os.path.exists(dst_path):
        print(f"[WARN] error 디렉토리에 같은 이름의 디렉토리가 이미 있습니다: {dst_path}")
        print("       이동을 건너뜁니다.")
        return False

    try:
        shutil.move(src_path, dst_path)
        print(f"[MOVE] {src_path}  ->  {dst_path}")
        return True
    except Exception as e:
        print(f"[ERROR] 이동 중 오류 발생: {e}")
        return False

def show_image(image_path, window_name="preview"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 이미지를 열 수 없습니다: {image_path}")
        return
    cv2.imshow(window_name, img)

def main():
    subdirs = get_subdirs(ROOT_DIR)
    if not subdirs:
        print("[ERROR] sub 디렉토리가 없습니다.")
        return

    print("조작 방법:")
    print("  → (오른쪽 방향키) : 다음 sub 디렉토리")
    print("  ← (왼쪽 방향키)   : 이전 sub 디렉토리")
    print("  e : 현재 sub 디렉토리를 ./error/ 로 이동")
    print("  ESC : 종료")
    print()

    index = 0
    window_name = "10th PNG preview"

    while True:
        if not subdirs:
            print("[INFO] 더 이상 남은 sub 디렉토리가 없습니다. 종료합니다.")
            break

        if index < 0:
            index = 0
        if index >= len(subdirs):
            index = len(subdirs) - 1

        current_name = subdirs[index]
        current_path = os.path.join(ROOT_DIR, current_name)

        cv2.destroyAllWindows()
        png_path = get_10th_png(current_path)

        print("=" * 60)
        print(f"[{index+1}/{len(subdirs)}] {current_name}")
        if png_path is None:
            print("  -> .png 파일이 10개 미만이라 10번째 이미지를 찾을 수 없습니다.")
        else:
            print(f"  -> 10번째 이미지: {os.path.basename(png_path)}")
            show_image(png_path, window_name=window_name)

        # 키 입력 대기
        key_raw = cv2.waitKey(0)
        key = key_raw & 0xFFFFFFFF  # raw 코드
        key_8 = key_raw & 0xFF      # 8bit 마스크

        # 방향키(환경마다 코드가 조금 다를 수 있어서 여러 값 허용)
        LEFT_KEYS = {81, 2424832}
        RIGHT_KEYS = {83, 2555904}

        if key_8 == 27:  # ESC
            print("[EXIT] ESC 입력으로 종료합니다.")
            break

        elif key_8 in (ord('e'), ord('E')):
            # 현재 디렉토리를 ./error/ 로 이동
            moved = move_to_error(ROOT_DIR, current_name)
            if moved:
                # 리스트에서 제거
                del subdirs[index]
                # 현재 index는 그대로 두되, 범위 조정
                if index >= len(subdirs):
                    index = len(subdirs) - 1
                # 바로 다음/이전 디렉토리로 넘어가도록 루프 계속
                continue

        elif key in LEFT_KEYS:
            # 이전 sub 디렉토리
            if index > 0:
                index -= 1
            else:
                print("[INFO] 이미 첫 번째 디렉토리입니다.")

        elif key in RIGHT_KEYS:
            # 다음 sub 디렉토리
            if index < len(subdirs) - 1:
                index += 1
            else:
                print("[INFO] 이미 마지막 디렉토리입니다.")

        # 혹시 방향키가 안 먹히면 a/d 키도 함께 지원 (백업용)
        elif key_8 in (ord('a'), ord('A')):
            if index > 0:
                index -= 1
            else:
                print("[INFO] 이미 첫 번째 디렉토리입니다.")
        elif key_8 in (ord('d'), ord('D')):
            if index < len(subdirs) - 1:
                index += 1
            else:
                print("[INFO] 이미 마지막 디렉토리입니다.")

        else:
            print(f"[INFO] 인식하지 못한 키 코드: {key_raw} (현재 디렉토리 유지)")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()