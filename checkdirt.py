from pathlib import Path
import sys

def count_files_in_target_folders(root_dir: str) -> int:
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"경로가 존재하지 않습니다: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"폴더 경로가 아닙니다: {root}")

    total = 0
    suffixes = ("_Agood", "_Raindrop")

    # root 아래 모든 "폴더"를 재귀적으로 훑음 (root 자신도 포함)
    for d in [root, *root.rglob("*")]:
        if not d.is_dir():
            continue

        if d.name.endswith(suffixes):
            # 해당 폴더 "바로 아래"의 파일만 카운트 (서브폴더 제외)
            count_here = sum(1 for p in d.iterdir() if p.is_file())
            total += count_here

    return total


if __name__ == "__main__":
    # 사용법: python script.py /path/to/root
    # 인자를 안 주면 현재 폴더를 기준으로 동작
    root_path = sys.argv[1] if len(sys.argv) > 1 else "."
    total_files = count_files_in_target_folders(root_path)
    print('clean:', total_files)
    print('block:', total_files)