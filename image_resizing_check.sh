#!/usr/bin/env bash

SRC_DIR='../../Dataset/00.scenario_v1/demo_251219'
RESIZE_DIR='../../Dataset/00.scenario_v1/resize/demo_251219'
DEST_DIR='../../Dataset/08_BLTN_GEN3/99_SCENARIO_SET_IMAGE_448x224'

# 1. 이미지 리사이즈
python3 image_resizing.py -i "$SRC_DIR" -o "$RESIZE_DIR"

# 2. RESIZE_DIR 안의 각 하위 파일/디렉토리 처리
for path in "$RESIZE_DIR"/*; do
    # RESIZE_DIR 안에 아무것도 없으면 패스
    [ -e "$path" ] || continue

    name="$(basename "$path")"
    dest_path="${DEST_DIR}/${name}"

    # DEST_DIR에 같은 이름이 이미 있으면 → 기존 걸 name_(1), name_(2)... 이런 식으로 변경
    if [ -e "$dest_path" ]; then
        i=1
        # 이미 name_(1)도 존재할 수 있으니, 비어 있는 번호를 찾기
        while [ -e "${DEST_DIR}/${name}_(${i})" ]; do
            i=$((i+1))
        done

        echo "기존 ${dest_path} 를 ${DEST_DIR}/${name}_(${i}) 로 이름 변경합니다."
        mv "$dest_path" "${DEST_DIR}/${name}_(${i})"
    fi

    # 이제 새 파일/디렉토리를 DEST_DIR로 이동
    echo "새 항목 ${path} -> ${DEST_DIR}/ 로 이동합니다."
    mv "$path" "$DEST_DIR/"
done

# 3. 체크리스트 실행
python3 scenario_checklist.py
