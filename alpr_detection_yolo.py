# -*- coding: utf-8 -*-
"""
============================================================
번호판 검출
------------------------------------------------------------
[실행 순서]
  1) pip install ultralytics
  2) 아래 "사용자 설정" 경로만 본인 환경에 맞게 수정
  3) python alpr_detection_yolo.py
============================================================
"""

import os
import shutil
import stat
import time
from pathlib import Path

from PIL import Image


# ============================================================
# 1. 설정
# ============================================================

# 원본 CCPD 이미지 폴더 (다음과 같은 경로로 사용)
# 이 폴더 안에 train / val / test 하위 폴더가 있어야 한다.
#   ccpd_green/
#     train/  *.jpg
#     val/    *.jpg
#     test/   *.jpg
# 주의: YOLO 라벨은 원본 이미지 크기 기준으로 정규화되므로 원본 폴더를 가리킨다.
IMAGE_DIR = r"C:\Users\USER\Downloads\CCPD2020\CCPD2020\ccpd_green"

# CCPD2020이 이미 나눠둔 train/val/test 하위 폴더 이름
# (혹시 폴더 이름이 다르면 여기만 바꾸면 된다. 예: "valid")
SPLIT_FOLDER_NAMES = {"train": "train", "val": "val", "test": "test"}

# YOLO 학습용 데이터셋이 만들어질 폴더
# 반드시 로컬 경로를 사용할 것.
DATASET_DIR = r"C:\cv_work\yolo_dataset"

# 데이터셋 재생성 여부
#   True  : 매번 폴더를 지우고 새로 만든다 (라벨 규칙을 바꿨을 때만 True)
#   False : 이미 만들어진 데이터셋이 있으면 그대로 재사용 (하이퍼파라미터
#           튜닝으로 학습만 다시 돌릴 때 수십 분 절약)
REBUILD_DATASET = False

# 클래스 이름 (번호판 1개 클래스)
CLASS_NAMES = ["plate"]

# 재현성을 위한 랜덤 시드
RANDOM_SEED = 42

# 빠른 테스트용: None이면 전체 사용, 정수면 그 개수만 사용
MAX_IMAGES = None

# 유효 이미지 확장자
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---- 학습 하이퍼파라미터 ----
PRETRAINED_WEIGHTS = "yolov8s.pt"
EPOCHS = 50
IMG_SIZE = 640        # 전처리에서 맞춘 입력 크기와 동일
BATCH_SIZE = 16       # GPU 메모리 부족하면 8 또는 4로 낮추기


def auto_device():
    """GPU가 있으면 0, 없으면 'cpu'를 자동 선택한다."""
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = auto_device()  # GPU 자동 감지 (없으면 CPU로 학습)
PROJECT_NAME = "alpr_detection"  # 결과 저장 폴더 이름


# ============================================================
# 2. CCPD 파일명 → YOLO 라벨 변환
# ============================================================

def get_image_size(image_path: str) -> tuple:
    """
    이미지의 (width, height)만 빠르게 읽는다.
    PIL은 헤더만 읽으므로 전체 픽셀을 디코딩하는 cv2보다 수십 배 빠르고,
    한글 경로도 문제없음.
    """
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def parse_ccpd_box_from_filename(image_path: str):
    """
    CCPD 파일명에서 번호판 bbox(픽셀 좌표)를 파싱한다.

    CCPD 파일명 패턴:
      area-tilt-bbox-corners-plate-brightness-blur.jpg
    bbox 부분 형식:
      x1&y1_x2&y2
    반환: (x1, y1, x2, y2)  또는  파싱 실패 시 None
    """
    file_stem = Path(image_path).stem
    parts = file_stem.split("-")
    if len(parts) < 3:
        return None

    bbox_part = parts[2]
    try:
        top_left, bottom_right = bbox_part.split("_")
        x1, y1 = map(int, top_left.split("&"))
        x2, y2 = map(int, bottom_right.split("&"))
    except ValueError:
        return None

    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def box_to_yolo_line(box, image_width: int, image_height: int):
    """
    픽셀 bbox (x1,y1,x2,y2) → YOLO 라벨 한 줄로 변환.
    YOLO 형식: class_id x_center y_center width height  (모두 0~1 정규화)

    [수정] 정규화 후 중심/크기를 따로 클리핑하면 박스 위치가 틀어지므로,
    픽셀 좌표를 먼저 이미지 경계로 클리핑한 뒤 변환한다.
    너비/높이 0 이하인 박스는 None을 반환해 건너뛰게 한다.
    """
    x1, y1, x2, y2 = box

    # 1) 픽셀 좌표를 이미지 경계 안으로 클리핑
    x1 = min(max(x1, 0), image_width)
    x2 = min(max(x2, 0), image_width)
    y1 = min(max(y1, 0), image_height)
    y2 = min(max(y2, 0), image_height)

    # 2) 불필요한 박스 제거
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None

    # 3) 정규화
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    # class_id = 0 (번호판 단일 클래스)
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# ============================================================
# 3. 데이터셋 수집 + 분할 + 폴더/라벨 생성
# ============================================================

def collect_images_in_folder(folder: str) -> list:
    """한 폴더(하위 폴더 포함)에서 이미지 경로를 모은다."""
    image_paths = []
    for root, _, files in os.walk(folder):
        for file_name in sorted(files):
            if Path(file_name).suffix.lower() in VALID_EXTENSIONS:
                image_paths.append(os.path.join(root, file_name))
    image_paths.sort()
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]
    return image_paths


def collect_splits_from_folders(image_dir: str) -> dict:
    """
    CCPD2020이 이미 나눠둔 train/val/test 폴더를 그대로 읽어 분할을 만든다.
    """
    splits = {}
    for split_name, folder_name in SPLIT_FOLDER_NAMES.items():
        folder = os.path.join(image_dir, folder_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"'{split_name}' 폴더를 찾을 수 없음: {folder}\n"
                f"→ IMAGE_DIR 안에 train/val/test 폴더가 있는지, "
                f"폴더 이름이 SPLIT_FOLDER_NAMES와 같은지 확인하세요."
            )
        splits[split_name] = collect_images_in_folder(folder)
    return splits


def build_yolo_dataset(splits: dict, dataset_dir: str) -> None:
    """
    Ultralytics가 요구하는 폴더 구조를 만든다:
      dataset_dir/
        images/train, images/val, images/test
        labels/train, labels/val, labels/test
    각 이미지를 복사하고, 같은 이름의 .txt 라벨을 생성한다.
    """
    # 폴더 생성
    for split_name in splits:
        os.makedirs(os.path.join(dataset_dir, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split_name), exist_ok=True)

    skipped = 0
    for split_name, paths in splits.items():
        for idx, image_path in enumerate(paths):
            box = parse_ccpd_box_from_filename(image_path)
            if box is None:
                skipped += 1
                continue

            # 라벨 정규화를 위해 원본 이미지 크기만 읽는다 (헤더만 읽어서 빠름)
            w, h = get_image_size(image_path)
            yolo_line = box_to_yolo_line(box, w, h)
            if yolo_line is None:  # 크기 0인 박스는 건너뛴다
                skipped += 1
                continue

            # 파일명 충돌 방지를 위해 인덱스를 붙인 stub 사용
            stub = f"{split_name}_{idx:06d}"

            # 이미지 복사
            # [수정] cv2 디코딩→재인코딩 방식은 JPEG 화질 손실 + 매우 느림.
            #        단순 파일 복사가 원본을 그대로 보존하고 수십 배 빠름.
            suffix = Path(image_path).suffix.lower()
            dst_image = os.path.join(dataset_dir, "images", split_name, stub + suffix)
            shutil.copy2(image_path, dst_image)

            # 라벨 저장
            dst_label = os.path.join(dataset_dir, "labels", split_name, stub + ".txt")
            with open(dst_label, "w", encoding="utf-8") as f:
                f.write(yolo_line + "\n")

        print(f"  - {split_name}: {len(paths)}장 처리")

    if skipped:
        print(f"  (bbox 파싱 실패로 건너뛴 이미지: {skipped}장)")


def remove_dir_safely(path: str, retries: int = 3, wait_seconds: float = 2.0) -> None:
    """
    폴더를 안전하게 삭제한다.

    * OneDrive 동기화나 탐색기가 파일을 잠그고 있으면
    shutil.rmtree가 PermissionError를 냄
    → 읽기 전용 속성을 해제하고, 잠시 기다렸다가 재시도
    """

    def _handle_readonly(func, p, _exc_info):
        os.chmod(p, stat.S_IWRITE)  # 읽기 전용 해제 후 재시도
        func(p)

    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path, onerror=_handle_readonly)
            return
        except PermissionError:
            if attempt == retries:
                raise PermissionError(
                    f"폴더 삭제 실패: {path}\n"
                    "→ 다른 프로그램이 이 폴더를 사용 중입니다.\n"
                    "  1) 탐색기에서 이 폴더를 열어뒀다면 닫기\n"
                    "  2) OneDrive 안의 경로라면 DATASET_DIR을 로컬 경로"
                    r"(예: C:\cv_work\yolo_dataset)로 변경"
                )
            print(f"  폴더가 사용 중... {wait_seconds}초 후 재시도 ({attempt}/{retries})")
            time.sleep(wait_seconds)


def write_data_yaml(dataset_dir: str) -> str:
    """Ultralytics 학습에 필요한 data.yaml 파일을 만든다."""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    # 경로 구분자는 yaml에서 슬래시로 통일하는 게 안전
    base = dataset_dir.replace("\\", "/")
    names_str = ", ".join(f"'{n}'" for n in CLASS_NAMES)

    content = (
        f"path: {base}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names_str}]\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return yaml_path


# ============================================================
# 4. YOLOv8 학습 / 평가 / 예측
# ============================================================

def train_and_evaluate(data_yaml_path: str) -> None:
    """
    pretrained YOLOv8로 학습하고, val/test에서 mAP를 평가한다.
    ultralytics는 학습 중 best.pt를 자동 저장한다.
    """
    # ultralytics는 필요 시점에만 import (전처리/라벨 단계에선 불필요)
    from ultralytics import YOLO

    # 1) pretrained 가중치 로드 (전이학습 시작점)
    model = YOLO(PRETRAINED_WEIGHTS)

    # 2) 학습
    print("\n[학습 시작] YOLOv8 fine-tuning ...")
    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name="train",
        patience=15,        # 15 epoch 동안 개선 없으면 조기 종료
        seed=RANDOM_SEED,
    )

    # [수정] 학습 직후 model 객체는 버전에 따라 '마지막 epoch' 가중치일 수
    # 있으므로, 가장 성능 좋았던 best.pt를 명시적으로 로드해서 평가한다.
    best_weights = str(Path(results.save_dir) / "weights" / "best.pt")
    best_model = YOLO(best_weights)

    # 3) 검증셋 평가 (mAP50, mAP50-95)
    # data 인자를 명시해야 버전에 따라 데이터셋을 못 찾는 문제가 생기지 않음.
    print("\n[검증셋 평가]")
    val_metrics = best_model.val(data=data_yaml_path, split="val", device=DEVICE)
    print(f"  val mAP50    : {val_metrics.box.map50:.4f}")
    print(f"  val mAP50-95 : {val_metrics.box.map:.4f}")

    # 4) 테스트셋 평가 (최종 성능 — 결과 종합할 때 전달)
    print("\n[테스트셋 평가]")
    test_metrics = best_model.val(data=data_yaml_path, split="test", device=DEVICE)
    print(f"  test mAP50    : {test_metrics.box.map50:.4f}")
    print(f"  test mAP50-95 : {test_metrics.box.map:.4f}")
    print(f"  test precision: {test_metrics.box.mp:.4f}")
    print(f"  test recall   : {test_metrics.box.mr:.4f}")

    print("\n[완료] 학습된 가중치(best.pt) 위치:")
    print(f"  {best_weights}")


def predict_sample(weights_path: str, sample_image_path: str) -> None:
    """
    학습된 모델로 한 장 예측해보는 예시 (팀원 3의 문자인식 단계로 crop 넘길 때 활용).
    필요할 때만 직접 호출.
    """
    from ultralytics import YOLO
    model = YOLO(weights_path)
    results = model.predict(source=sample_image_path, imgsz=IMG_SIZE, device=DEVICE, save=True)
    for r in results:
        print(f"검출된 번호판 개수: {len(r.boxes)}")
        print(f"결과 이미지 저장 위치: {r.save_dir}")


# ============================================================
# 5. 메인 파이프라인
# ============================================================

def main() -> None:
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없음: {IMAGE_DIR}")

    print("=" * 60)
    print("STEP 1~2) CCPD2020 원본 train/val/test 폴더 읽기")
    splits = collect_splits_from_folders(IMAGE_DIR)
    total = sum(len(v) for v in splits.values())
    if total == 0:
        raise ValueError(f"이미지가 없음: {IMAGE_DIR}")
    for k, v in splits.items():
        print(f"  {k}: {len(v)}장")
    print(f"  총 {total}장")

    print("STEP 3) YOLO 라벨 생성 + 데이터셋 폴더 구성")
    if os.path.exists(DATASET_DIR) and not REBUILD_DATASET:
        # 이미 만들어둔 데이터셋 재사용 (학습만 다시 돌릴 때 시간 절약)
        print(f"  기존 데이터셋 재사용: {DATASET_DIR}")
    else:
        if os.path.exists(DATASET_DIR):
            remove_dir_safely(DATASET_DIR)
        build_yolo_dataset(splits, DATASET_DIR)

    print("STEP 4) data.yaml 생성")
    data_yaml_path = write_data_yaml(DATASET_DIR)
    print(f"  {data_yaml_path}")

    print("STEP 5) YOLOv8 학습 + 평가")
    train_and_evaluate(data_yaml_path)

    print("\n완료!")


if __name__ == "__main__":
    main()
