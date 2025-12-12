import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import argparse
import numpy as np

# 같은 폴더에 있는 classifier_model.py에서 모델 구조 가져오기
from model import get_animal_model

# --- [설정] ---
# 학습된 모델 경로 (Docker 내부 경로)
MODEL_PATH = '/app/models/Animals-10/classifier/animals10_resnet18.pth'

# 클래스 정의 (Animals-10)
CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
           'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# 하이퍼파라미터
NUM_MC_SAMPLES = 30  # 불확실성 계산을 위한 반복 횟수
ENTROPY_THRESHOLD = 0.6  # OOD 판단 기준값
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 전처리 (학습 때와 동일) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model():
    """학습된 모델을 로드합니다."""
    # pretrained=False: 구조만 가져오고 가중치는 내가 학습한 것을 씀
    model = get_animal_model(num_classes=len(CLASSES), pretrained=False)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # 가중치 덮어쓰기
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    return model


def enable_dropout(model):
    """추론(Eval) 중에도 Dropout을 켜서 불확실성을 계산할 수 있게 함"""
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def predict_image(model, image_path):
    # 1. 이미지 로드
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: Cannot open image '{image_path}'. ({e})")
        return

    # 2. 전처리 및 배치 차원 추가 (3, 224, 224) -> (1, 3, 224, 224)
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 3. MC Dropout 활성화
    enable_dropout(model)

    # 4. 반복 추론 (MC Sampling)
    mc_outputs = []
    with torch.no_grad():
        for _ in range(NUM_MC_SAMPLES):
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            mc_outputs.append(probs.cpu().numpy())

    # 5. 결과 계산
    # (30, 1, 10) -> (1, 10) 평균 확률
    mc_probs = np.vstack(mc_outputs)
    mean_prob = np.mean(mc_probs, axis=0)

    # Entropy (불확실성) 계산
    epsilon = 1e-12
    entropy = -np.sum(mean_prob * np.log(mean_prob + epsilon))

    # 가장 높은 확률의 클래스 찾기
    pred_idx = np.argmax(mean_prob)
    pred_class = CLASSES[pred_idx]
    confidence = mean_prob[pred_idx]

    # 6. OOD 판정
    is_ood = entropy > ENTROPY_THRESHOLD
    result_str = "OOD (Pokemon/Unknown)" if is_ood else f"ID ({pred_class})"

    # 7. 결과 출력
    print("-" * 50)
    print(f"📂 Image      : {os.path.basename(image_path)}")
    print(f"📊 Entropy    : {entropy:.4f} (Threshold: {ENTROPY_THRESHOLD})")
    print(f"🏷️ Prediction : {pred_class} ({confidence * 100:.1f}%)")
    print(f"🎯 Result     : {result_str}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect OOD from a single image")
    parser.add_argument('--image', type=str, required=True, help="Path to the image file")
    args = parser.parse_args()

    # 모델 로드 및 추론 실행
    model = load_model()
    predict_image(model, args.image)