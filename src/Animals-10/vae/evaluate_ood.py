import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from model import BayesianVAE
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- [설정] ---
# 학습된 모델 경로
MODEL_PATH = '/app/models/Animals-10/vae/vae_final.pth'

ID_DATA_DIR = '/app/data/animals'
OOD_DATA_DIR = '/app/data/pokemon'

# [핵심] 결과 저장 경로 (Animals-10/vae)
BASE_RESULT_DIR = '/app/results/Animals-10/vae'

# VAE 이상치 판단 임계값 (상황에 맞춰 조정)
ANOMALY_THRESHOLD = 0.025
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 실행 폴더 관리 ---
def get_next_run_dir(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    i = 1
    while True:
        run_dir = os.path.join(base_dir, f"run_{i}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            print(f">>> Created new result directory: {run_dir}")
            return run_dir, i
        i += 1


# --- 커스텀 데이터셋 (경로 반환용) ---
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        filename = os.path.basename(path)
        return original[0], path, filename


class OODSystem:
    def __init__(self, model_path):
        self.device = DEVICE
        self.model = BayesianVAE().to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        # 키 이름 정리 (_orig_mod 제거)
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state)
        self.model.eval()

    def detect(self, img_tensor, samples=30):
        # [H100 Optimization] 배치 복제
        batch = img_tensor.repeat(samples, 1, 1, 1).to(self.device)
        with torch.no_grad():
            recon_batch, _, _ = self.model(batch)

        target = img_tensor.to(self.device).squeeze(0)

        # Score = MSE + Variance
        recon_error = F.mse_loss(recon_batch.mean(0), target).item()
        uncertainty = torch.var(recon_batch, dim=0).mean().item()

        return recon_error + uncertainty


def run():
    # 1. 실행 폴더 생성 (run_X)
    run_dir, run_id = get_next_run_dir(BASE_RESULT_DIR)

    # 2. 하위 폴더 생성
    img_save_dir = os.path.join(run_dir, 'sorted_images')
    path_ood = os.path.join(img_save_dir, 'Predicted_OOD')
    path_id = os.path.join(img_save_dir, 'Predicted_ID')
    os.makedirs(path_ood, exist_ok=True)
    os.makedirs(path_id, exist_ok=True)

    csv_path = os.path.join(run_dir, f'vae_results_run_{run_id}.csv')

    # 시스템 초기화
    try:
        system = OODSystem(MODEL_PATH)
    except FileNotFoundError:
        print("Please run train.py first!")
        return

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    print(">>> Loading Datasets...")
    # ID 데이터 로딩 (랜덤 샘플링)
    full_id = ImageFolderWithPaths(root=ID_DATA_DIR, transform=transform)
    id_indices = np.random.choice(len(full_id), min(len(full_id), 500), replace=False)
    id_loader = DataLoader(full_id, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(id_indices))

    # OOD 데이터 로딩
    full_ood = ImageFolderWithPaths(root=OOD_DATA_DIR, transform=transform)
    ood_indices = np.random.choice(len(full_ood), min(len(full_ood), 500), replace=False)
    ood_loader = DataLoader(full_ood, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(ood_indices))

    def evaluate_and_save(loader, label, writer):
        scores = []
        print(f"Evaluating {label}...")
        for img, path, filename in tqdm(loader):
            if img.shape[1] != 3: continue

            score = system.detect(img)
            scores.append(score)

            is_ood = score > ANOMALY_THRESHOLD
            prediction = "OOD" if is_ood else "ID"

            # CSV 기록
            writer.writerow([filename, label, score, prediction, path])

            # 이미지 복사
            dest = path_ood if is_ood else path_id
            shutil.copy(path[0], os.path.join(dest, f"[{score:.4f}]_{label}_{filename}"))

        return scores

    # 실행 및 저장
    f = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['Filename', 'True_Label', 'Anomaly_Score', 'Prediction', 'Original_Path'])

    id_scores = evaluate_and_save(id_loader, "Animals", writer)
    ood_scores = evaluate_and_save(ood_loader, "Pokemon", writer)

    f.close()

    # 시각화
    if id_scores and ood_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(id_scores, bins=50, alpha=0.6, label='Animals (ID)', density=True, color='blue')
        plt.hist(ood_scores, bins=50, alpha=0.6, label='Pokemon (OOD)', density=True, color='red')
        plt.axvline(x=ANOMALY_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({ANOMALY_THRESHOLD})')

        plt.title(f"Bayesian VAE OOD Detection (Run {run_id})")
        plt.xlabel("Anomaly Score (MSE + Uncertainty)")
        plt.legend()

        plt.savefig(os.path.join(run_dir, f'histogram_run_{run_id}.png'))
        print(f"\n>>> Run {run_id} Completed! Saved to: {run_dir}")
    else:
        print("Error: Not enough data.")


if __name__ == "__main__":
    run()