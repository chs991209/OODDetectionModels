import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from model import BayesianVAE
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

# --- [Configuration] ---
MODEL_PATH = '/app/models/Animals-10/vae/vae_final.pth'
ID_DATA_DIR = '/app/data/animals'
OOD_DATA_DIR = '/app/data/pokemon'
BASE_RESULT_DIR = '/app/results/Animals-10/vae_full_analysis'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_next_run_dir(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    i = 1
    while True:
        run_dir = os.path.join(base_dir, f"run_{i}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            print(f">>> [System] Created new result directory: {run_dir}")
            return run_dir, i
        i += 1


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
        torch.backends.cudnn.benchmark = True

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state)
        self.model.eval()

    def extract_raw_metrics(self, img_tensor, samples=30):
        """
        [Architectural Fix 3] 지표 추출과 점수 계산의 분리
        점수를 즉시 합산하지 않고, 정규화를 위해 3가지 순수 지표만 반환합니다.
        """
        img_on_device = img_tensor.to(self.device, non_blocking=True)
        batch = img_on_device.repeat(samples, 1, 1, 1)
        target = img_on_device.repeat(samples, 1, 1, 1)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                recon_batch, mu, logvar = self.model(batch)

        recon_batch = recon_batch.float()
        mu = mu.float()
        logvar = logvar.float()

        # 1. Negative ELBO
        recon_loss = F.mse_loss(recon_batch, target, reduction='none').sum(dim=(1, 2, 3))
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        expected_elbo = (recon_loss + kld_loss).mean().item()

        # 2. Mutual Information
        pixel_mi = recon_batch.var(dim=0).sum().item()
        latent_mi = mu.var(dim=0).sum().item()

        return expected_elbo, latent_mi, pixel_mi


def normalize_zscore(data_list):
    """Z-score 정규화를 수행하여 스케일을 통일합니다."""
    arr = np.array(data_list)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0: return np.zeros_like(arr)
    return (arr - mean) / std


def run_full_analysis():
    torch.set_float32_matmul_precision('high')
    current_run_dir, run_id = get_next_run_dir(BASE_RESULT_DIR)
    csv_path = os.path.join(current_run_dir, f'full_analysis_report_run_{run_id}.csv')
    roc_plot_path = os.path.join(current_run_dir, f'roc_curve_run_{run_id}.png')

    # [Ablation Factor 1] 전처리: 학습과 동일한 찌그러짐 적용 vs CenterCrop 비교
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 학습과 동일한 조건으로 세팅
        transforms.ToTensor()
    ])

    dataset_id = ImageFolderWithPaths(root=ID_DATA_DIR, transform=transform)
    loader_id = DataLoader(dataset_id, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    dataset_ood = ImageFolderWithPaths(root=OOD_DATA_DIR, transform=transform)
    loader_ood = DataLoader(dataset_ood, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    system = OODSystem(MODEL_PATH)

    # [Ablation Factor 2] MC 샘플 수 조절 가능하도록 변수화
    MC_SAMPLES = 30

    # [Ablation Factor 4] 지표별 가중치 조절 변수
    W_ELBO = 1.0
    W_LATENT = 3.0
    W_PIXEL = 1.0

    # ID와 OOD 지표를 철저히 분리하여 저장 (데이터 누수 방지)
    id_elbos, id_latent_mis, id_pixel_mis = [], [], []
    ood_elbos, ood_latent_mis, ood_pixel_mis = [], [], []

    y_true, file_info = [], []

    print(f"\n>>> Extracting metrics from ID (Animals) with MC={MC_SAMPLES}...")
    for img, path, filename in tqdm(loader_id):
        if img.shape[1] != 3: continue
        elbo, lat_mi, pix_mi = system.extract_raw_metrics(img, samples=MC_SAMPLES)
        y_true.append(0)
        file_info.append((filename, 'ID_Animal', path[0]))
        id_elbos.append(elbo);
        id_latent_mis.append(lat_mi);
        id_pixel_mis.append(pix_mi)

    print(f"\n>>> Extracting metrics from OOD (Pokemon) with MC={MC_SAMPLES}...")
    for img, path, filename in tqdm(loader_ood):
        if img.shape[1] != 3: continue
        elbo, lat_mi, pix_mi = system.extract_raw_metrics(img, samples=MC_SAMPLES)
        y_true.append(1)
        file_info.append((filename, 'OOD_Pokemon', path[0]))
        ood_elbos.append(elbo);
        ood_latent_mis.append(lat_mi);
        ood_pixel_mis.append(pix_mi)

    # ---------------------------------------------------------
    # [Ablation Factor 3] 학술적으로 엄밀한 Z-Score (기준을 ID에만 둠)
    # ---------------------------------------------------------
    print("\n>>> Strict Normalizing and Aggregating Scores...")

    def strict_zscore(id_list, ood_list):
        mean = np.mean(id_list)
        std = np.std(id_list) if np.std(id_list) > 0 else 1e-6
        z_id = (np.array(id_list) - mean) / std
        z_ood = (np.array(ood_list) - mean) / std
        return np.concatenate([z_id, z_ood])

    z_elbo = strict_zscore(id_elbos, ood_elbos)
    z_latent_mi = strict_zscore(id_latent_mis, ood_latent_mis)
    z_pixel_mi = strict_zscore(id_pixel_mis, ood_pixel_mis)

    # 가중치 합산
    y_scores = (z_elbo * W_ELBO) + (z_latent_mi * W_LATENT) + (z_pixel_mi * W_PIXEL)

    # ---------------------------------------------------------
    # [버그 픽스] CSV 저장을 위해 분리했던 순수 지표(Raw Metrics)를 순서대로 병합합니다.
    # y_true 및 file_info가 ID -> OOD 순서로 저장되어 있으므로 순서를 그대로 맞춥니다.
    # ---------------------------------------------------------
    raw_elbos = id_elbos + ood_elbos
    raw_latent_mis = id_latent_mis + ood_latent_mis
    raw_pixel_mis = id_pixel_mis + ood_pixel_mis


    # CSV 저장
    f = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['Filename', 'Type', 'Score', 'Raw_ELBO', 'Raw_LatentMI', 'Raw_PixelMI', 'Path'])

    for i in range(len(y_true)):
        writer.writerow([
            file_info[i][0], file_info[i][1], f"{y_scores[i]:.4f}",
            f"{raw_elbos[i]:.4f}", f"{raw_latent_mis[i]:.4f}", f"{raw_pixel_mis[i]:.4f}",
            file_info[i][2]
        ])
    f.close()

    # --- Metrics (AUROC) ---
    auroc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    print(f"==========================================")
    print(f" Run ID:                 {run_id}")
    print(f" Total Images Scanned:   {len(y_true)}")
    print(f" AUROC Score:            {auroc:.5f}")
    print(f" AUPR Score:             {pr_auc:.5f}")
    print(f" Saved Results to:       {current_run_dir}")
    print(f"==========================================")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {auroc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'OOD Detection Performance (Z-Score Aggregation)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(roc_plot_path)


if __name__ == "__main__":
    run_full_analysis()