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

# Point to the full dataset directories
ID_DATA_DIR = '/app/data/animals'
OOD_DATA_DIR = '/app/data/pokemon'

# Base directory for all runs
BASE_RESULT_DIR = '/app/results/Animals-10/vae_full_analysis'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- [New Feature] Directory Management ---
def get_next_run_dir(base_dir):
    """
    Checks the base directory and creates a new 'run_X' folder
    that doesn't exist yet (e.g., run_1, run_2, ...).
    """
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


# --- Helper Classes ---
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
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state)
        self.model.eval()

    def detect_bayesian(self, img_tensor, samples=30):
        # [H100 Optimization] Replicate input for batch processing
        batch = img_tensor.repeat(samples, 1, 1, 1).to(self.device)
        target = img_tensor.to(self.device).repeat(samples, 1, 1, 1)

        with torch.no_grad():
            recon_batch, mu, logvar = self.model(batch)

        # 1. Negative ELBO (Model Fit)
        recon_loss = F.mse_loss(recon_batch, target, reduction='none').sum(dim=(1, 2, 3))
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # Expected Value over 30 samples
        expected_elbo = (recon_loss + kld_loss).mean().item()

        # 2. Epistemic Uncertainty (Model Confusion)
        # Variance of the latent vector across 30 samples
        latent_variance = mu.var(dim=0).sum().item()

        # Tuning the Alpha
        # This weight determines how much we trust "Uncertainty" vs "Reconstruction Error"
        alpha = 100.0

        return expected_elbo + (latent_variance * alpha)


def run_full_analysis():
    # 1. [Modified] Setup Directory using the new function
    current_run_dir, run_id = get_next_run_dir(BASE_RESULT_DIR)

    # All files will now be saved inside 'current_run_dir'
    csv_path = os.path.join(current_run_dir, f'full_analysis_report_run_{run_id}.csv')
    roc_plot_path = os.path.join(current_run_dir, f'roc_curve_run_{run_id}.png')

    # 2. Load Full Datasets (No Random Sampling)
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    print(f">>> Loading Full ID Dataset from: {ID_DATA_DIR}")
    # No SubsetRandomSampler -> Loads everything
    dataset_id = ImageFolderWithPaths(root=ID_DATA_DIR, transform=transform)
    loader_id = DataLoader(dataset_id, batch_size=1, shuffle=False, num_workers=4)

    print(f">>> Loading Full OOD Dataset from: {OOD_DATA_DIR}")
    dataset_ood = ImageFolderWithPaths(root=OOD_DATA_DIR, transform=transform)
    loader_ood = DataLoader(dataset_ood, batch_size=1, shuffle=False, num_workers=4)

    system = OODSystem(MODEL_PATH)

    # Metric Arrays
    y_true = []  # 0 = ID, 1 = OOD
    y_scores = []  # Anomaly Scores

    f = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['Filename', 'Type', 'Score', 'Path'])

    # --- Processing ID ---
    print(f"Processing {len(dataset_id)} Animal images...")
    for img, path, filename in tqdm(loader_id):
        if img.shape[1] != 3: continue
        score = system.detect_bayesian(img)
        y_true.append(0)
        y_scores.append(score)
        writer.writerow([filename, 'ID_Animal', f"{score:.4f}", path[0]])

    # --- Processing OOD ---
    print(f"Processing {len(dataset_ood)} Pokemon images...")
    for img, path, filename in tqdm(loader_ood):
        if img.shape[1] != 3: continue
        score = system.detect_bayesian(img)
        y_true.append(1)
        y_scores.append(score)
        writer.writerow([filename, 'OOD_Pokemon', f"{score:.4f}", path[0]])

    f.close()

    # --- Advanced Metrics (AUROC) ---
    print("\n>>> Calculating OOD Performance Metrics...")

    # AUROC: The probability that a random OOD image has a higher score than a random ID image.
    auroc = roc_auc_score(y_true, y_scores)

    # AUPR: Area Under Precision-Recall Curve (Good if datasets are imbalanced)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    print(f"==========================================")
    print(f" Run ID:                 {run_id}")
    print(f" Total Images Scanned:   {len(y_true)}")
    print(f" AUROC Score (Accuracy): {auroc:.5f} (Target: > 0.95)")
    print(f" AUPR Score:             {pr_auc:.5f}")
    print(f" Saved Results to:       {current_run_dir}")
    print(f"==========================================")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {auroc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (False Alarms)')
    plt.ylabel('True Positive Rate (Detection)')
    plt.title(f'OOD Detection Performance\n(Run {run_id} - Full Dataset Analysis)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Save to the specific run directory
    plt.savefig(roc_plot_path)
    print(f"Saved ROC Curve to {roc_plot_path}")


if __name__ == "__main__":
    run_full_analysis()