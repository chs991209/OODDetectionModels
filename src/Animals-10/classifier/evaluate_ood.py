import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model import get_animal_model

# --- [설정] ---
# ... (설정 부분은 변경 없음) ...
MODEL_PATH = '/app/models/Animals-10/classifier/animals10_resnet18.pth'

ID_DATA_DIR = '/app/data/animals'
OOD_DATA_DIR = '/app/data/pokemon'

BASE_RESULT_DIR = '/app/results/Animals-10/classifier'

CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
           'elephant', 'horse', 'sheep', 'spider', 'squirrel']
NUM_MC_SAMPLES = 30
ENTROPY_THRESHOLD = 0.6
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 전처리, Dataset 정의, load_trained_model, get_next_run_dir 함수들은 변경 없음 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- 실행 폴더 자동 생성 ---
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


# --- Dataset 정의 ---
class OODDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        valid_extensions = ('.jpg', '.jpeg', '.png')
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    path = os.path.join(root, file)
                    self.samples.append((path, file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, filename = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, path, filename
        except Exception:
            return torch.zeros(3, 224, 224), "", ""


def load_trained_model():
    model = get_animal_model(num_classes=len(CLASSES), pretrained=False)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    return model


# --- 배치 처리 및 저장 ---
def process_dataloader(model, dataloader, label_type, csv_writer, run_dir):
    scores = []
    model.eval()
    # MC Dropout 활성화
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    img_save_dir = os.path.join(run_dir, 'sorted_images')
    path_ood = os.path.join(img_save_dir, 'Predicted_OOD')
    path_id = os.path.join(img_save_dir, 'Predicted_ID')
    os.makedirs(path_ood, exist_ok=True)
    os.makedirs(path_id, exist_ok=True)

    print(f"Processing {label_type}...")
    with torch.no_grad():
        for images, paths, filenames in tqdm(dataloader):
            images = images.to(DEVICE)
            if images.sum() == 0: continue

            mc_outputs = []
            for _ in range(NUM_MC_SAMPLES):
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                mc_outputs.append(probs.unsqueeze(0))

            mc_probs = torch.cat(mc_outputs, dim=0).mean(dim=0)
            epsilon = 1e-12
            entropy_batch = -torch.sum(mc_probs * torch.log(mc_probs + epsilon), dim=1)
            entropy_list = entropy_batch.cpu().numpy().tolist()
            pred_indices = torch.argmax(mc_probs, dim=1).cpu().numpy()

            scores.extend(entropy_list)

            for i in range(len(paths)):
                score = entropy_list[i]
                file_path = paths[i]
                file_name = filenames[i]
                pred_class_name = CLASSES[pred_indices[i]]

                is_ood = score > ENTROPY_THRESHOLD
                prediction = "OOD" if is_ood else "ID"

                csv_writer.writerow([file_name, label_type, score, prediction, pred_class_name, file_path])

                dest_folder = path_ood if is_ood else path_id
                dest_name = f"[{score:.4f}]_{prediction}_{file_name}"
                shutil.copy(file_path, os.path.join(dest_folder, dest_name))
    return scores


def main():
    print(f"Using Device: {DEVICE}")
    model = load_trained_model()

    # 실행 폴더 생성 (run_X)
    run_dir, run_id = get_next_run_dir(BASE_RESULT_DIR)
    csv_path = os.path.join(run_dir, f'ood_results_run_{run_id}.csv')

    print(f">>> Loading datasets...")
    id_dataset = OODDataset(ID_DATA_DIR, transform=transform)
    ood_dataset = OODDataset(OOD_DATA_DIR, transform=transform)

    id_loader = DataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    if len(id_dataset) == 0:
        print("Error: No ID data found.")
        return

    f = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['Filename', 'True_Label', 'Entropy_Score', 'Final_Prediction', 'Pred_Class', 'Full_Path'])

    print(f"\n=== Starting Evaluation (Run {run_id}) ===")
    id_scores = process_dataloader(model, id_loader, "ID(Animal)", writer, run_dir)
    ood_scores = process_dataloader(model, ood_loader, "OOD(Pokemon)", writer, run_dir)
    f.close()

    if not id_scores or not ood_scores:
        print("Error: Not enough data.")
        return

    # 1. 평균 엔트로피 계산
    mean_id_entropy = np.mean(id_scores)
    mean_ood_entropy = np.mean(ood_scores)

    # 2. 결과 텍스트 파일 저장
    results_txt_path = os.path.join(run_dir, f'mean_entropy_run_{run_id}.txt')
    with open(results_txt_path, 'w') as txt_file:
        txt_file.write(f"--- OOD Evaluation Summary (Run {run_id}) ---\n")
        txt_file.write(f"ID (Animals) Mean Entropy: {mean_id_entropy:.4f}\n")
        txt_file.write(f"OOD (Pokemon) Mean Entropy: {mean_ood_entropy:.4f}\n")
        txt_file.write(f"Entropy Threshold Used: {ENTROPY_THRESHOLD:.4f}\n")

    print(f"\nMean entropy summary saved to: {results_txt_path}")

    # 3. 그래프 저장
    plt.figure(figsize=(10, 6))
    plt.hist(id_scores, bins=50, alpha=0.5, label='ID: Animals', color='blue', density=True)
    plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD: Pokemon', color='red', density=True)
    plt.axvline(x=ENTROPY_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({ENTROPY_THRESHOLD})')

    plt.xlabel('Uncertainty (Entropy)')
    plt.ylabel('Density')
    plt.title(f'OOD Detection Result (Run {run_id})\nID Mean: {mean_id_entropy:.4f}, OOD Mean: {mean_ood_entropy:.4f}')
    plt.legend()

    plot_path = os.path.join(run_dir, f'histogram_run_{run_id}.png')
    plt.savefig(plot_path)

    print(f"\n>>> Run {run_id} Completed!")
    print(f"    Saved to: {run_dir}")


if __name__ == "__main__":
    main()