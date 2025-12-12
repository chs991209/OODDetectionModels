import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import BayesianVAE, vae_loss_function
import os

# --- [설정] ---
DATA_PATH = '/app/data/animals'

# [핵심] 모델 저장 경로 수정 (Animals-10/vae)
MODEL_SAVE_PATH = '/app/models/Animals-10/vae/vae_final.pth'

BATCH_SIZE = 256
NUM_EPOCHS = 50


def train():
    # [H100 Optimization]
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    # 모델 저장 폴더 자동 생성
    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model = BayesianVAE().to(device)
    model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(">>> H100 학습 시작...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for data, _ in dataloader:
            data = data.to(device)
            optimizer.zero_grad()

            # [H100] BF16 Mixed Precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                recon, mu, logvar = model(data)
                loss = vae_loss_function(recon, data, mu, logvar)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: Loss {total_loss / len(dataset):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f">>> 모델 저장 완료: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()