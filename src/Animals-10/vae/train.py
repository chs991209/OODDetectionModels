import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import BayesianVAE, vae_loss_function
import os

# --- [설정] ---
DATA_PATH = '/app/data/animals'
MODEL_SAVE_PATH = '/app/models/Animals-10/vae/vae_final.pth'

BATCH_SIZE = 256
NUM_EPOCHS = 50

# [Architectural Fix 2] 가상 대용량 배치 (Gradient Accumulation)
# RTX 4090의 VRAM 한계(256)를 극복하고 H100 수준의 배치(1024) 효과를 냅니다.
ACCUMULATION_STEPS = 4


def train():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Accumulation Steps: {ACCUMULATION_STEPS}")
    print(f"Learning Plan: Epochs: {NUM_EPOCHS}")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")

    # transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    # # Modification
    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    #     transforms.RandomHorizontalFlip(),  # 동물은 좌우 반전되어도 동물이므로 학습에 매우 좋습니다
    #     transforms.ToTensor()
    # ])

    # Modification
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model = BayesianVAE().to(device)
    model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    #
    # print(">>> 학습 시작 (RTX 4090 / BF16 + FP32 Loss + Grad Accumulation)...")
    model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        optimizer.zero_grad()  # 에포크 시작 시 초기화

        for i, (data, _) in enumerate(dataloader):
            data = data.to(device, non_blocking=True)

            # 네트워크 Forward는 bfloat16으로 가속 (Tensor Core 활용)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                recon, mu, logvar = model(data)

                # vae_loss_function 내부에서 자동으로 FP32 캐스팅을 수행합니다.
                loss = vae_loss_function(recon, data, mu, logvar)

            # Loss 스케일링 (누적되는 그래디언트의 평균을 맞추기 위함)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            total_loss += (loss.item() * ACCUMULATION_STEPS)

            # 지정된 스텝(또는 마지막 배치)에 도달했을 때 가중치 업데이트
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()  # 업데이트 후 초기화

        # scheduler.step()
        # current_lr = scheduler.get_last_lr()[0]

        # print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: Loss {total_loss / len(dataset):.4f} | LR: {current_lr:.6f}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: Loss {total_loss / len(dataset):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f">>> 모델 저장 완료: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()