# OOD Detection System

Out-of-Distribution (OOD) Detection system using two different approaches: Classifier-based (ResNet18 + MC Dropout) and VAE-based (Bayesian Variational Autoencoder) methods.

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Directory Structure](#directory-structure)
4. [Methods](#methods)
   - [Method 1: Classifier-Based OOD Detection](#method-1-classifier-based-ood-detection)
   - [Method 2: VAE-Based OOD Detection](#method-2-vae-based-ood-detection)
5. [Quick Start](#quick-start)
6. [Docker Usage Guide](#docker-usage-guide)
7. [Output Format](#output-format)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)
10. [References & Learning Resources](#references--learning-resources)

---

## 📋 Overview

This system implements **Out-of-Distribution (OOD) Detection** using two different approaches to identify images that don't belong to the training distribution. The system is designed to work with the **Animals-10** dataset (In-Distribution) and **Pokemon** dataset (Out-of-Distribution).

### What is OOD Detection?

OOD detection is the task of identifying whether a new input belongs to the same distribution as the training data. In this system:
- **ID (In-Distribution)**: Animals-10 dataset (butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel)
- **OOD (Out-of-Distribution)**: Pokemon dataset (images that are not animals)

---

## 🏗️ System Architecture

The system consists of two independent OOD detection methods. Below are 5 different architectural views of the system:

---

### Architecture View 1: System Overview

**High-level component diagram showing the overall system structure:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      OOD Detection System                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────┐    ┌──────────────────────────┐      │
│  │   Method 1: Classifier   │    │   Method 2: VAE          │      │
│  │   ────────────────────   │    │   ────────────────────   │      │
│  │                          │    │                          │      │
│  │  ResNet18 + MC Dropout   │    │  Bayesian VAE            │      │
│  │  • Pretrained ImageNet   │    │  • Encoder-Decoder       │      │
│  │  • Fine-tuned on Animals │    │  • Latent Space (128D)   │      │
│  │  • Entropy-based OOD     │    │  • Reconstruction-based  │      │
│  └──────────┬───────────────┘    └──────────┬───────────────┘      │
│             │                                │                      │
│             │                                │                      │
│             └────────────┬───────────────────┘                      │
│                          │                                          │
│                  ┌───────▼────────┐                                 │
│                  │  Results Layer │                                 │
│                  │  • CSV Reports │                                 │
│                  │  • Histograms  │                                 │
│                  │  • Sorted Imgs │                                 │
│                  └────────────────┘                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Architecture View 2: Data Flow Architecture

**How data flows through the system from input to output:**

```
┌─────────────┐
│   Input     │  Animals-10 (ID) / Pokemon (OOD)
│   Images    │
└──────┬──────┘
       │
       ├─────────────────────────────┬─────────────────────────────┐
       │                             │                             │
       ▼                             ▼                             ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│ Preprocessing│            │ Preprocessing│            │ Preprocessing│
│ (224x224)    │            │ (64x64)      │            │ (224x224)    │
│ Normalize    │            │ ToTensor     │            │ Normalize    │
└──────┬───────┘            └──────┬───────┘            └──────┬───────┘
       │                          │                            │
       │                          │                            │
       ▼                          ▼                            ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│  Classifier  │            │     VAE      │            │ Single Image │
│   Pipeline   │            │   Pipeline   │            │  Detection   │
│              │            │              │            │              │
│ • 30x MC     │            │ • 30x MC     │            │ • 30x MC     │
│   Forward    │            │   Reconstruct│            │   Forward    │
│ • Entropy    │            │ • MSE + Var  │            │ • Entropy    │
│   Calc       │            │   Calc       │            │   Calc       │
└──────┬───────┘            └──────┬───────┘            └──────┬───────┘
       │                          │                            │
       │                          │                            │
       └──────────────┬───────────┴────────────┬───────────────┘
                      │                       │
                      ▼                       ▼
              ┌──────────────┐       ┌──────────────┐
              │   Decision   │       │   Results    │
              │   Logic      │       │   Storage    │
              │              │       │              │
              │ ID/OOD       │       │ • CSV        │
              │ Threshold    │       │ • Images     │
              │ Comparison   │       │ • Plots      │
              └──────────────┘       └──────────────┘
```

---

### Architecture View 3: Component Interaction Architecture

**How different components interact with each other:**

```
┌──────────────────────────────────────────────────────────────────┐
│                    Component Interaction View                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────┐│
│  │   Docker     │────────▶│   Source     │────────▶│  Models  ││
│  │  Containers  │  Mount  │   Code      │  Train  │  Storage ││
│  │              │         │              │         │          ││
│  │ • Classifier │         │ • train.py   │         │ • .pth   ││
│  │ • VAE        │         │ • evaluate   │         │ • Weights││
│  └──────┬───────┘         └──────┬───────┘         └────┬─────┘│
│         │                        │                      │      │
│         │                        │                      │      │
│         │                        ▼                      │      │
│         │              ┌──────────────┐                │      │
│         │              │   Data       │                │      │
│         │              │   Loader     │                │      │
│         │              │              │                │      │
│         │              │ • Animals    │                │      │
│         │              │ • Pokemon    │                │      │
│         │              └──────┬───────┘                │      │
│         │                     │                        │      │
│         │                     │                        │      │
│         └────────────────────┼────────────────────────┘      │
│                              │                               │
│                              ▼                               │
│                    ┌──────────────┐                          │
│                    │  Evaluation  │                          │
│                    │   Engine     │                          │
│                    │              │                          │
│                    │ • MC Sampling│                          │
│                    │ • Score Calc │                          │
│                    │ • Threshold  │                          │
│                    └──────┬───────┘                          │
│                           │                                  │
│                           ▼                                  │
│                    ┌──────────────┐                          │
│                    │   Results    │                          │
│                    │   Manager    │                          │
│                    │              │                          │
│                    │ • CSV Writer │                          │
│                    │ • Image Copy │                          │
│                    │ • Plot Gen   │                          │
│                    └──────────────┘                          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### Architecture View 4: Training Pipeline Architecture

**Detailed flow of the training process for both methods:**

```
┌──────────────────────────────────────────────────────────────────┐
│                    Training Pipeline Architecture                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CLASSIFIER TRAINING PIPELINE:                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  [Animals Dataset]                                       │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [DataLoader] ──► [Transform: 224x224, Normalize]        │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [ResNet18] ──► [Pretrained ImageNet Weights]            │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Modify FC] ──► [Dropout(0.5) + Linear(10)]            │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Training Loop]                                          │   │
│  │    • Forward Pass                                         │   │
│  │    • CrossEntropy Loss                                    │   │
│  │    • Backward Pass                                        │   │
│  │    • Adam Optimizer                                       │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Save Model] ──► /app/models/Animals-10/classifier/     │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  VAE TRAINING PIPELINE:                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  [Animals Dataset]                                       │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [DataLoader] ──► [Transform: 64x64, ToTensor]           │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Bayesian VAE]                                          │   │
│  │       │                                                   │   │
│  │       ├─► [Encoder] ──► [μ, log(σ²)] ──► [z ~ N(μ,σ²)]  │   │
│  │       │                                                   │   │
│  │       └─► [Decoder] ──► [Reconstruction]                  │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Loss Calculation]                                      │   │
│  │    • MSE (Reconstruction)                                │   │
│  │    • KL Divergence (Regularization)                      │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Training Loop] (BF16 Mixed Precision)                 │   │
│  │    • Forward Pass                                        │   │
│  │    • Loss Backward                                       │   │
│  │    • Adam Optimizer                                      │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Save Model] ──► /app/models/Animals-10/vae/            │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### Architecture View 5: Inference Pipeline Architecture

**Detailed flow of the OOD detection/evaluation process:**

```
┌──────────────────────────────────────────────────────────────────┐
│                  Inference Pipeline Architecture                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CLASSIFIER INFERENCE PIPELINE:                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  [Input Image] ──► [Preprocess: 224x224, Normalize]      │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [MC Dropout Loop: 30 iterations]                        │   │
│  │       │                                                   │   │
│  │       ├─► [Forward Pass 1] ──► [Logits] ──► [Softmax]   │   │
│  │       ├─► [Forward Pass 2] ──► [Logits] ──► [Softmax]   │   │
│  │       ├─► ...                                            │   │
│  │       └─► [Forward Pass 30] ──► [Logits] ──► [Softmax]  │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Average Probabilities] ──► [Mean Distribution]         │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Entropy Calculation]                                    │   │
│  │    H = -Σ(p_i * log(p_i))                                │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Decision]                                              │   │
│  │    if H > 0.6: OOD                                       │   │
│  │    else: ID (with predicted class)                       │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  VAE INFERENCE PIPELINE:                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  [Input Image] ──► [Preprocess: 64x64, ToTensor]         │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [MC Sampling Loop: 30 iterations]                       │   │
│  │       │                                                   │   │
│  │       ├─► [Encode] ──► [Sample z₁] ──► [Decode] ──► [Recon₁]│
│  │       ├─► [Encode] ──► [Sample z₂] ──► [Decode] ──► [Recon₂]│
│  │       ├─► ...                                            │   │
│  │       └─► [Encode] ──► [Sample z₃₀] ──► [Decode] ──► [Recon₃₀]│
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Calculate Scores]                                       │   │
│  │    • Mean Reconstruction = mean(Recon₁...Recon₃₀)         │   │
│  │    • Reconstruction Error = MSE(Original, Mean Recon)     │   │
│  │    • Uncertainty = Variance(Recon₁...Recon₃₀)            │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Anomaly Score]                                          │   │
│  │    Score = Reconstruction Error + Uncertainty             │   │
│  │       │                                                   │   │
│  │       ▼                                                   │   │
│  │  [Decision]                                               │   │
│  │    if Score > 0.025: OOD                                  │   │
│  │    else: ID                                               │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  COMMON OUTPUT PROCESSING:                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  [OOD Decision] ──► [Result Storage]                     │   │
│  │       │                      │                            │   │
│  │       │                      ├─► [CSV File]               │   │
│  │       │                      ├─► [Image Copy]             │   │
│  │       │                      └─► [Histogram Plot]         │   │
│  │       │                                                   │   │
│  │       └─► [Visualization] ──► [Results Directory]        │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
OOD/
├── data/                          # Dataset storage
│   ├── animals/                   # In-Distribution data (Animals-10)
│   │   ├── butterfly/
│   │   ├── cat/
│   │   ├── chicken/
│   │   └── ... (10 animal classes)
│   └── pokemon/                   # Out-of-Distribution data
│       └── unknown/
│
├── models/                        # Trained model weights
│   └── Animals-10/
│       ├── classifier/            # ResNet18 classifier model
│       │   └── animals10_resnet18.pth
│       └── vae/                   # Bayesian VAE model
│           └── vae_final.pth
│
├── results/                       # Evaluation results
│   └── Animals-10/
│       ├── classifier/
│       │   └── run_1/             # Each run creates a new folder
│       │       ├── ood_results_run_1.csv
│       │       ├── histogram_run_1.png
│       │       └── sorted_images/
│       └── vae/
│           └── run_1/
│               ├── vae_results_run_1.csv
│               ├── histogram_run_1.png
│               └── sorted_images/
│
├── src/                           # Source code
│   └── Animals-10/
│       ├── classifier/            # Classifier-based OOD detection
│       │   ├── model.py          # ResNet18 with MC Dropout
│       │   ├── train.py          # Training script
│       │   ├── evaluate_ood.py   # Batch evaluation
│       │   └── detect_ood.py    # Single image detection
│       └── vae/                   # VAE-based OOD detection
│           ├── model.py          # Bayesian VAE architecture
│           ├── train.py          # Training script
│           └── evaluate_ood.py  # Evaluation script
│
├── docker/                        # Docker configuration
│   ├── Dockerfile.classifier     # Classifier container
│   └── Dockerfile.vae            # VAE container
│
└── docker-compose.yml            # Container orchestration
```

---

## Methods

### Method 1: Classifier-Based OOD Detection

#### Architecture

- **Model**: ResNet18 (pretrained on ImageNet)
- **Technique**: Monte Carlo (MC) Dropout for uncertainty estimation
- **Detection Metric**: Entropy of predicted class probabilities

#### How It Works

1. **Training Phase** (`classifier/train.py`):
   - Loads ResNet18 pretrained on ImageNet
   - Replaces final layer with Dropout (p=0.5) + Linear layer
   - Fine-tunes on Animals-10 dataset
   - Saves model to `/app/models/Animals-10/classifier/`

2. **Detection Phase** (`classifier/evaluate_ood.py`):
   - For each image, performs **30 forward passes** with Dropout enabled
   - Calculates average probability distribution across all passes
   - Computes **entropy** of the distribution:
     ```
     Entropy = -Σ(p_i * log(p_i))
     ```
   - **High entropy** → Model is uncertain → Likely OOD
   - **Low entropy** → Model is confident → Likely ID

3. **Decision Rule**:
   - If `entropy > 0.6` → **OOD** (Pokemon/Unknown)
   - If `entropy ≤ 0.6` → **ID** (Animal class)

#### Key Features

- **MC Dropout**: Enables uncertainty quantification during inference
- **Entropy-based scoring**: Measures prediction confidence
- **Batch processing**: Efficient evaluation of large datasets

---

### Method 2: VAE-Based OOD Detection

#### Architecture

- **Model**: Bayesian Variational Autoencoder (VAE)
- **Technique**: Reconstruction error + uncertainty estimation
- **Detection Metric**: Anomaly score (MSE + variance)

#### How It Works

1. **Training Phase** (`vae/train.py`):
   - Trains a VAE to reconstruct animal images
   - Encoder: Compresses images to latent space (128 dimensions)
   - Decoder: Reconstructs images from latent codes
   - Uses **MSE loss + KL divergence** (standard VAE loss)
   - Optimized for H100 GPU with mixed precision (BF16)
   - Saves model to `/app/models/Animals-10/vae/`

2. **Detection Phase** (`vae/evaluate_ood.py`):
   - For each image, performs **30 reconstructions** (MC sampling)
   - Calculates:
     - **Reconstruction Error**: MSE between original and mean reconstruction
     - **Uncertainty**: Variance across 30 reconstructions
   - **Anomaly Score** = Reconstruction Error + Uncertainty
   - **High score** → Poor reconstruction → Likely OOD
   - **Low score** → Good reconstruction → Likely ID

3. **Decision Rule**:
   - If `anomaly_score > 0.025` → **OOD** (Pokemon)
   - If `anomaly_score ≤ 0.025` → **ID** (Animal)

#### Key Features

- **Reconstruction-based**: Learns the distribution of ID data
- **Bayesian uncertainty**: Quantifies model uncertainty
- **H100 optimized**: Uses torch.compile and BF16 precision

---

## 🚀 Quick Start

### Step 1: Prepare Data

```bash
# Extract datasets
unzip data/animals.zip -d data/
unzip pokemon.zip -d data/pokemon/
```

### Step 2: Start Containers

```bash
docker-compose up -d
```

### Step 3: Train Models

**Train Classifier:**
```bash
docker exec -it animals_classifier_container bash
cd /app/src/Animals-10/classifier
python train.py
```

**Train VAE:**
```bash
docker exec -it ood_vae_container bash
cd /app/src/Animals-10/vae
python train.py
```

### Step 4: Evaluate OOD Detection

**Evaluate with Classifier:**
```bash
docker exec -it animals_classifier_container bash
cd /app/src/Animals-10/classifier
python evaluate_ood.py
```

**Evaluate with VAE:**
```bash
docker exec -it ood_vae_container bash
cd /app/src/Animals-10/vae
python evaluate_ood.py
```

### Step 5: Single Image Detection (Classifier only)

```bash
docker exec -it animals_classifier_container bash
cd /app/src/Animals-10/classifier
python detect_ood.py --image /path/to/image.jpg
```

---

## 🐳 Docker Usage Guide

### Prerequisites

#### Required Software

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **NVIDIA Docker Runtime**: For GPU support (nvidia-docker2)
- **NVIDIA GPU**: With CUDA support (for training/evaluation)

#### Verify Installation

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Container Overview

The system uses two Docker containers:

#### 1. Classifier Container
- **Container Name**: `animals_classifier_container`
- **Image**: `animals-classifier:v1`
- **Purpose**: Classifier training and evaluation
- **Base Image**: `nvcr.io/nvidia/pytorch:23.10-py3`
- **Ports**: 
  - `8889:8888` (Jupyter Lab)
  - `6006:6006` (TensorBoard)

#### 2. VAE Container
- **Container Name**: `ood_vae_container`
- **Image**: `ood-vae:h100`
- **Purpose**: VAE training and evaluation
- **Base Image**: `nvcr.io/nvidia/pytorch:23.10-py3`
- **Ports**: 
  - `8888:8888` (Jupyter Lab)
- **Optimization**: H100 GPU optimized with BF16 support

### Initial Setup

#### Step 1: Build Docker Images

```bash
# Build all containers
docker-compose build

# Or build individually
docker-compose build classifier
docker-compose build vae
```

#### Step 2: Start Containers

```bash
# Start containers in detached mode
docker-compose up -d

# View container status
docker-compose ps
```

#### Step 3: Verify Containers

```bash
# Check if containers are running
docker ps

# View container logs
docker-compose logs classifier
docker-compose logs vae
```

### Container Management

#### Starting Containers

```bash
# Start all containers
docker-compose up -d

# Start specific container
docker-compose up -d classifier
docker-compose up -d vae
```

#### Stopping Containers

```bash
# Stop all containers
docker-compose down

# Stop without removing volumes
docker-compose stop

# Stop specific container
docker-compose stop classifier
```

#### Restarting Containers

```bash
# Restart all containers
docker-compose restart

# Restart specific container
docker-compose restart classifier
```

#### Viewing Logs

```bash
# View logs for all containers
docker-compose logs

# View logs for specific container
docker-compose logs classifier
docker-compose logs vae

# Follow logs in real-time
docker-compose logs -f classifier

# View last 100 lines
docker-compose logs --tail=100 classifier
```

### Running Commands

#### Interactive Shell Access

**Classifier Container:**
```bash
# Enter interactive bash shell
docker exec -it animals_classifier_container bash

# Once inside, you're in /app directory
cd /app/src/Animals-10/classifier
```

**VAE Container:**
```bash
# Enter interactive bash shell
docker exec -it ood_vae_container bash

# Once inside, you're in /app directory
cd /app/src/Animals-10/vae
```

#### Running Python Scripts

**From Host (without entering container):**
```bash
# Run classifier training
docker exec -it animals_classifier_container \
  python /app/src/Animals-10/classifier/train.py

# Run classifier evaluation
docker exec -it animals_classifier_container \
  python /app/src/Animals-10/classifier/evaluate_ood.py

# Run VAE training
docker exec -it ood_vae_container \
  python /app/src/Animals-10/vae/train.py

# Run VAE evaluation
docker exec -it ood_vae_container \
  python /app/src/Animals-10/vae/evaluate_ood.py

# Single image detection
docker exec -it animals_classifier_container \
  python /app/src/Animals-10/classifier/detect_ood.py \
  --image /app/data/pokemon/unknown/image.jpg
```

**From Inside Container:**
```bash
# Enter container first
docker exec -it animals_classifier_container bash

# Then run scripts
cd /app/src/Animals-10/classifier
python train.py
python evaluate_ood.py
python detect_ood.py --image /app/data/pokemon/unknown/image.jpg
```

#### Running with GPU

Both containers are configured with GPU support. Verify GPU access:

```bash
# Check GPU in classifier container
docker exec -it animals_classifier_container nvidia-smi

# Check GPU in VAE container
docker exec -it ood_vae_container nvidia-smi

# Run Python with GPU check
docker exec -it animals_classifier_container \
  python -c "import torch; print(torch.cuda.is_available())"
```

### Volume Mounts

The containers use volume mounts to share data between host and containers:

#### Volume Mapping

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./src` | `/app/src` | Source code |
| `./data` | `/app/data` | Datasets |
| `./models` | `/app/models` | Trained models |
| `./results` | `/app/results` | Evaluation results |

#### Accessing Files

**From Host to Container:**
- Files in `./src/` are accessible at `/app/src/` in container
- Files in `./data/` are accessible at `/app/data/` in container
- Models saved to `/app/models/` appear in `./models/` on host
- Results saved to `/app/results/` appear in `./results/` on host

**Example:**
```bash
# On host: create a test file
echo "test" > ./src/test.txt

# In container: access the file
docker exec -it animals_classifier_container cat /app/src/test.txt
```

#### Important Notes

- **Real-time Sync**: Changes in host directories are immediately visible in containers
- **No Copy Needed**: Files are shared, not copied
- **Persistent Storage**: Data persists even after container removal (unless using `-v` flag)

### Ports and Services

#### Port Mapping

| Container | Host Port | Container Port | Service |
|-----------|-----------|----------------|---------|
| Classifier | 8889 | 8888 | Jupyter Lab |
| Classifier | 6006 | 6006 | TensorBoard |
| VAE | 8888 | 8888 | Jupyter Lab |

#### Accessing Services

**Jupyter Lab (Classifier):**
```bash
# Access at: http://localhost:8889
# Default password/token: Check container logs
docker-compose logs classifier | grep token
```

**Jupyter Lab (VAE):**
```bash
# Access at: http://localhost:8888
# Default password/token: Check container logs
docker-compose logs vae | grep token
```

**TensorBoard (Classifier):**
```bash
# Start TensorBoard inside container
docker exec -it animals_classifier_container \
  tensorboard --logdir=/app/results --port=6006 --host=0.0.0.0

# Access at: http://localhost:6006
```

### Common Workflows

#### Workflow 1: Complete Training and Evaluation

```bash
# 1. Start containers
docker-compose up -d

# 2. Train classifier
docker exec -it animals_classifier_container \
  python /app/src/Animals-10/classifier/train.py

# 3. Train VAE
docker exec -it ood_vae_container \
  python /app/src/Animals-10/vae/train.py

# 4. Evaluate classifier
docker exec -it animals_classifier_container \
  python /app/src/Animals-10/classifier/evaluate_ood.py

# 5. Evaluate VAE
docker exec -it ood_vae_container \
  python /app/src/Animals-10/vae/evaluate_ood.py
```

#### Workflow 2: Interactive Development

```bash
# 1. Start containers
docker-compose up -d

# 2. Enter classifier container
docker exec -it animals_classifier_container bash

# 3. Inside container, navigate and work
cd /app/src/Animals-10/classifier
python train.py  # Edit code on host, run in container
```

#### Workflow 3: Single Image Testing

```bash
# Test single image with classifier
docker exec -it animals_classifier_container \
  python /app/src/Animals-10/classifier/detect_ood.py \
  --image /app/data/pokemon/unknown/pikachu.jpg
```

#### Workflow 4: Monitoring Training

```bash
# Terminal 1: Start training
docker exec -it animals_classifier_container \
  python /app/src/Animals-10/classifier/train.py

# Terminal 2: Monitor logs
docker-compose logs -f classifier

# Terminal 3: Check GPU usage
watch -n 1 docker exec animals_classifier_container nvidia-smi
```

### Docker Troubleshooting

#### Issue 1: Container Won't Start

**Symptoms**: Container exits immediately after starting

**Solutions**:
```bash
# Check logs
docker-compose logs classifier

# Check if port is already in use
netstat -tulpn | grep 8889

# Rebuild container
docker-compose build --no-cache classifier
docker-compose up -d classifier
```

#### Issue 2: GPU Not Available

**Symptoms**: `torch.cuda.is_available()` returns `False`

**Solutions**:
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check container GPU access
docker exec -it animals_classifier_container nvidia-smi

# Verify docker-compose.yml has runtime: nvidia
cat docker-compose.yml | grep runtime
```

#### Issue 3: Permission Denied

**Symptoms**: Cannot write to mounted volumes

**Solutions**:
```bash
# Check file permissions
ls -la ./models
ls -la ./results

# Fix permissions (if needed)
sudo chown -R $USER:$USER ./models ./results
```

#### Issue 4: Out of Memory

**Symptoms**: CUDA out of memory errors

**Solutions**:
```bash
# Reduce batch size in training scripts
# Edit: src/Animals-10/classifier/train.py
# Change: BATCH_SIZE = 32  # Reduce from 64
```

#### Issue 5: Module Not Found

**Symptoms**: `ImportError: No module named 'X'`

**Solutions**:
```bash
# Install missing package in container
docker exec -it animals_classifier_container pip install package_name

# Or rebuild container with new dependencies
# Edit Dockerfile, then:
docker-compose build classifier
```

### Docker Quick Reference

#### Essential Commands

```bash
# Start all
docker-compose up -d

# Stop all
docker-compose down

# View logs
docker-compose logs -f

# Enter container
docker exec -it animals_classifier_container bash
docker exec -it ood_vae_container bash

# Run script
docker exec -it animals_classifier_container python /app/src/.../script.py

# Check GPU
docker exec -it animals_classifier_container nvidia-smi

# Rebuild
docker-compose build

# Clean restart
docker-compose down && docker-compose up -d
```

#### File Paths Reference

| Task | Host Path | Container Path |
|------|-----------|----------------|
| Edit code | `./src/...` | `/app/src/...` |
| Add data | `./data/...` | `/app/data/...` |
| Check models | `./models/...` | `/app/models/...` |
| View results | `./results/...` | `/app/results/...` |

---

## 📊 Output Format

### Results Directory Structure

Each evaluation run creates a new `run_X` folder:

```
results/Animals-10/classifier/run_1/
├── ood_results_run_1.csv          # Detailed results per image
├── mean_entropy_run_1.txt          # Summary statistics
├── histogram_run_1.png             # Visualization
└── sorted_images/
    ├── Predicted_ID/               # Images classified as ID
    └── Predicted_OOD/              # Images classified as OOD
```

### CSV Format

**Classifier Results:**
- `Filename`: Image filename
- `True_Label`: ID(Animal) or OOD(Pokemon)
- `Entropy_Score`: Uncertainty score
- `Final_Prediction`: ID or OOD
- `Pred_Class`: Predicted animal class
- `Full_Path`: Original image path

**VAE Results:**
- `Filename`: Image filename
- `True_Label`: Animals or Pokemon
- `Anomaly_Score`: Reconstruction error + uncertainty
- `Prediction`: ID or OOD
- `Original_Path`: Original image path

---

## 🔧 Configuration

### Key Configuration Parameters

#### Classifier Method
- `NUM_MC_SAMPLES = 30`: Number of forward passes for uncertainty estimation
- `ENTROPY_THRESHOLD = 0.6`: OOD detection threshold
- `BATCH_SIZE = 64`: Evaluation batch size
- `NUM_EPOCHS = 10`: Training epochs

#### VAE Method
- `ANOMALY_THRESHOLD = 0.025`: OOD detection threshold
- `BATCH_SIZE = 256`: Training batch size
- `NUM_EPOCHS = 50`: Training epochs
- `latent_dim = 128`: Latent space dimensionality

### Comparison of Methods

| Aspect | Classifier Method | VAE Method |
|--------|------------------|------------|
| **Approach** | Discriminative | Generative |
| **Detection** | Entropy (uncertainty) | Reconstruction error |
| **Training** | Faster (10 epochs) | Slower (50 epochs) |
| **Inference** | 30 forward passes | 30 reconstructions |
| **Interpretability** | Class probabilities | Visual reconstruction |
| **Use Case** | When you have labels | When you only have ID data |

---

## 🛠️ Troubleshooting

### General Issues

1. **Model not found**: Ensure training scripts have been run first
2. **CUDA out of memory**: Reduce batch size in evaluation scripts
3. **No data found**: Check that datasets are extracted in `data/` directory
4. **Container issues**: Use `docker-compose logs` to check container status

### Understanding the Results

#### Classifier Method
- **Low entropy** (< 0.6): Model is confident → ID
- **High entropy** (> 0.6): Model is uncertain → OOD

#### VAE Method
- **Low anomaly score** (< 0.025): Good reconstruction → ID
- **High anomaly score** (> 0.025): Poor reconstruction → OOD

#### Visualization
The histogram plots show the distribution of scores for ID and OOD samples. A good OOD detector should show:
- Clear separation between ID and OOD distributions
- ID samples clustered at low scores
- OOD samples spread at high scores

---

## 📚 References & Learning Resources

### Academic References

- **MC Dropout**: Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation
- **VAE**: Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes
- **ResNet**: He, K., et al. (2016). Deep Residual Learning for Image Recognition

### Computer Vision Course

The following video series will help you improve your computer vision skills and deepen your understanding of the concepts used in this OOD detection system:

<div align="center">

<iframe width="560" height="315" src="https://www.youtube.com/embed/2fq9wYslV0A?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

</div>

**Direct Link**: [Computer Vision Course - YouTube](https://www.youtube.com/watch?v=2fq9wYslV0A&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16)

This course covers essential computer vision topics that are directly relevant to this OOD detection system, including:
- Deep learning architectures (ResNet, VAE)
- Uncertainty estimation techniques
- Out-of-distribution detection methods
- Model evaluation and interpretation

### Additional Resources

- **Docker Documentation**: https://docs.docker.com/
- **Docker Compose Documentation**: https://docs.docker.com/compose/
- **NVIDIA Container Toolkit**: https://github.com/NVIDIA/nvidia-docker

---

## 📝 Notes

- Both methods use **Monte Carlo sampling** (30 samples) for uncertainty estimation
- Results are automatically organized into `run_X` folders to track multiple experiments
- Images are copied to `sorted_images/` folders for visual inspection
- The system is optimized for GPU execution (CUDA)
- VAE method is specifically optimized for H100 GPUs with BF16 precision

---

*Last Updated: Complete documentation for OOD Detection System with Animals-10 and Pokemon datasets*

