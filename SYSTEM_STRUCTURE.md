# OOD Detection System - Structure Documentation

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

## 🔬 Method 1: Classifier-Based OOD Detection

### Architecture

- **Model**: ResNet18 (pretrained on ImageNet)
- **Technique**: Monte Carlo (MC) Dropout for uncertainty estimation
- **Detection Metric**: Entropy of predicted class probabilities

### How It Works

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

### Key Features

- **MC Dropout**: Enables uncertainty quantification during inference
- **Entropy-based scoring**: Measures prediction confidence
- **Batch processing**: Efficient evaluation of large datasets

---

## 🎨 Method 2: VAE-Based OOD Detection

### Architecture

- **Model**: Bayesian Variational Autoencoder (VAE)
- **Technique**: Reconstruction error + uncertainty estimation
- **Detection Metric**: Anomaly score (MSE + variance)

### How It Works

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

### Key Features

- **Reconstruction-based**: Learns the distribution of ID data
- **Bayesian uncertainty**: Quantifies model uncertainty
- **H100 optimized**: Uses torch.compile and BF16 precision

---

## 🐳 Docker Setup

The system uses Docker containers for isolated execution environments.

### Services

1. **Classifier Container** (`docker-compose.yml` → `classifier`):
   - Image: `animals-classifier:v1`
   - Ports: 8889 (Jupyter), 6006 (TensorBoard)
   - Used for: Classifier training and evaluation

2. **VAE Container** (`docker-compose.yml` → `vae`):
   - Image: `ood-vae:h100`
   - Ports: 8888 (Jupyter)
   - Used for: VAE training and evaluation
   - Optimized for H100 GPU

### Volume Mounts

All containers share the same volume structure:
- `./src` → `/app/src` (source code)
- `./data` → `/app/data` (datasets)
- `./models` → `/app/models` (trained models)
- `./results` → `/app/results` (evaluation results)

---

## 🚀 Usage Workflow

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

## 🔧 Key Configuration Parameters

### Classifier Method
- `NUM_MC_SAMPLES = 30`: Number of forward passes for uncertainty estimation
- `ENTROPY_THRESHOLD = 0.6`: OOD detection threshold
- `BATCH_SIZE = 64`: Evaluation batch size
- `NUM_EPOCHS = 10`: Training epochs

### VAE Method
- `ANOMALY_THRESHOLD = 0.025`: OOD detection threshold
- `BATCH_SIZE = 256`: Training batch size
- `NUM_EPOCHS = 50`: Training epochs
- `latent_dim = 128`: Latent space dimensionality

---

## 🎯 Comparison of Methods

| Aspect | Classifier Method | VAE Method |
|--------|------------------|------------|
| **Approach** | Discriminative | Generative |
| **Detection** | Entropy (uncertainty) | Reconstruction error |
| **Training** | Faster (10 epochs) | Slower (50 epochs) |
| **Inference** | 30 forward passes | 30 reconstructions |
| **Interpretability** | Class probabilities | Visual reconstruction |
| **Use Case** | When you have labels | When you only have ID data |

---

## 📝 Notes

- Both methods use **Monte Carlo sampling** (30 samples) for uncertainty estimation
- Results are automatically organized into `run_X` folders to track multiple experiments
- Images are copied to `sorted_images/` folders for visual inspection
- The system is optimized for GPU execution (CUDA)
- VAE method is specifically optimized for H100 GPUs with BF16 precision

---

## 🔍 Understanding the Results

### Classifier Method
- **Low entropy** (< 0.6): Model is confident → ID
- **High entropy** (> 0.6): Model is uncertain → OOD

### VAE Method
- **Low anomaly score** (< 0.025): Good reconstruction → ID
- **High anomaly score** (> 0.025): Poor reconstruction → OOD

### Visualization
The histogram plots show the distribution of scores for ID and OOD samples. A good OOD detector should show:
- Clear separation between ID and OOD distributions
- ID samples clustered at low scores
- OOD samples spread at high scores

---

## 🛠️ Troubleshooting

1. **Model not found**: Ensure training scripts have been run first
2. **CUDA out of memory**: Reduce batch size in evaluation scripts
3. **No data found**: Check that datasets are extracted in `data/` directory
4. **Container issues**: Use `docker-compose logs` to check container status

---

## 📚 References

- **MC Dropout**: Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation
- **VAE**: Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes
- **ResNet**: He, K., et al. (2016). Deep Residual Learning for Image Recognition

---

## 🎓 Learning Resources

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

---

*Last Updated: System documentation for OOD Detection with Animals-10 and Pokemon datasets*
