# Docker Usage Guide

Complete guide for using Docker containers in the OOD Detection System.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Container Overview](#container-overview)
3. [Initial Setup](#initial-setup)
4. [Container Management](#container-management)
5. [Running Commands](#running-commands)
6. [Volume Mounts](#volume-mounts)
7. [Ports and Services](#ports-and-services)
8. [Common Workflows](#common-workflows)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### Required Software

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **NVIDIA Docker Runtime**: For GPU support (nvidia-docker2)
- **NVIDIA GPU**: With CUDA support (for training/evaluation)

### Verify Installation

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

---

## Container Overview

The system uses two Docker containers:

### 1. Classifier Container
- **Container Name**: `animals_classifier_container`
- **Image**: `animals-classifier:v1`
- **Purpose**: Classifier training and evaluation
- **Base Image**: `nvcr.io/nvidia/pytorch:23.10-py3`
- **Ports**: 
  - `8889:8888` (Jupyter Lab)
  - `6006:6006` (TensorBoard)

### 2. VAE Container
- **Container Name**: `ood_vae_container`
- **Image**: `ood-vae:h100`
- **Purpose**: VAE training and evaluation
- **Base Image**: `nvcr.io/nvidia/pytorch:23.10-py3`
- **Ports**: 
  - `8888:8888` (Jupyter Lab)
- **Optimization**: H100 GPU optimized with BF16 support

---

## Initial Setup

### Step 1: Build Docker Images

Build both containers from the project root:

```bash
# Build all containers
docker-compose build

# Or build individually
docker-compose build classifier
docker-compose build vae
```

### Step 2: Start Containers

```bash
# Start containers in detached mode
docker-compose up -d

# View container status
docker-compose ps
```

Expected output:
```
NAME                          STATUS
animals_classifier_container  Up
ood_vae_container             Up
```

### Step 3: Verify Containers

```bash
# Check if containers are running
docker ps

# View container logs
docker-compose logs classifier
docker-compose logs vae
```

---

## Container Management

### Starting Containers

```bash
# Start all containers
docker-compose up -d

# Start specific container
docker-compose up -d classifier
docker-compose up -d vae
```

### Stopping Containers

```bash
# Stop all containers
docker-compose down

# Stop without removing volumes
docker-compose stop

# Stop specific container
docker-compose stop classifier
```

### Restarting Containers

```bash
# Restart all containers
docker-compose restart

# Restart specific container
docker-compose restart classifier
```

### Removing Containers

```bash
# Remove containers (keeps volumes)
docker-compose down

# Remove containers and volumes (WARNING: deletes data)
docker-compose down -v
```

### Viewing Logs

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

---

## Running Commands

### Interactive Shell Access

#### Classifier Container

```bash
# Enter interactive bash shell
docker exec -it animals_classifier_container bash

# Once inside, you're in /app directory
cd /app/src/Animals-10/classifier
```

#### VAE Container

```bash
# Enter interactive bash shell
docker exec -it ood_vae_container bash

# Once inside, you're in /app directory
cd /app/src/Animals-10/vae
```

### Running Python Scripts

#### From Host (without entering container)

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

#### From Inside Container

```bash
# Enter container first
docker exec -it animals_classifier_container bash

# Then run scripts
cd /app/src/Animals-10/classifier
python train.py
python evaluate_ood.py
python detect_ood.py --image /app/data/pokemon/unknown/image.jpg
```

### Running with GPU

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

---

## Volume Mounts

The containers use volume mounts to share data between host and containers:

### Volume Mapping

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./src` | `/app/src` | Source code |
| `./data` | `/app/data` | Datasets |
| `./models` | `/app/models` | Trained models |
| `./results` | `/app/results` | Evaluation results |

### Accessing Files

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

### Important Notes

- **Real-time Sync**: Changes in host directories are immediately visible in containers
- **No Copy Needed**: Files are shared, not copied
- **Persistent Storage**: Data persists even after container removal (unless using `-v` flag)

---

## Ports and Services

### Port Mapping

| Container | Host Port | Container Port | Service |
|-----------|-----------|----------------|---------|
| Classifier | 8889 | 8888 | Jupyter Lab |
| Classifier | 6006 | 6006 | TensorBoard |
| VAE | 8888 | 8888 | Jupyter Lab |

### Accessing Services

#### Jupyter Lab (Classifier)

```bash
# Access at: http://localhost:8889
# Default password/token: Check container logs
docker-compose logs classifier | grep token
```

#### Jupyter Lab (VAE)

```bash
# Access at: http://localhost:8888
# Default password/token: Check container logs
docker-compose logs vae | grep token
```

#### TensorBoard (Classifier)

```bash
# Start TensorBoard inside container
docker exec -it animals_classifier_container \
  tensorboard --logdir=/app/results --port=6006 --host=0.0.0.0

# Access at: http://localhost:6006
```

---

## Common Workflows

### Workflow 1: Complete Training and Evaluation

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

### Workflow 2: Interactive Development

```bash
# 1. Start containers
docker-compose up -d

# 2. Enter classifier container
docker exec -it animals_classifier_container bash

# 3. Inside container, navigate and work
cd /app/src/Animals-10/classifier
python train.py  # Edit code on host, run in container
```

### Workflow 3: Single Image Testing

```bash
# Test single image with classifier
docker exec -it animals_classifier_container \
  python /app/src/Animals-10/classifier/detect_ood.py \
  --image /app/data/pokemon/unknown/pikachu.jpg
```

### Workflow 4: Monitoring Training

```bash
# Terminal 1: Start training
docker exec -it animals_classifier_container \
  python /app/src/Animals-10/classifier/train.py

# Terminal 2: Monitor logs
docker-compose logs -f classifier

# Terminal 3: Check GPU usage
watch -n 1 docker exec animals_classifier_container nvidia-smi
```

---

## Troubleshooting

### Issue 1: Container Won't Start

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

### Issue 2: GPU Not Available

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

### Issue 3: Permission Denied

**Symptoms**: Cannot write to mounted volumes

**Solutions**:
```bash
# Check file permissions
ls -la ./models
ls -la ./results

# Fix permissions (if needed)
sudo chown -R $USER:$USER ./models ./results

# Or run container with user mapping
# Add to docker-compose.yml:
# user: "${UID}:${GID}"
```

### Issue 4: Out of Memory

**Symptoms**: CUDA out of memory errors

**Solutions**:
```bash
# Reduce batch size in training scripts
# Edit: src/Animals-10/classifier/train.py
# Change: BATCH_SIZE = 32  # Reduce from 64

# Or limit GPU memory
# Add to docker-compose.yml:
# environment:
#   - CUDA_VISIBLE_DEVICES=0
```

### Issue 5: Module Not Found

**Symptoms**: `ImportError: No module named 'X'`

**Solutions**:
```bash
# Install missing package in container
docker exec -it animals_classifier_container pip install package_name

# Or rebuild container with new dependencies
# Edit Dockerfile, then:
docker-compose build classifier
```

### Issue 6: Files Not Visible

**Symptoms**: Files created in container not visible on host

**Solutions**:
```bash
# Verify volume mounts
docker inspect animals_classifier_container | grep Mounts

# Check if files are in correct path
# Container: /app/results/...
# Host: ./results/...
```

---

## Advanced Usage

### Custom Environment Variables

Add to `docker-compose.yml`:

```yaml
services:
  classifier:
    environment:
      - CUSTOM_VAR=value
      - PYTHONPATH=/app/src
```

### Running Multiple Experiments

```bash
# Start multiple containers with different configurations
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

### Resource Limits

Add to `docker-compose.yml`:

```yaml
services:
  classifier:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Backup and Restore

```bash
# Backup models
tar -czf models_backup.tar.gz ./models

# Backup results
tar -czf results_backup.tar.gz ./results

# Restore
tar -xzf models_backup.tar.gz
```

### Clean Up

```bash
# Remove stopped containers
docker-compose rm

# Remove unused images
docker image prune -a

# Remove all unused resources
docker system prune -a
```

### Debugging

```bash
# Enter container with root privileges
docker exec -it --user root animals_classifier_container bash

# Install debugging tools
docker exec -it animals_classifier_container \
  apt-get update && apt-get install -y vim htop

# Check container resource usage
docker stats animals_classifier_container
```

---

## Quick Reference

### Essential Commands

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

### File Paths Reference

| Task | Host Path | Container Path |
|------|-----------|----------------|
| Edit code | `./src/...` | `/app/src/...` |
| Add data | `./data/...` | `/app/data/...` |
| Check models | `./models/...` | `/app/models/...` |
| View results | `./results/...` | `/app/results/...` |

---

## Additional Resources

- **Docker Documentation**: https://docs.docker.com/
- **Docker Compose Documentation**: https://docs.docker.com/compose/
- **NVIDIA Container Toolkit**: https://github.com/NVIDIA/nvidia-docker

---

*Last Updated: Docker usage guide for OOD Detection System*


