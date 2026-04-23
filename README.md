# ArgoCD Project

## Overview
This repository contains two main projects: **AI-emotion** and **AI-koder**, both focused on generative AI applications.

---

## AI-emotion
A project for emotion classification using a lightweight model. It includes:
- **train.py**: Script to train the model on synthetic or small datasets.
- **model_wrapper.py**: Handles model loading and preprocessing.
- **gradio_app.py**: Web interface for user interaction.

### Build & Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/softaav325/argo.git
   ```
2. Navigate to the project directory:  
   ```bash
   cd AI-emotion
   ```
3. Build the Docker image:  
   ```bash
   docker build -t demo:latest .
   ```

---

## AI-koder
A generative AI project for code or poetry generation in a specific style. Features:
- **train.py**: Trains a small neural network for text generation.
- **model_wrapper.py**: Manages model architecture and data handling.
- **gradio_app.py**: Web interface for input/output.

### Build & Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/softaav325/argo.git
   ```
2. Navigate to the project directory:  
   ```bash
   cd AI-koder
   ```
3. Build the Docker image:  
   ```bash
   docker build -t demo:latest .
   ```

---

## Common Requirements
- **Docker**: Required for local builds.
- **Python**: Ensure dependencies are installed via `requirements-build.txt` and `requirements-run.txt`.

---

## Deployment Options with Argo CD

### Option 1: Deploy via Helm Charts
1. **Install Argo CD with Helm**:
   ```bash
   helm repo add argo https://argoproj.github.io/argo-helm
   helm repo update
   helm install argocd argo/argo-cd --namespace argocd --set controller.metrics.enabled=true
   ```
2. **Deploy Applications via Helm**:
   - If Helm charts for AI-emotion/AI-koder exist, use them:
     ```bash
     kubectl apply -f argocd/applications/demo-app.yaml
     ```
   - If no charts exist, create them from existing manifests (see Option 2).

### Option 2: Deploy via Kubernetes Manifests
1. **Use Existing Manifests**:
   - The `k8s/deployment.yaml` and `k8s/service.yml` files define Kubernetes resources.
2. **Add Manifests to Argo CD**:
   - Commit the manifests to the repository.
   - Argo CD will automatically sync them to the cluster.
3. **Apply Manifests**:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yml
   ```
   - Alternatively, let Argo CD manage the sync via its UI or CLI.

---

## Common Requirements
- **Docker**: Required for local builds.
- **Kubernetes**: Cluster must be configured with Argo CD.
- **Helm**: Optional for Option 1.

## Contributing
- Fork the repository.
- Submit pull requests for improvements or new features.

## License
MIT License