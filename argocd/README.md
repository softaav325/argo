## Install Argo CD

```kubectl create namespace argocd```

# Добавьте репозиторий
helm repo add argo https://argoproj.github.io/argo-helm
helm repo update

# Установите Argo CD с включением CRD
 helm install argocd argo/argo-cd --namespace argocd \
  --set controller.metrics.enabled=true

## Access The Argo CD API Server

```kubectl port-forward svc/argocd-server -n argocd 8080:443```

## Login Using The CLI

```kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo```




