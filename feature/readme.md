## optinal: Feature Store: Run feast on Kubernetes

helm repo add feast-charts https://feast-helm-charts.storage.googleapis.com
helm repo update
helm install feast-release feast-charts/feast

See:
https://docs.feast.dev/v0.11-branch/feast-on-kubernetes/getting-started/install-feast/kubernetes-with-helm