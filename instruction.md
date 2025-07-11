# SF Compute Setup Instructions for temporal_llms

This guide explains how to build, deploy, and run training/inference jobs for the temporal_llms repo on SF Compute using Kubernetes.

---

## 1. Prerequisites
- Access to SF Compute cluster and kubectl
- Docker installed locally
- Access to a container registry (e.g., Docker Hub, SF registry)
- Persistent Volume Claim (PVC) set up on SF Compute for data storage

---

## 2. Build and Push Docker Image

1. **Build the Docker image:**
   ```bash
   docker build -t <YOUR_DOCKER_IMAGE> .
   ```
   Replace `<YOUR_DOCKER_IMAGE>` with your registry path, e.g., `docker.io/username/temporal-llms:latest`.

2. **Push the image to your registry:**
   ```bash
   docker push <YOUR_DOCKER_IMAGE>
   ```

---

## 3. Configure Persistent Storage

- Ensure a PersistentVolumeClaim (PVC) named `sf-pvc` exists on SF Compute. This will be mounted at `/app/data` in the container.
- If not, create one or ask your admin for the correct claim name.

---

## 4. Edit Kubernetes Job YAML

- Open `k8s_job.yaml` and set the correct Docker image:
  ```yaml
  image: <YOUR_DOCKER_IMAGE>
  ```
- Adjust resource requests/limits as needed.
- To run a different script, override the command in the YAML:
  ```yaml
  command: ["python", "scripts/openthoughts_sft_inference.py"]
  ```
  (Add this under the `containers` section.)

---

## 5. Submit Job to SF Compute

1. **Apply the job YAML:**
   ```bash
   kubectl apply -f k8s_job.yaml
   ```
2. **Check job status:**
   ```bash
   kubectl get jobs
   kubectl describe job temporal-llms-job
   kubectl logs job/temporal-llms-job
   ```

---

## 6. Running Training/Inference Scripts

- By default, the container runs `scripts/openthoughts_sft_training.py`.
- To run other scripts (e.g., inference), edit the `CMD` in Dockerfile or set the `command` in the YAML as shown above.
- Example for running inference:
  ```yaml
  command: ["python", "scripts/openthoughts_sft_inference.py"]
  ```

---

## 7. Accessing Results

- Output files will be saved in `/app/data` (mounted from PVC).
- Retrieve results from the PVC or use `kubectl cp` to copy files from the pod.

---

## 8. Troubleshooting

- Check logs for errors:
  ```bash
  kubectl logs job/temporal-llms-job
  ```
- Make sure all dependencies are listed in `requirements.txt`.
- Ensure PVC is correctly mounted and accessible.

---

## 9. Example: Run a Custom Script

To run `scripts/openthoughts_sft_inference_AIME2024.py`, edit the YAML:
```yaml
command: ["python", "scripts/openthoughts_sft_inference_AIME2024.py"]
```

---

## 10. Clean Up

- Delete completed jobs:
  ```bash
  kubectl delete job temporal-llms-job
  ```

---

## References
- SF Compute documentation
- Kubernetes Job API docs

---

For further help, contact your SF Compute admin or refer to the official documentation.
