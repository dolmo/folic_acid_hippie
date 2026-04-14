# /nrp-monitor - NRP Job Monitoring

Monitor Kubernetes jobs, check completion status, and diagnose failures.

## Usage

```
/nrp-monitor [--run-tag <tag>] [--status <filter>] [--logs <job>] [--diagnose <job>]
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--run-tag` | Filter jobs by run tag | All `jgf-*` jobs |
| `--status` | Status filter: `all`, `running`, `completed`, `failed` | `all` |
| `--logs` | Show logs for specific job/pod | - |
| `--diagnose` | Full diagnosis for a failed job | - |

## Monitoring Commands

### List All Jobs
```bash
kubectl get jobs | grep jgf-
```

### Filter by Status
```bash
# Running pods
kubectl get pods | grep -E "jgf-.*Running"

# Completed pods
kubectl get pods | grep -E "jgf-.*Completed"

# Failed/Error pods
kubectl get pods | grep -E "jgf-.*(Error|Failed|CrashLoopBackOff)"
```

### View Job Logs
```bash
# Get pod name for a job
kubectl get pods -l job-name=<job-name>

# Stream logs
kubectl logs -f <pod-name>

# Last 100 lines
kubectl logs --tail=100 <pod-name>
```

### Diagnose Failures
```bash
# Describe job (shows events, conditions)
kubectl describe job <job-name>

# Describe pod (shows events, container status)
kubectl describe pod <pod-name>

# Get previous container logs (for restarts)
kubectl logs --previous <pod-name>
```

## Common Failure Patterns

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `ImagePullBackOff` | Docker image not found | `/docker-build --push` |
| `OOMKilled` | Out of memory | Increase memory limits in job YAML |
| `ErrImagePull` | Registry auth issue | Check Docker Hub credentials |
| `CrashLoopBackOff` | Script error | Check logs with `kubectl logs` |
| `Pending` (long) | No resources available | Wait or reduce resource requests |
| `CreateContainerError` | Storage issue | Reduce ephemeral-storage request |

## Progress Tracking

For a multi-job benchmark run, track overall progress:

```bash
# Count jobs by status
echo "Running: $(kubectl get jobs | grep jgf- | grep -c '0/1')"
echo "Completed: $(kubectl get jobs | grep jgf- | grep -c '1/1')"
echo "Failed: $(kubectl get pods | grep jgf- | grep -cE 'Error|Failed')"

# List incomplete jobs
kubectl get jobs | grep jgf- | grep '0/1'
```

## Watch Mode

Continuously monitor job status:
```bash
watch -n 10 "kubectl get jobs | grep jgf- && echo '---' && kubectl get pods | grep jgf-"
```

## Cleanup

After a benchmark run:
```bash
# Delete completed jobs
kubectl delete jobs -l run-tag=<tag>

# Delete all jgf-* jobs (careful!)
kubectl delete jobs $(kubectl get jobs -o name | grep jgf-)
```

## Examples

```bash
# Check status of all jobs
/nrp-monitor --status all

# Show only running jobs
/nrp-monitor --status running

# View logs for specific job
/nrp-monitor --logs jgf-cps-hippie-hull-cell-type

# Full diagnosis of failed job
/nrp-monitor --diagnose jgf-cps-nemo-hausser-cell-ty

# Filter by run tag
/nrp-monitor --run-tag smoke-20260408T123456Z
```

## Related Skills

- `/nrp-deploy` - Deploy jobs
- `/s3-data` - Check results uploaded to S3
- `/aggregate-results` - Aggregate completed results
