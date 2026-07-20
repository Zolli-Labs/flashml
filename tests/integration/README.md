# Integration tests

Everything in this folder needs **external infrastructure** — Docker, a
Kubernetes cluster, or a MinIO/S3 endpoint. Nothing here runs in the default
test invocation, and nothing in the library itself imports any of it: the
core `flashruntime` package stays pure Python (pydantic only).

```bash
pytest                     # unit tests only (default: -m "not integration")
pytest -m integration      # run these — with the environment below available
pytest -m integration tests/integration/test_minio_artifact_store.py
```

Tests **skip themselves** (with a reason) when their environment is absent,
so `pytest -m integration` is always safe to run — it does what it can and
tells you what it couldn't.

## Environments

| Requirement | Provided by | Detected via |
|---|---|---|
| Docker daemon | Docker Desktop / colima | `docker info` succeeds |
| Kubernetes + KubeRay | workspace `make poc-local-up` (kind cluster) or any cluster with the KubeRay operator | `kubectl` reaches a context; `rayjobs.ray.io` CRD exists |
| MinIO / S3 endpoint | workspace `make poc-local-up` + `make poc-local-forward`, or any S3-compatible store | `FLASHML_TEST_MINIO_ENDPOINT` (+ `_ACCESS_KEY`, `_SECRET_KEY`) env vars set |

The local kind + MinIO + KubeRay stack is owned by the **workspace**, not
this repo (`Makefile` at the workspace root, manifests in
`flashml-cloud/infra/local/`). Image builds live in `deploy/docker/`. This
separation is deliberate: the library is `pip install`-clean; environments
are test/deployment concerns.

## Writing a new integration test

1. Put it in this folder; the `conftest.py` here applies the `integration`
   marker automatically.
2. Gate it on one of the fixtures (`docker_available`,
   `kubernetes_available`, `minio_config`) so it skips gracefully.
3. Never bake credentials in — read env vars, document them in the table
   above.
