# Docker self-hosted runner

This runs one Linux GitHub Actions self-hosted runner in Docker. The runner talks
to a Docker-in-Docker daemon, so workflow steps such as the ROCm `docker run`
build can bind-mount `$GITHUB_WORKSPACE` safely.

## Start

1. In GitHub, open the repository settings and create a new self-hosted runner
   registration token for Linux x64.
2. Copy `.env.example` to `.env`.
3. Set:
   - `GITHUB_URL=https://github.com/<owner>/<repo>`
   - `RUNNER_TOKEN=<registration token>`
4. Start the runner:

```bash
docker compose up -d --build
```

The runner registers with the `djl-linux-docker` label. The `Build JNI` workflow
uses that label for Linux jobs.

## Operations

Check logs:

```bash
docker compose logs -f runner
```

Stop without deleting runner state or Docker image cache:

```bash
docker compose down
```

Fully reset the registration and cached Docker images:

```bash
docker compose down -v
```

After a full reset, create a fresh GitHub runner registration token and update
`.env` before starting again.

The Docker-in-Docker cache persists in the `docker-data` volume. To reclaim disk
space while keeping the runner registered:

```bash
docker compose exec docker docker system prune -af
```

## Security

This runner has access to a privileged Docker daemon. Use it only for trusted
workflows and trusted branches/tags. Do not route untrusted pull request code
from forks to this runner.
