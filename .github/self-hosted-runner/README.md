# Docker self-hosted runner

This runs Linux GitHub Actions self-hosted runners in Docker. The runners talk
to a shared Docker-in-Docker daemon, so workflow steps such as the ROCm
`docker run` build can bind-mount `$GITHUB_WORKSPACE` safely.

## Start

1. In GitHub, open the repository settings and create a new self-hosted runner
   registration token for Linux x64.
2. Copy `.env.example` to `.env`.
3. Set:
   - `GITHUB_URL=https://github.com/<owner>/<repo>`
   - optionally `RUNNER_VERSION=2.334.0`
4. Configure the runner with a short-lived token:

```bash
RUNNER_TOKEN=<registration token> RUNNER_CONFIG_ONLY=true docker compose run --rm runner1
RUNNER_TOKEN=<registration token> RUNNER_CONFIG_ONLY=true docker compose run --rm runner2
RUNNER_TOKEN=<registration token> RUNNER_CONFIG_ONLY=true docker compose run --rm runner3
```

5. Start the runner without keeping the token in the long-running container:

```bash
docker compose up -d --build
```

The Linux runners register with the `djl-linux-docker` label. The `Build JNI`
workflow uses that label for Linux jobs. `runner1`, `runner2`, and `runner3`
use independent runner state and work directories under
`/runner/runner1/_work`, `/runner/runner2/_work`, and `/runner/runner3/_work`,
while sharing one Docker image cache.

## Operations

Check logs:

```bash
docker compose logs -f runner1
docker compose logs -f runner2
docker compose logs -f runner3
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

## Windows Runner

Windows JNI builds use a normal Windows self-hosted runner, not Docker. A
physical NVIDIA GPU is not required because the workflow only compiles against
the CUDA toolkit and does not run GPU runtime tests.

Host requirements:

- Git for Windows
- Visual Studio 2022 Build Tools with the x64 C++ toolchain
- enough free disk for CUDA installers and libtorch archives

Register the runner with the `djl-windows` label:

```powershell
.\config.cmd --unattended `
  --url https://github.com/<owner>/<repo> `
  --token <registration token> `
  --name djl-rocm-win-$env:COMPUTERNAME `
  --labels djl-windows,windows `
  --work _work `
  --replace
```

Start it interactively:

```powershell
.\run.cmd
```

For durable operation, configure it as a Windows service with
`config.cmd --runasservice` from an elevated shell and use an appropriate service
account.
