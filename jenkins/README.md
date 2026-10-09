# Jenkins setup

GitHub Actions is the main CI for this repo (`.github/workflows`). Each project also has a
`Jenkinsfile` that runs the same checks (lint, tests, Docker build) on a Jenkins server.

## Start Jenkins locally

```bash
docker compose -f jenkins/docker-compose.yml up -d --build
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

Open http://localhost:8080, paste the password and choose "Install suggested plugins".
The image already contains the Git, Pipeline and Docker Pipeline plugins.

## Create a job per project

1. New Item -> **Multibranch Pipeline**.
2. Branch Sources -> Git -> `https://github.com/ImanDashtpeyma/Computer-vision.git`
   (public repo, no credentials needed).
3. Build Configuration -> Script Path:
   - `orange-pi-object-detection/Jenkinsfile`, or
   - `fastapi-pose-detection/Jenkinsfile`
4. Scan Multibranch Pipeline Triggers -> "Periodically if not otherwise run", 5 minutes.

Polling is used because a Jenkins running on a laptop can't receive GitHub webhooks. To use a
webhook instead, expose Jenkins with a tunnel (ngrok, cloudflared) and add it under
Settings -> Webhooks in the GitHub repo.

## What each pipeline does

1. **Lint and test**: inside a `python:3.11` container, create a venv, install
   `requirements-dev.txt`, run `ruff check .` and `pytest -q`.
2. **Docker build**: build the project's image on the Jenkins host's Docker.

## Notes

- Mounting `/var/run/docker.sock` gives Jenkins control of the host's Docker. That is fine for a
  personal machine and not something to copy onto shared infrastructure.
- The pipelines have been checked command by command outside Jenkins; their first full run in a
  real Jenkins is still to be done.
