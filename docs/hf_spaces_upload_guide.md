# Hugging Face Upload Guide for OpsArena

## Quick Answer First
Yes, there is a standard Hugging Face way to upload and run this project, and then there are competition-specific rules layered on top.

- Standard part: create a Hugging Face Space (Docker SDK), push repository, configure secrets/variables, and verify the Space endpoint is live.
- Competition-specific part: strict inference script contract, strict stdout log format, exact env var names, and validator script behavior.

This guide covers both so you can deploy once and avoid repeated paid trial-and-error runs.

## What Is Standard vs Competition-Specific

### Standard Hugging Face Spaces
These are generic platform rules and behavior:
- A Space is a git repository that auto-builds on every push.
- Docker Spaces require a README YAML frontmatter with sdk set to docker.
- Runtime variables and secrets are configured in the Space Settings UI.
- The app must expose the configured app_port.
- Validation of app liveliness is usually done by hitting your Space URL.

### Competition-Specific
These are not universal HF rules and can differ by hackathon:
- The file must be named inference.py at repo root.
- Required env variable names are usually fixed by the challenge.
- Log lines must follow exactly the required START/STEP/END pattern.
- Their pre-validation script may require exact /reset response behavior and local docker build success.

## Source References Used
This guide is based on current public Hugging Face docs and OpenEnv package docs:
- Spaces overview and deployment behavior
- Docker Spaces setup and secrets/variables handling
- Spaces config reference (README frontmatter)
- Spaces sync via GitHub Actions
- OpenEnv CLI usage including openenv validate and openenv push

## Prerequisites Checklist
Before uploading, make sure you have:
- Hugging Face account
- Hugging Face user access token with write access to Spaces
- Docker installed and running locally
- Python environment with openenv command available
- A live repository with:
  - Dockerfile in repo root
  - openenv.yaml in repo root
  - root-level inference.py

## Recommended Deployment Strategy for OpsArena
Use Docker Space deployment, because this repo already has Dockerfile and OpenEnv server setup.

## Step-by-Step: Create and Upload Space (UI + Git)

### Step 1: Create the Space
1. Go to https://huggingface.co/spaces.
2. Click Create new Space.
3. Pick owner (your user or org).
4. Set Space name.
5. Choose SDK: Docker.
6. Choose visibility (public/private depending on competition requirement).
7. Create Space.

### Step 2: Set README Frontmatter for Docker
At repo root README, ensure YAML frontmatter includes at least:
- sdk: docker
- app_port: whichever your app serves (for OpsArena commonly 8000 if your container serves that port)

If app_port in README and the port your container serves do not match, Space will boot incorrectly.

### Step 3: Configure Secrets and Variables in Space Settings
In Space Settings:
- Add secret: HF_TOKEN
- Add variable: API_BASE_URL
- Add variable: MODEL_NAME
- Add variable if needed: LOCAL_IMAGE_NAME

Important:
- Keep HF_TOKEN as secret, not plain variable.
- Variables and secrets become runtime env vars in Docker Spaces.

### Step 4: Push Code to Space
Option A: direct git remote
1. Add remote:
   git remote add space https://huggingface.co/spaces/<HF_USERNAME>/<SPACE_NAME>
2. First sync push:
   git push --force space main
3. Subsequent pushes:
   git push space main

Option B: GitHub Actions sync
- Configure workflow that pushes main branch to Space with HF_TOKEN in GitHub secrets.
- Use LFS if any file exceeds 10MB.

### Step 5: Watch Build Logs
In Space page:
- Open Logs tab
- Confirm image builds successfully
- Confirm app starts and listens on configured app_port

### Step 6: Validate Endpoints
Once running, test Space URL and challenge-required endpoint(s), commonly:
- POST https://<space>.hf.space/reset

If this endpoint does not return expected status, the pre-validation script will fail at Step 1.

## Step-by-Step: Local Pre-Validation Before Paid Runs
Run in repo root:
1. openenv validate
2. docker build .
3. bash scripts/validate-submission.sh https://<your-space>.hf.space .

Expected script behavior:
- Step 1 checks Space /reset returns HTTP 200
- Step 2 checks Docker build locally
- Step 3 checks openenv validate locally

If any step fails, it stops immediately.

## How This Maps to Your Current Repo
Current repo already has:
- Dockerfile
- openenv.yaml
- root inference.py
- pre-validation script under scripts

So your remaining work is mostly deployment wiring:
- Create Space
- Set secrets/variables
- Push code
- Run validator against real Space URL

## Competition Strictness Notes
For challenge submissions, assume exactness matters:
- Inference file name and location must match exactly.
- Env var names must match exactly.
- START/STEP/END line shape should match exactly.
- Score output should be normalized if required by challenge.

Do not rely on “close enough” formatting when validator is strict.

## Common Failure Modes and Fixes

### Failure: Missing HF_TOKEN at runtime
Cause:
- Secret not configured in Space.
Fix:
- Add HF_TOKEN under Space Settings > Secrets.

### Failure: /reset not returning 200
Cause:
- App not started, wrong route, wrong base path, or boot failed.
Fix:
- Check Space logs, confirm server startup and route exposure.

### Failure: Docker build fails in validator
Cause:
- Build depends on unavailable system packages or wrong context.
Fix:
- Build locally first with docker build . and fix image errors before push.

### Failure: openenv validate fails
Cause:
- openenv.yaml mismatch, app import path mismatch, or metadata inconsistency.
Fix:
- Run openenv validate locally and fix reported issues before deployment.

### Failure: Logs rejected by competition parser
Cause:
- Extra fields, different key names, wrong boolean casing, wrong score formatting.
Fix:
- Keep exactly required START/STEP/END schema.

## Recommended Cost-Control Loop
Because paid platform runs are expensive:
1. Validate locally first (openenv validate + docker build).
2. Dry run inference locally with real env vars.
3. Deploy to Space.
4. Run pre-validation script with real Space URL.
5. Submit only after all checks pass.

## If You Share Competition Video Transcript
If you send the transcript, I can produce a second guide that is competition-specific and line-by-line mapped to the organizer’s exact wording.

That will include:
- Exact required defaults
- Exact output grammar and examples
- Exact failure interpretation for each validator step
- Exact final submission sequence

## One-Page Execution Plan
- Create Docker Space
- Configure HF_TOKEN, API_BASE_URL, MODEL_NAME, optional LOCAL_IMAGE_NAME
- Push repo
- Confirm logs and /reset route
- Run scripts/validate-submission.sh with real Space URL
- Fix any failed step
- Submit
