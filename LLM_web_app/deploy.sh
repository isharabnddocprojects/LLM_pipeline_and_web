#!/bin/bash
# Deployment script for LLM Web App on Google Cloud Run

set -euo pipefail

# Configuration (override via env vars)
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-llm-web-app}"
SERVICE_NAME="${SERVICE_NAME:-llm-web-app}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-llm-web-app-sa}"
ARTIFACT_LOCATION="${ARTIFACT_LOCATION:-${REGION}}"
ARTIFACT_HOST="${ARTIFACT_LOCATION}-docker.pkg.dev"
IMAGE_URI="${ARTIFACT_HOST}/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}:${IMAGE_TAG}"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "ERROR: PROJECT_ID is not set. Run 'gcloud config set project <PROJECT_ID>' or export PROJECT_ID."
  exit 1
fi

echo "=== Deploying LLM Web App to Google Cloud Run ==="
echo "Project      : ${PROJECT_ID}"
echo "Region       : ${REGION}"
echo "Repo Location: ${ARTIFACT_LOCATION}"
echo "Repository   : ${REPOSITORY}"
echo "Service      : ${SERVICE_NAME}"
echo "Image URI    : ${IMAGE_URI}"
echo ""

# Step 1: Enable required APIs
echo "Step 1: Enabling Google Cloud APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  --project "${PROJECT_ID}"

# Step 2: Create Artifact Registry repository if needed
echo ""
echo "Step 2: Ensuring Artifact Registry repo exists..."
if ! gcloud artifacts repositories describe "${REPOSITORY}" \
  --location="${ARTIFACT_LOCATION}" \
  --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${ARTIFACT_LOCATION}" \
    --description="Container images for ${SERVICE_NAME}" \
    --project "${PROJECT_ID}"
else
  echo "Repository already exists."
fi

# Step 3: Configure Docker authentication
echo ""
echo "Step 3: Configuring Docker auth for Artifact Registry..."
gcloud auth configure-docker "${ARTIFACT_HOST}" --project "${PROJECT_ID}"

# Step 4: Build Docker image
echo ""
echo "Step 4: Building Docker image..."
docker build -t "${IMAGE_URI}" .

# Step 5: Push Docker image
echo ""
echo "Step 5: Pushing Docker image..."
docker push "${IMAGE_URI}"

# Step 6: Ensure Cloud Run service account exists
echo ""
echo "Step 6: Ensuring service account exists..."
SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
    --display-name "${SERVICE_NAME} service account" \
    --project "${PROJECT_ID}"
fi

# Step 7: Deploy Cloud Run service
echo ""
echo "Step 7: Deploying Cloud Run service..."
TEMP_YAML=$(mktemp cloudrun_rendered_XXXXXX.yaml 2>/dev/null || echo cloudrun_rendered.yaml)
env IMAGE_URI="${IMAGE_URI}" PROJECT_ID="${PROJECT_ID}" envsubst < cloudrun.yaml > "${TEMP_YAML}"
gcloud run services replace "${TEMP_YAML}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}"
rm -f "${TEMP_YAML}"

# Step 8: Allow unauthenticated access
echo ""
echo "Step 8: Allowing unauthenticated access..."
gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
  --member="allUsers" \
  --role="roles/run.invoker" \
  --project "${PROJECT_ID}" \
  --region "${REGION}"

# Step 9: Output service URL
echo ""
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --platform managed \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format='value(status.url)')

echo "=== Deployment Complete ==="
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Tail logs:"
echo "gcloud logs tail --project ${PROJECT_ID} --region ${REGION} --service ${SERVICE_NAME}"
echo ""

