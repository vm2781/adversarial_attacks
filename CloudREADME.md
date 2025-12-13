# Cloud Adversarial Attack Evaluation System

A serverless system that generates adversarial examples against neural networks and tests their transferability to Google's Gemini 2.0 Flash vision model. Inside folder `cloud_attacks`

## What It Does

1. Takes a source model (e.g., LeNet trained on MNIST)
2. Generates adversarial examples using PGD, MIFGSM, or Pixle attacks
3. Filters for examples that successfully fool the source model
4. Sends those adversarial images to Gemini 2.0 Flash for classification
5. Reports how accurately Gemini classifies the adversarial examples

This measures whether adversarial attacks crafted for small CNNs transfer to large commercial vision models.

## Architecture
```
User (curl request)
        │
        ▼
Google Cloud Run (Docker container)
        │
        ├── Loads source model
        ├── Generates adversarial examples
        ├── Filters for successful attacks
        │
        ▼
Google Gemini API (classifies images)
        │
        ▼
JSON response with accuracy metrics
```

## Setup & Deployment

### Prerequisites
- Google Cloud account with billing enabled
- gcloud CLI installed
- Gemini API key from https://aistudio.google.com/app/apikey

### 1. Clone and prepare
```bash
git clone <your-repo>
cd adversarial-attack-cloud
```

### 2. Add your model weights (OPTIONAL)

Place your trained `.pth` files in `app/model_weights/`. There should already be model weights for SqueezeNet and LeNet either trained regularly, or trained to be robust on the 3 attack types.


### 3. Enable GCP services
```bash
gcloud config set project YOUR-PROJECT-ID
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

gcloud artifacts repositories create mnist-repo \
    --repository-format=docker \
    --location=us-central1
```

### 4. Build and deploy
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR-PROJECT-ID/mnist-repo/adversarial-mnist:latest

gcloud run deploy adversarial-mnist \
    --image=us-central1-docker.pkg.dev/YOUR-PROJECT-ID/mnist-repo/adversarial-mnist:latest \
    --region=us-central1 \
    --platform=managed \
    --allow-unauthenticated \
    --set-env-vars=GEMINI_API_KEY=YOUR_GEMINI_API_KEY \
    --cpu=4 \
    --memory=16Gi \
    --timeout=600
```

## Usage

### MNIST Evaluation
```bash
curl -X POST https://YOUR-SERVICE-URL/generate \
    -H "Content-Type: application/json" \
    -d '{"source_model": "lenet", "attack_type": "pgd", "num_examples": 50}'
```

**Parameters:**
- `source_model`: lenet (STANDARD MODEL), squeezenet (STANDARD MODEL), lenet_pgd, squeezenet_pgd, lenet_mifgsm, squeezenet_mifgsm, lenet_pixle, squeezenet_pixle
- `attack_type`: pgd, mifgsm, pixle
- `num_examples`: number of adversarial examples to test (default: 20)

### ImageNette Evaluation
```bash
curl -X POST https://YOUR-SERVICE-URL/imagenette \
    -H "Content-Type: application/json" \
    -d '{"attack_type": "pgd", "num_examples": 50}'
```

Uses pretrained SqueezeNet1.1 as the source model.

## Example Response
```json
{
  "source_model": "lenet",
  "attack_type": "pgd",
  "vision_model": "gemini-2.0-flash",
  "total_tested": 50,
  "correct": 46,
  "accuracy": 0.92,
  "results": [
    {"index": 0, "true_label": 7, "predicted": 7, "correct": true},
    {"index": 1, "true_label": 2, "predicted": 3, "correct": false},
    ...
  ]
}
```

## Batch Testing
Use `run_tests.sh` to generate MNIST adversaries and evaluate Gemini, or use `imagenet_run.sh` to generate Imagenette adversaries and evaluate Gemini.

## Managing the Service

### Shut down (block public access)
```bash
gcloud run services update adversarial-mnist --region=us-central1 --ingress=internal
```

### Restart (restore public access)
```bash
gcloud run services update adversarial-mnist --region=us-central1 --ingress=all
```

### Delete service
```bash
gcloud run services delete adversarial-mnist --region=us-central1
```

### View logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=adversarial-mnist" --limit=50 --format="table(timestamp,textPayload)"
```

## Cost

- **Cloud Run**: Free when idle, ~$0.00002 per request
- **Gemini API**: ~$0.003 per 100 images
- **Artifact Registry**: ~$0.10/GB/month for image storage

Running 100 examples costs less than $0.01.
