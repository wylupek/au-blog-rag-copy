PROJECT_ID := au-blog-rag
SERVICE_NAME := slack-rag-bot
CLIENT_EMAIL := au-blog-rag@au-blog-rag.iam.gserviceaccount.com
REGION := us-central1
IMAGE := gcr.io/$(PROJECT_ID)/$(SERVICE_NAME)
KEY_FILE := au-blog-rag-b18c9ebcbeeb.json
PLATFORM := linux/amd64,linux/arm64  # Specify target platforms for multi-architecture

.PHONY: all build push deploy clean

all: build push deploy

# Set up Google Cloud authentication and configuration
setup-gcloud:
	@echo "Setting up Google Cloud configuration..."
	gcloud config set project $(PROJECT_ID)
	gcloud auth activate-service-account $(CLIENT_EMAIL) --key-file $(KEY_FILE)

# Build the Docker image using Buildx
build:
	@echo "Building the Docker image using Buildx..."
	docker buildx build \
		--platform $(PLATFORM) \
		-t $(IMAGE):latest \
		--push \
		.

# Push the Docker image (not needed if --push is used in build)
push:
	@echo "Pushing the Docker image..."
	docker push $(IMAGE):latest

deploy: 
	@echo "Deploying to Google Cloud Run..."
	@env_vars=$(shell grep -v '^#' .env | xargs | sed 's/ /,/g') && \
	gcloud run deploy $(SERVICE_NAME) \
		--image $(IMAGE):latest \
		--region $(REGION) \
		--platform managed \
		--allow-unauthenticated \
		--update-env-vars $$env_vars

# Run the container locally
run-local:
	@echo "Running the Docker container locally..."
	docker run -it -p 8080:8080 --env-file .env $(IMAGE):latest

# Clean up Docker artifacts
clean:
	@echo "Cleaning up Docker artifacts..."
	docker system prune -f
