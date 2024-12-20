GCP_PROJECT := au-blog-rag
GCP_SERVICE_ACCOUNT := au-blog-rag@au-blog-rag.iam.gserviceaccount.com
GCP_REGION := us-central1
BOT_SERVICE_NAME := slack-rag-bot
DATA_JOB := slack-rag-bot-embedings-job
KEY_FILE := au-blog-rag-b18c9ebcbeeb.json
PLATFORM := linux/amd64


.PHONY: all build push deploy clean

all: build push deploy

# Set up Google Cloud authentication and configuration
setup-gcloud:
	@echo "Setting up Google Cloud configuration..."
	gcloud config set project $(GCP_PROJECT)
	gcloud auth activate-service-account $(GCP_SERVICE_ACCOUNT) --key-file $(KEY_FILE)

deploy-bot:
	@docker buildx build --platform $(PLATFORM) \
		-t gcr.io/$(GCP_PROJECT)/$(BOT_SERVICE_NAME):latest \
		-f Dockerfile . --push

	@env_vars=$(shell grep -v '^#' .env | xargs | sed 's/ /,/g') && \
	gcloud run deploy $(BOT_SERVICE_NAME) \
		--image gcr.io/$(GCP_PROJECT)/$(BOT_SERVICE_NAME):latest \
		--region $(GCP_REGION) \
		--platform managed \
		--allow-unauthenticated \
		--update-env-vars $$env_vars

#
run-local-bot:
	@echo "Running the Docker container locally..."
	docker run -it -p 8080:8080 --env-file .env gcr.io/$(GCP_PROJECT)/$(BOT_SERVICE_NAME):latest


#
deploy-data: 
	@docker buildx build --platform $(PLATFORM) \
		-t gcr.io/$(GCP_PROJECT)/$(DATA_JOB):latest \
		-f processor/Dockerfile . --push
	@gcloud run jobs update $(DATA_JOB) \
		--image gcr.io/$(GCP_PROJECT)/$(DATA_JOB):latest \
		--region=$(GCP_REGION) \
		--service-account=$(GCP_SERVICE_ACCOUNT)

#
execute-data: 
	@gcloud run jobs execute $(DATA_JOB) --region=$(GCP_REGION) --wait


# Clean up Docker artifacts
clean:
	@echo "Cleaning up Docker artifacts..."
	docker system prune -f
