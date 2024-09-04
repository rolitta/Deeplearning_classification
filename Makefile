DOCKER_BUILDKIT=1
AWS_PROFILE ?= default

auth-image:
	aws ecr get-login-password --region ca-central-1 --profile $(AWS_PROFILE) | docker login --username AWS --password-stdin 763104351884.dkr.ecr.ca-central-1.amazonaws.com

run-local:
	$(MAKE) auth-image
	python sagemaker_pipeline.py

deploy:
	$(MAKE) auth-image
	cd cdk && cdk deploy

lint:
	python -m black ./