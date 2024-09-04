class Settings:
    log_level = "INFO"

    pipeline_name = "template-pipeline-v0"

    gpu_instance: str = "ml.g5.xlarge"

    # using spot instances can save money, but can also mean longer wait times and many restarts
    use_spot_instances: bool = False
    max_run: int = 259200  # 3 days
    max_wait: int = 259200  # ignored if not using spot instances

    # location in the bucket of the input data used for training:
    input_data_path: str = "s3://sagemaker-ca-central-1-630933869751/landcover_semantic_segmentation_input_dataset/"
    bucket = "sagemaker-ca-central-1-630933869751"

    # training parameters (defaults)
    max_epochs = 26
    batch_size = 10
    amp: bool = True

    # ssm parameter names for lookup:
    # bucket name:
    ssm_bucket_param = "/sagemaker/bucket_name"
    # rolae ARN:
    ssm_role_param = "/sagemaker/role_arn"
    # TODO: tha above parameters should be stored automatically by the sagemaker domain stack when the bucket & role is created

    app_tags = [
        {"key": "DevelopedBy", "value": "Sparkgeo Consulting Inc"},
        {"key": "project", "value": "template-pipeline"},
    ]
