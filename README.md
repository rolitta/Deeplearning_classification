# ds_templates


## This template is still in development, and has a lot of room for improvement!!

Currently the repository is set up as a simple working example of training a Pytorch model on the MNIST dataset. The goal is to give the framework for how to set up files and where to place the logic in order to train a deep learning model using both Sagemaker and PyTorch Lightning, along with an AWS CDK stack to deploy the Sagemaker pipeline into the cloud for a productionized system.

One missing piece you'll find is that there is not any inference mechanisms in the current setup. This is something I hope to include in the near future, but the main blocker is that setting up inference endpoints in Sagemaker has a cost associated, and so far we have not needed to invoke that cost.

# High Level Organization

The makefile provides recipes for all the main tasks you might need to do on a regular basis (run locally, deploy to AWS, run the linter, etc.)

The file [sagemaker_pipeline.py](./sagemaker_pipeline.py) contains the definition for the Sagemaker Pipeline, and all other logic is wrapped into this. When we run the pipeline locally or in the cloud, we do it using this as the entrypoint.

The [cdk](./cdk/) folder contains the AWS CDK stack for deploying a pipeline into an AWS environment. This is not entirely stand alone because the stack uses the AWS parameter store in the `spk-data` AWS account to avoid hard coding the AWS acount ID and the Sagemaker IAM Role.

The folder named [code](./code/) contains everything needed to train the model through PyTorch Lightning. Note that in order for Sagemaker to deploy the code and set up the necessary modules inside of a Sagemaker docker image, the [requirements.txt](./code/requirements.txt) file is located within the code folder. 

The main entrypoint for training the model inside of the code folder is located in the [run.py](./code/run.py) file.

# What to change for a new model

If you are familiar with how PyTorch Lightning works you should be able to pick up the general layout of the files inside of the [code](./code/) folder. The main idea is that we have a model defined in Lightning format in the [model.py](./code/models/model.py) file, and a Lightning DataModule defined in the [datamodule.py](./code/datasets/datamodule.py) file. These two files contain most of what you need to change when implementing a new model trained on a new dataset.

The actual glue holding together the training logic can be found in the [train.py](./code/train/train.py) file.

In order to test that your PyTorch Lightning implementation works as expected you may need to make some changes to the [sagemaker_pipeline.py](./sagemaker_pipeline.py) file in order to pass any new hyperparameters, to handle customized arguments to the training script, add a preprocessing step to the Sagemaker pipeline, or change some of the execution parameters (docker image tag for example).

# How to run this locally

The setup for Sagemaker is minimal, but there is a little work you may have to do:

1. Make sure docker is installed and running on your machine.
1. Make sure you have the AWS CLI runnign on your machine and you have admin permissions set up for the `spk-data` account.
1. Authorize your AWS client.
    ```
    aws sso login
    ```
1. Run the pipeline locally:
    ```
    make run-local
    ```

    The first time you do this it will take a while to download the sagemaker docker image (it is very large ... >5GB).

# Deploy

Deploy the pipeline to the Sagemaker Pipeline service, where it can be run and monitoried through the UI using more compute resources than you probably have available locally.

Use [these](./cdk/README.md) instructions for deploying using the makefile recipe.