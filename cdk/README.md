
# CDk Deploy pipeline

The Sagemaker pipeline is set up to be deployed through AWS CDK. The app contains all logic for training and making inferences, and the CDK app (located here in this `cdk` folder) sets things up to run as a Sagemaker Pipeline.

**Important Note:**

The CDK app only deploys a Sagemaker pipeline for training the model, but does not kick off training. Once the deploy is finished, you can access the Sagemaker pipeline in Sagemaker Studio, and execute a training run from there.

## To deploy from your local machine:

1. Set up CDK if you have not done so already. Follow the [AWS instructions](https://docs.aws.amazon.com/cdk/v2/guide/getting_started.html).

1. Authenticate to AWS 
    ```
    aws sso login
    ```

1. Set the environment variables:

    ```
    export CDK_DEPLOY_ACCOUNT = <AWS account ID>
    export CDK_DEPLOY_REGION = <AWS region>
    ```

1. Deploy the cdk app

    This needs to be done from the root folder of this repository, since the makefile contains the instructions for deploying

    ```
    make deploy
    ```

Note that deploying through github actions is not currently possible, because the docker image used for training is too large to build there. Future work to try and find a more slim docker image that Sagemaker can still use.