import os
import subprocess


import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterBoolean,
)
from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker import image_uris
from sagemaker.workflow.pipeline_context import PipelineSession, LocalPipelineSession

from cdk.settings import Settings


class SagemakerPipeline:
    """
    concrete class for deploying the solar panel model pipeline to sagemaker
    """

    def __init__(
        self,
        input_data_uri: str,
        output_uri: str,
        sm_execution_role_arn: str,
        region: str,
        local_mode: bool = False,
    ) -> None:
        self.local_mode = local_mode

        # spot instance and run time params
        self.use_spot_instances = Settings().use_spot_instances
        self.max_run = Settings().max_run  # 3 days
        self.max_wait = Settings().max_wait if Settings().use_spot_instances else None

        # URI for the docker image used for training
        self.model_image_uri = image_uris.retrieve(
            framework="pytorch",
            region=region,
            version="2.0.0",
            py_version="py310",
            image_scope="training",
            instance_type=Settings().gpu_instance,
        )

        self.input_data_uri = input_data_uri
        self.output_uri = output_uri
                # check for GPU:
        try:
            subprocess.check_output("nvidia-smi")
            gpu = True
        except Exception:
            gpu = False
        if self.local_mode & gpu:
            self.gpu_instance = "local_gpu"
        else:
            self.gpu_instance = Settings().gpu_instance

        self.sm_execution_role_arn = sm_execution_role_arn

        # format parameters to allow configuration through the UI and hyperparameter jobs
        self.max_epochs = ParameterInteger(
            name="max_epochs", default_value=Settings().max_epochs
        )
        self.batch_size = ParameterInteger(
            name="batch_size", default_value=Settings().batch_size
        )
        self.tensorboard_dir = ParameterString(
            name="tensorboard_dir", default_value="/opt/ml/output/tensorboard"
        )
        self.amp = ParameterBoolean(name="amp", default_value=Settings().amp)

    def build_steps(self, sm_session: sagemaker.Session) -> list[Step]:
        """
        define pipeline steps
        """
        # set up tensorboard logging
        tensorboard_output_config = TensorBoardOutputConfig(
            s3_output_path=os.path.join(self.output_uri, "tensorboard"),
            container_local_output_path="/opt/ml/output/tensorboard",
        )

        hyperparameters = {
            "max-epochs": self.max_epochs,
            "batch-size": self.batch_size,
            "tensorboard-dir": self.tensorboard_dir,
            "amp": self.amp,
        }

        extra_args = {}
        if not self.local_mode:
            extra_args = dict(
                checkpoint_s3_uri=f"{self.output_uri}/checkpoints",
                checkpoint_local_path="/opt/ml/checkpoints",
                output_path=self.output_uri,
                tensorboard_output_config=tensorboard_output_config,
                use_spot_instances=self.use_spot_instances,
                max_run=self.max_run,
                max_wait=self.max_wait,
            )
        print(self.model_image_uri)
        estimator = Estimator(
            image_uri=self.model_image_uri,
            role=self.sm_execution_role_arn,
            instance_type=self.gpu_instance,
            instance_count=1,
            base_job_name="training",
            sagemaker_session=sm_session,
            hyperparameters=hyperparameters,  # type: ignore
            source_dir="code",
            entry_point="run.py",
            **extra_args,  # type: ignore
        )

        estimator_args = estimator.fit(
            {
                "train": TrainingInput(
                    s3_data=self.input_data_uri,
                ),
            },
        )

        return [
            TrainingStep(
                name="train-model",
                step_args=estimator_args,  # type: ignore
            )
        ]

    def create(
        self,
        sm_session: sagemaker.Session,
    ) -> Pipeline:
        # Create a definition configuration and toggle on custom prefixing
        return Pipeline(
            name=Settings().pipeline_name,
            parameters=[
                self.max_epochs,
                self.batch_size,
                self.tensorboard_dir,
                self.amp,
            ],
            steps=self.build_steps(sm_session),
            sagemaker_session=sm_session,
        )


if __name__ == "__main__":
    pipeline_def = SagemakerPipeline(
        input_data_uri="s3://" + Settings().bucket + "/" + Settings().input_data_path,
        output_uri="s3://" + Settings().bucket + "/sagemaker-output/",
        sm_execution_role_arn=sagemaker.get_execution_role(),
        region="ca-central-1",
        local_mode=True,
    )

    # Create a SageMaker session
    if pipeline_def.local_mode:
        sess = LocalPipelineSession(
            default_bucket=Settings().bucket,
            default_bucket_prefix="local_runs",
        )
    else:
        sess = PipelineSession(
            default_bucket=Settings().bucket,
            default_bucket_prefix=Settings().pipeline_name,
        )

    pipeline = pipeline_def.create(sess)

    pipeline.create(
        role_arn=sagemaker.get_execution_role(),
        description="local pipeline example",
    )
    execution = pipeline.start()
