import json
from typing import Tuple
import datetime

from aws_cdk import (
    aws_sagemaker as sm,
    aws_ssm as ssm,
    Stack,
)
from constructs import Construct
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker_pipeline import SagemakerPipeline
from settings import Settings


class PipeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, id, **kwargs)

        # Load infrastructure params from SSM
        sources_bucket_name = ssm.StringParameter.value_from_lookup(
            self, Settings().ssm_bucket_param
        )
        sm_execution_role_arn = ssm.StringParameter.value_from_lookup(
            self, Settings().ssm_role_param
        )

        # Create a configured pipeline
        self.example_pipeline, self.example_pipeline_arn = (
            self.create_pipeline_resource(
                pipeline_factory=SagemakerPipeline(
                    sm_execution_role_arn=sm_execution_role_arn,
                    input_data_uri=f"s3://{sources_bucket_name}/{Settings().input_data_path}",
                    output_uri=(
                        f"s3://{sources_bucket_name}/sagemaker-output/{Settings().pipeline_name}/"
                        f"{datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}"
                    ),
                    region=self.region,
                ),
                sources_bucket_name=sources_bucket_name,
                sm_execution_role_arn=sm_execution_role_arn,
            )
        )

    def create_pipeline_resource(
        self,
        pipeline_factory: SagemakerPipeline,
        sources_bucket_name: str,
        sm_execution_role_arn: str,
    ) -> Tuple[sm.CfnPipeline, str]:
        # Create a SageMaker session
        sess = PipelineSession(default_bucket=sources_bucket_name)

        pipeline = pipeline_factory.create(
            sm_session=sess,
        )
        pipeline_def_json = json.dumps(
            json.loads(pipeline.definition()), indent=2, sort_keys=True
        )

        # Define CloudFormation resource for the pipeline, so it can be deployed to your account
        pipeline_cfn = sm.CfnPipeline(
            self,
            id=f"SagemakerPipeline-{Settings().pipeline_name}",
            pipeline_name=Settings().pipeline_name,
            pipeline_definition={"PipelineDefinitionBody": pipeline_def_json},
            role_arn=sm_execution_role_arn,
        )
        arn = self.format_arn(
            service="sagemaker",
            resource="pipeline",
            resource_name=pipeline_cfn.pipeline_name,
        )
        return pipeline_cfn, arn
