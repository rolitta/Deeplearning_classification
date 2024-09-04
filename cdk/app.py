import os

import aws_cdk as cdk

from pipe.pipe_stack import PipeStack
from settings import Settings


app: cdk.App = cdk.App()

PipeStack(
    app,
    "TrainingPipelineTemplate",
    env={
        "account": os.environ.get("CDK_DEPLOY_ACCOUNT"),
        "region": os.environ.get("CDK_DEPLOY_REGION"),
    },
)

for tag in Settings().app_tags:
    cdk.Tags.of(app).add(tag["key"], tag["value"])

app.synth()
