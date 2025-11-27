from sagemaker.model import Model
from sagemaker.local import LocalSession
import os
from pathlib import Path
from dotenv import load_dotenv
import traceback

env_path = Path.cwd().parent.parent / ".env"
print("Loading .env from:", env_path)
load_dotenv(dotenv_path=env_path)

sagemaker_session = LocalSession()

RESULTS_BUCKET = os.getenv("RESULTS_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION")


model = Model(
    image_uri="model_app:local",
    role="arn:aws:iam::199001473120:role/LabRole",
    model_data="file:///Users/ratchanonkhongsawi/Desktop/CMKL/3rd/Scalable/assessment/assessment_2/model_app/inference/model-bare.tar.gz",
    env={
        'AWS_REGION': AWS_REGION,
        'RESULTS_BUCKET': RESULTS_BUCKET,
        'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY_ID,
        'AWS_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY,
        'AWS_SESSION_TOKEN': AWS_SESSION_TOKEN,
        'DOCKER_DEFAULT_PLATFORM': 'linux/amd64',
    },
    entry_point=None,
    source_dir=None,
    sagemaker_session=sagemaker_session,
)

print("Deploying local endpoint...")

try:
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='local',
        container_startup_health_check_timeout=150,
    )
    print("DEPLOYED LOCAL ENDPOINT:", predictor.endpoint_name)
except Exception:
    print("ERROR DURING DEPLOY:")
    traceback.print_exc()