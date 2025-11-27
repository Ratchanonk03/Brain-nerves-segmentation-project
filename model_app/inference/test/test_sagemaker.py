from sagemaker.model import Model
from sagemaker.local import LocalSession
import os

from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/ratchanonkhongsawi/Desktop/CMKL/3rd/Scalable/assessment/assessment_2/ai_gateway/.env")

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}


RESULTS_BUCKET = os.getenv("RESULTS_BUCKET")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION")

model = Model(
    image_uri="model_app:local",
    role="arn:aws:iam::123456789012:role/LabRole",
    model_data="file://Users/ratchanonkhongsawi/Desktop/CMKL/3rd/Scalable/assessment/assessment_2/model_app/inference/model.tar.gz",
    env={
        'AWS_REGION': AWS_REGION,
        'RESULTS_BUCKET': RESULTS_BUCKET,
        'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY_ID,
        'AWS_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY,
        'AWS_SESSION_TOKEN': AWS_SESSION_TOKEN,
    },
    sagemaker_session=sagemaker_session
)

# Deploy to local "endpoint"
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='local'
)

predictor.predict({"ping": True})