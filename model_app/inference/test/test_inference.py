from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/ratchanonkhongsawi/Desktop/CMKL/3rd/Scalable/assessment/assessment_2/ai_gateway/.env")


import os
from ..code.inference_handler import model_fn, input_fn, predict_fn, output_fn
import json
from pathlib import Path


# 1) Simulate SageMaker calling model_fn
print("CWD:", os.getcwd())
print(Path("/Users/ratchanonkhongsawi/Desktop/CMKL/3rd/Scalable/assessment/assessment_2/model/inference/").exists())
model = model_fn("/Users/ratchanonkhongsawi/Desktop/CMKL/3rd/Scalable/assessment/assessment_2/model/inference/")

# 2) Simulate HTTP body from your AI Gateway
request_body = json.dumps({
    "inputs": [
        {
            "key": "preprocessed/832f629e07084cbba4395667e437ba18.npy",
            "bucket": os.getenv("PREPROCESSED_BUCKET")
        },
        {
            "key": "preprocessed/876e485c3fa948b79bcf220a239057e1.npy",
            "bucket": os.getenv("PREPROCESSED_BUCKET")
        }
    ]
})

content_type = "application/json"

# 3) Simulate input_fn
data = input_fn(request_body, content_type)

# 4) Simulate predict_fn
prediction = predict_fn(data, model)

# 5) Simulate output_fn
response_body = output_fn(prediction, "application/json")
print(response_body)
