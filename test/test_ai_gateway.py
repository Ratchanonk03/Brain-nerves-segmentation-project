import app.main as main
import app.helper as helper
import requests

print("Health Check:", main.health_check())
print("Model Health Check:", main.check_model_health())
print("S3 Health Check:", main.check_s3_health(main.PREPROCESSED_BUCKET))

image_1 = helper.load_image("/Users/ratchanonkhongsawi/Desktop/CMKL/3rd/Scalable/assessment/assessment_2/dataset/images/1.png")
image_2 = helper.load_image("/Users/ratchanonkhongsawi/Desktop/CMKL/3rd/Scalable/assessment/assessment_2/dataset/images/2.png")

files = [
    ("payloads", image_1),
    ("payloads", image_2),
]

health_response = requests.get("http://localhost:8000/health")
print("Health Endpoint:", health_response.json())

inference_response = requests.post("http://localhost:8000/infer", files=files)
print(inference_response.json())