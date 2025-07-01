import os
import cv2
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

load_dotenv()
CLIENT=InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

def scan_plate(car):
    print("Extracting plate ...")
    result = CLIENT.infer(car,model_id="license-plate-recognition-rxg4e/11")

    if not result:
        return None
    
    print(result)
    predictions = (result['predictions'][0])
