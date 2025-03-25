import requests
import time

   
payload = {"image_url": "https://media.istockphoto.com/id/157671964/photo/portrait-of-a-tabby-cat-looking-at-the-camera.jpg?s=612x612&w=0&k=20&c=iTsJO6vuQ5w3hL5pWn42C91ziMRUsYd725oUGRRewjM="}
headers = {"Content-Type": "application/json"}
api_url = "http://localhost:8000/detect-landmarks"
start_time = time.time()
response = requests.post(api_url, json=payload, headers=headers)
end_time = time.time()

response_time = end_time - start_time
status_code = response.status_code

result = {
    "response_time": response_time,
    "status_code": status_code,
    "success": status_code == 200
}

if status_code == 200:
    result["response_data"] = response.json()
else:
    result["error"] = response.text

print(result)