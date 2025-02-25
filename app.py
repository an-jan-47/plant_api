import os
import uvicorn
import tensorflow.lite as tflite
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import base64
import io

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']

# Define FastAPI app
app = FastAPI()

# Define request format
class ImageInput(BaseModel):
    image: str  # Base64 encoded image

# Function to preprocess image
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")  # Ensure image is in RGB mode
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image)

    # Normalize based on model input type
    if input_dtype == np.float32:
        image = image.astype(np.float32) / 255.0
    elif input_dtype == np.uint8:
        image = image.astype(np.uint8)

    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(request: ImageInput):
    try:
        # Decode Base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        image = preprocess_image(image)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Get predicted class and confidence
        predicted_class = int(np.argmax(output))  # Classification task
        confidence = float(np.max(output))

        return {"class_name": str(predicted_class), "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Hello, Render!"}

# Run the app with dynamic port for Render
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
