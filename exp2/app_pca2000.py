import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import tensorflow as tf
import time


# Initialize the app
app = FastAPI()

# Load the TFLite model
MODEL_PATH = './models/self_driving_model_pca_2000.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Process the uploaded file
        img = Image.open(file.file).resize((64, 64))
        img_array = np.expand_dims(np.array(img) / 64.0, axis=0).astype(np.float32)  # Normalize and reshape
        
        # Perform inference with TFLite
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]  # Get the output
        end_time = time.time()

        # CIFAR-10 class labels
        self_driving_classes = [
            "left", "forward", "right"
        ]

        # Get the predicted class and confidence
        predicted_class = self_driving_classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))  # Convert to Python float

        # Calculate inference time
        inference_time = end_time - start_time

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            # "prediction": prediction.tolist(),  # Convert to list for JSON serialization
            "inference_time_seconds": inference_time
        }
    except Exception as e:
        return {"error": str(e)}


# Main entry point for local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9020)