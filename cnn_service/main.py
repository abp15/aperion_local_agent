import tensorflow as tf
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Aperion CNN Predictor")

# Load the model directly as a TFSMLayer
MODEL_DIR = "./my_model"
try:
    sm_layer = keras.layers.TFSMLayer(MODEL_DIR, call_endpoint='serving_default')
    print("✅ Aperion Model Loaded.")
except Exception as e:
    print(f"❌ Load Error: {e}")

class PredictionRequest(BaseModel):
    instances: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Extract the string from the JSON payload
        b64_string = request.instances[0].get("image_bytes")
        
        # Convert to a 1D Tensor of strings
        # The model expects a batch, so we wrap in a list: [b64_string]
        input_tensor = tf.constant([b64_string], dtype=tf.string)
        
        # Invoke the model
        raw_output = sm_layer(input_tensor)
        
        # Handle the dictionary output (usually 'price_prediction' or 'output_0')
        first_key = list(raw_output.keys())[0]
        cost = float(np.array(raw_output[first_key]).flatten()[0])
            
        return {"predictions": [[cost]]}
    
    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
