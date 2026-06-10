# Temp function for hot code
from pprint import pformat

from fastapi.responses import JSONResponse
from lerobot.policies.pi05_exp.modeling_pi05_exp import PI05ExpPolicy
import numpy
import torch

def predict(data, policy: PI05ExpPolicy, preprocessor, postprocessor):
    for key, value in data.items():  
        if isinstance(value, numpy.ndarray):  
            data[key] = torch.from_numpy(value.copy()).float()
    # Performs prediciton
    pre_data = preprocessor(data)
    prediction = policy.predict_action_chunk(pre_data)
    processed: torch.Tensor = postprocessor(prediction)
    print(f"Output: {pformat(processed, depth=2, compact=True)}")
    
    processed_np = processed.numpy()
    # Return action
    return JSONResponse(processed_np)


