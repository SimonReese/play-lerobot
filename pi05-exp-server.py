import importlib
from pprint import pformat

from fastapi.responses import JSONResponse
import json_numpy

from fastapi import FastAPI
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05_exp.configuration_pi05_exp import PI05ExpConfig
from lerobot.policies.pi05_exp.modeling_pi05_exp import PI05ExpPolicy
from lerobot.utils.import_utils import register_third_party_plugins
import torch
import uvicorn

json_numpy.patch()

DATASET_PATH = "./datasets/lerobot-20-ep-v3"
CHECKPOINTS_PATH = "./outputs/train/pi05_exp_evo/checkpoints/050000/pretrained_model"


class ModelServer():
    """Serve a HF Lerobot policy via FASTAPI"""

    DATASET_REPO_ID = "SimonReese/lerobot-20-ep-v3"
    """HF id of the dataset, seems required also when local path is used"""
    
    DATASET_PATH = "./datasets/lerobot-20-ep-v3"
    """Path to dataset directory for extracting statistics"""

    CHECKPOINTS_PATH = "./pretrained_model"
    """Path to checkpoint directory containing config.json"""

    def __init__(self, checkpoints_path = None, dataset_path = None) -> None:
        self.DATASET_PATH = dataset_path if dataset_path is not None else self.DATASET_PATH
        self.CHECKPOINTS_PATH = checkpoints_path if checkpoints_path is not None else self.CHECKPOINTS_PATH
        # Register for processors
        register_third_party_plugins()
        # Load dataset
        dataset = LeRobotDataset(
            repo_id=self.DATASET_REPO_ID,
            root=self.DATASET_PATH
        )
        # Load policy config from pretrained
        config:PI05ExpConfig = PreTrainedConfig.from_pretrained(self.CHECKPOINTS_PATH) # WARN: do not attempt to construct the specific PreTrainedConfig since Draccus will already solve it #type:ignore
        # Customize config
        config.dtype = "float32"
        config.device = "cuda"
        
        # Load model
        self.policy = PI05ExpPolicy.from_pretrained(self.CHECKPOINTS_PATH, config=config)
        self.policy.eval()

        # Load pre_post processors
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            dataset_stats=dataset.meta.stats #type: ignore
        )

        # Print useful info
        print("Model loaded")
        print(f"Allocated {torch.cuda.memory_allocated(config.device) / (1e9)}GB")
        print(f"Model will take:\n- Input:{pformat(config.input_features)},\n- Output:{pformat(config.output_features)} and of course a 'task': list[str] key")


    def predict(self, data: dict):
        import model_hot_code
        importlib.reload(model_hot_code)
        # Patch numpy json.load 
        #json_numpy.patch()
        out = model_hot_code.predict(data, self.policy, self.preprocessor, self.postprocessor)
        return out

    def listen(self, host = "0.0.0.0", port = 8042):
        self.app = FastAPI()
        self.app.post("/predict")(self.predict)
        uvicorn.run(self.app, host=host, port=port)

def main():
    model = ModelServer(checkpoints_path=CHECKPOINTS_PATH, dataset_path=DATASET_PATH)
    model.listen()

if __name__ == "__main__":
    main()