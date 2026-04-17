# 1 open rlbench dataset
# 2 test data loading
# 3 convert to lerobot

import os
import pickle
import time
from typing import List
from numpy import ndarray
import numpy
from rlbench_utils.demo import Demo
from rlbench_utils.observation import Observation
from rlbench_utils.observation_config import CameraConfig, ObservationConfig
import rlbench_utils.utils 
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

# DATASETS PATHS
RLBENCH_DATASET_ROOT = "./datasets/rlbench/generated-16-04-00-00"
TASKS = os.listdir(RLBENCH_DATASET_ROOT)

# CAMERA CONFIGURATIONS
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)
CAMERA_CONFIG = CameraConfig(
    image_size= IMAGE_SIZE
)
OBS_CONFIG = ObservationConfig(
    left_shoulder_camera= CAMERA_CONFIG,
    right_shoulder_camera= CAMERA_CONFIG,
    overhead_camera= CAMERA_CONFIG,
    wrist_camera= CAMERA_CONFIG,
    front_camera= CAMERA_CONFIG,
    gripper_joint_positions=True
)

# LeRobot features dictionary
FEATURES_DICT = {
    "image": {
        "dtype": "image",
        "shape": IMAGE_SHAPE,
        "names": ["width", "height", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": IMAGE_SHAPE,
        "names": ["width", "height", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (9,), # Joint states (7) + Gripper joint states
        "names": ["state"],
    },
    
    "actions": {
        "dtype": "float32",
        "shape": (8,), # Joint velocities (7) + Gripper_is_open
        "names": ["actions"],
    }
}

FRAME_DICT = {
    "image": ndarray,
    "wrist_image": ndarray,
    "state" : ndarray,
    "actions" : ndarray,
    "task": str
}

def main():

    lerobot_dataset = LeRobotDataset.create(
        repo_id=f"RLBench/16-test",
        fps=10,
        features=FEATURES_DICT,
        root=f"./datasets/lerobot/16-test",
        robot_type="panda"
    )

    # Open every task
    for TASK in TASKS:
        VARIATIONS = get_variations_ids(RLBENCH_DATASET_ROOT, TASK)
        # Open every variation
        for VARIATION in VARIATIONS:
            EPISODES = get_episodes_number(RLBENCH_DATASET_ROOT, TASK, VARIATION)
            # Open every episode
            for EPISODE in EPISODES:
                
                # Get the demo for the episode, since loading all episodes is expensive
                DEMOS = rlbench_utils.utils.get_stored_demos(
                    amount=1,
                    image_paths=False,
                    dataset_root=RLBENCH_DATASET_ROOT,
                    variation_number=VARIATION,
                    task_name=TASK,
                    obs_config=OBS_CONFIG,
                    random_selection=False,
                    from_episode_number=EPISODE
                )
                DEMO = DEMOS.pop()
                print(f"Processing TASK:{TASK}, VARIATION:{VARIATION}, EP:{EPISODE}:\n{DEMO.demo_description}")
                # Given an image at timestep t, we want the action to reach next position
                _prev_obs: Observation
                for seq, observation in enumerate(DEMO):
                    FRAME_DICT["image"] = observation.front_rgb # TODO: Which format?? Now is uint8 rgb with (width, height, channels)
                    FRAME_DICT["wrist_image"] = observation.wrist_rgb
                    FRAME_DICT["state"] = numpy.concatenate((observation.joint_positions, observation.gripper_joint_positions), dtype=numpy.float32)
                    FRAME_DICT["actions"] = numpy.concatenate((observation.joint_velocities, [observation.gripper_open]),dtype=numpy.float32) # TODO: convert to delta xyz
                    if type(DEMO.demo_description) == list:
                        FRAME_DICT["task"] = DEMO.demo_description[0]
                    else:
                        FRAME_DICT["task"] = DEMO.demo_description

                    lerobot_dataset.add_frame(FRAME_DICT)
                lerobot_dataset.save_episode()
                
        

# ----- UTILITY FUNCTIONS -----

# Get list of variations for task
def get_variations_ids(dataset_path: str, task_name:str, VARIATION_FOLDER_PREFIX = "variation") -> List[int]:
    # Open variation
    VARIATION_FOLDER_PREFIX = "variation"
    variation_folders = os.listdir(os.path.join(dataset_path, task_name))
    if "all_variations" in variation_folders: variation_folders.remove("all_variations")
    variation_ids = []
    for variation in variation_folders:
        if not os.path.isdir(os.path.join(dataset_path, task_name, variation)): continue
        if VARIATION_FOLDER_PREFIX not in variation: continue
        id = variation.removeprefix(VARIATION_FOLDER_PREFIX)
        variation_ids.append(int(id))
    return variation_ids

def get_episodes_number(dataset_path: str, task_name:str, variation_id: int, VARIATION_FOLDER_PREFIX = "variation", EPISODES_FOLDER = "episodes",EPISODE_FOLDER_PREFIX = "episode") -> List[int]:
    episodes_folders = os.listdir(os.path.join(dataset_path, task_name, f"{VARIATION_FOLDER_PREFIX}{variation_id}", EPISODES_FOLDER))
    episode_ids = []
    for ep in episodes_folders:
        if not os.path.isdir(os.path.join(dataset_path, task_name, f"{VARIATION_FOLDER_PREFIX}{variation_id}", EPISODES_FOLDER, ep)): continue
        if EPISODE_FOLDER_PREFIX not in ep: continue
        id = ep.removeprefix(EPISODE_FOLDER_PREFIX)
        episode_ids.append(int(id))
    episode_ids.sort()
    return episode_ids

if __name__ == "__main__":
    main()