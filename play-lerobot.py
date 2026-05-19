from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from lerobot.policies.factory import make_pre_post_processors

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode

# DATASETS PATHS
LEROBOT_REPO_ID_V3 = "SimonReese/lerobot-20-ep-v3"
LEROBOT_DATASET_ROOT_V3 = "./datasets/lerobot-20-ep-v3"
# Model ID
MODEL_ID = "lerobot/pi05_base"


def explore_dataset(dataset: LeRobotDataset, policy, pre, post):
    ep_index = 0
    start_idx = dataset.meta.episodes["dataset_from_index"][ep_index]
    end_idx = dataset.meta.episodes["dataset_to_index"][ep_index]

    for f in range(start_idx, end_idx):
        frame = dict(dataset[f])
        batch = pre(frame)
        #print(batch.keys())
        prediction = policy.select_action(batch)
        processed = post(prediction)
        print(processed)
        exit()

def main():
    # Load dataset
    dataset = LeRobotDataset(
        repo_id = LEROBOT_REPO_ID_V3,
        root= LEROBOT_DATASET_ROOT_V3
    ) 
    print(dataset.meta.features)

    # Config
    config = PI05Config(
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 244)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(8, ))
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
        },
        device="cuda",
        use_relative_actions=True
    )
    # Load model
    policy = PI05Policy.from_pretrained(MODEL_ID, config=config).eval()
    print(policy.config.input_features)
    pre, post = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset.meta.stats #type: ignore
    )
    idx = 0
    for frame in dataset:
        if idx == 10: break
        processed = pre(frame)
        pred = policy.select_action(processed)
        processed = post(pred)
        print(pred)
        idx +=1
    

if __name__ == "__main__":
    main()

def remap_features(policy):
    print("Changing features")
    assert policy.config.input_features is not None
    old_keys = list(policy.config.input_features.keys())
    for k in old_keys:
        if policy.config.input_features[k].type == FeatureType.VISUAL:
            del policy.config.input_features[k]

    
    # Add new mappings
    policy.config.input_features["observation.images.image"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3,256, 256))
    policy.config.input_features["observation.images.wrist_image"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))

    policy.config.normalization_mapping["STATE"] = NormalizationMode.MEAN_STD
    policy.config.normalization_mapping["ACTION"] = NormalizationMode.MEAN_STD

    policy.config.output_features["action"] = PolicyFeature(FeatureType.ACTION, shape=(7,))