from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from lerobot.policies.factory import make_pre_post_processors

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode

# DATASETS PATHS
LEROBOT_REPO_ID_V3 = "SimonReese/lerobot-20-ep-v3"
LEROBOT_DATASET_ROOT_V3 = "./datasets/lerobot-20-ep-v3"

MODEL_ID = "lerobot/pi05_base"
#LeRobotDataset("lerobot/libero_spatial_image")
dataset = LeRobotDataset(
    repo_id = LEROBOT_REPO_ID_V3,
    root= LEROBOT_DATASET_ROOT_V3
) 
print(dataset.meta.features)


policy = PI05Policy.from_pretrained(model_id).to("cuda").eval()
print(policy.config.input_features)
exit()
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

pre, post = make_pre_post_processors(
    policy.config,
    dataset_stats=dataset.meta.stats #type: ignore
)


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

