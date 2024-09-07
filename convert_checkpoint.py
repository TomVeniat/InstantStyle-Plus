import torch
import json
from diffusers import UNet2DConditionModel
from tqdm import tqdm

# Load the index file
index_file = './checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE/unet/diffusion_pytorch_model.bin.index.json'
with open(index_file, 'r') as f:
    index_data = json.load(f)
print(index_data)
# Initialize an empty state_dict
state_dict = {}

# Load each shard and update the state_dict
it =  tqdm(set(index_data['weight_map'].values()))
for shard_file in it:
    shard_path = f"./checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE/unet/{shard_file}"
    it.set_description(shard_path)
    shard_state_dict = torch.load(shard_path, map_location="cpu")
    state_dict.update(shard_state_dict)

# Load the UNet model using the state_dict
model = UNet2DConditionModel.from_config('./checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE/unet')
model.load_state_dict(state_dict)

# Save the model as a single file
model.save_pretrained('./checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE/unet_single_file', )

print("Model successfully converted to a single file!")
