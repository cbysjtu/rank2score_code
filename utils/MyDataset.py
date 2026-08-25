import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import random
from torchvision import transforms
from torch.utils import data
from PIL import Image
from torch.utils.data import Sampler
from collections import defaultdict


class MYDataset(data.Dataset):
    def __init__(self, data_dir, datainfo_path, data_name, transform, patch_num, crop_size=224, img_length_read=6, is_train=True):
        super(MYDataset, self).__init__()
        self.data_name = data_name
        self.crop_size = crop_size
        self.patch_num = patch_num
        self.data_dir = data_dir
        self.transform = transform
        self.img_length_read = img_length_read
        self.is_train = is_train

        dataInfo = pd.read_csv(datainfo_path, header=0, sep=',', index_col=False, encoding="utf-8-sig")

        if data_name == "MATE-3D":
            self.name_model = [item for item in dataInfo['model'].tolist() for _ in range(patch_num)]
            self.name_prompt = [item for item in dataInfo['prompt'].tolist() for _ in range(patch_num)]
            self.prompt_all = None
            self.mos_alignment = [item for item in dataInfo['Alignment'].tolist() for _ in range(patch_num)]
            self.mos_geometry = [item for item in dataInfo['Geometry'].tolist() for _ in range(patch_num)]
            self.mos_texture = [item for item in dataInfo['Texture'].tolist() for _ in range(patch_num)]
            self.mos_overall = [item for item in dataInfo['Overall'].tolist() for _ in range(patch_num)]

        elif data_name == "3DGCQA":
            obj_info = dataInfo['Image'].tolist()
            name_model_list, name_prompt_list, prompt_all_list = [], [], []
            for obj in obj_info:
                model, rest = obj.split("\\", 1)
                prompt, _ = rest.split("@", 1)
                prompt = prompt.replace("_", " ")
                name_model_list.append(model)
                name_prompt_list.append(prompt)
                prompt_all_list.append(rest)

            self.name_model = [item for item in name_model_list for _ in range(patch_num)]
            self.name_prompt = [item for item in name_prompt_list for _ in range(patch_num)]
            self.prompt_all = [item for item in prompt_all_list for _ in range(patch_num)]
            self.mos_alignment = [item for item in dataInfo['alignment'].tolist() for _ in range(patch_num)]
            self.mos_geometry = None
            self.mos_texture = None
            self.mos_overall = [item for item in dataInfo['quality'].tolist() for _ in range(patch_num)]
            
        elif data_name == "T23D-CompBench":
            self.name_model = [item for item in dataInfo['model'].tolist() for _ in range(patch_num)]
            self.name_prompt = [item for item in dataInfo['prompt'].tolist() for _ in range(patch_num)]
            self.prompt_all = None
            self.mos_object_alignment = [item for item in dataInfo['Object Alignment'].tolist() for _ in range(patch_num)]
            self.mos_attribute_alignment = [item for item in dataInfo['Attribute Alignment'].tolist() for _ in range(patch_num)]
            self.mos_interaction_alignment = [item for item in dataInfo['Interaction Alignment'].tolist() for _ in range(patch_num)]
            self.mos_overall_alignment = [item for item in dataInfo['Overall Alignment'].tolist() for _ in range(patch_num)]
            self.mos_texture_clarity = [item for item in dataInfo['Texture Clarity'].tolist() for _ in range(patch_num)]
            self.mos_texture_aesthetics = [item for item in dataInfo['Texture Aesthetics'].tolist() for _ in range(patch_num)]
            self.mos_geometry_loss = [item for item in dataInfo['Geometry Loss'].tolist() for _ in range(patch_num)]
            self.mos_geometry_redundancy = [item for item in dataInfo['Geometry Redundancy'].tolist() for _ in range(patch_num)]
            self.mos_geometry_roughness = [item for item in dataInfo['Geometry Roughness'].tolist() for _ in range(patch_num)]
            self.mos_overall_visual = [item for item in dataInfo['Overall Visual'].tolist() for _ in range(patch_num)]
            self.mos_3d_authentic = [item for item in dataInfo['3D Authentic'].tolist() for _ in range(patch_num)]
            self.mos_overall_quality = [item for item in dataInfo['Overall Quality'].tolist() for _ in range(patch_num)]
            
    
        else:
            raise ValueError(f"Unknown data_name: {data_name}")

        self.length = len(self.name_model)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name_model = self.name_model[idx]
        name_prompt = self.name_prompt[idx]
        name_prompt_for_path = name_prompt.replace(" ", "_")

        if self.data_name == "MATE-3D":
            img_folder = os.path.join(self.data_dir, name_model, name_prompt_for_path)
        elif self.data_name == "T23D-CompBench":
            img_folder = os.path.join(self.data_dir, name_model, name_prompt_for_path)
        elif self.data_name == "3DGCQA":
            img_folder = os.path.join(self.data_dir, name_model, self.prompt_all[idx])
        else:
            raise ValueError(f"Unknown data_name: {self.data_name}")

        img_transformed = torch.zeros([self.img_length_read, 3, self.crop_size, self.crop_size])
        img_read_index = 0
        for i in range(self.img_length_read):
            img_name = os.path.join(img_folder, f'rendered_view_{i}.png')
            if os.path.exists(img_name):
                img = Image.open(img_name).convert('RGB')
                img = transforms.ToTensor()(img)
                img = self.transform(img)
                img_transformed[i] = img
                img_read_index += 1
            else:
                print(img_name)
                print('Image does not exist!')

        if img_read_index < self.img_length_read:
            for j in range(img_read_index, self.img_length_read):
                img_transformed[j] = img_transformed[img_read_index - 1]


        if self.data_name == "MATE-3D":
            mos_values = [self.mos_alignment[idx], self.mos_geometry[idx], self.mos_texture[idx], self.mos_overall[idx]]
        elif self.data_name == "3DGCQA":
            mos_values = [self.mos_alignment[idx], self.mos_overall[idx]]
        elif self.data_name == "T23D-CompBench":
            mos_values = [self.mos_object_alignment[idx], self.mos_attribute_alignment[idx], self.mos_interaction_alignment[idx], self.mos_overall_alignment[idx], self.mos_texture_clarity[idx], self.mos_texture_aesthetics[idx], self.mos_geometry_loss[idx], self.mos_geometry_redundancy[idx],self.mos_geometry_roughness[idx], self.mos_overall_visual[idx], self.mos_3d_authentic[idx], self.mos_overall_quality[idx]]

            
        mos = torch.FloatTensor(np.array(mos_values))
        return img_transformed, name_prompt, mos


class PromptCurriculumBatchSampler(Sampler):
    """Balanced sampler implementing the prompt-number curriculum (Eq. 14)."""

    def __init__(self, prompt_list, batch_size, num_prompts=1, seed=2001, epoch=0):
        self.prompt_to_indices = defaultdict(list)
        for idx, prompt in enumerate(prompt_list):
            self.prompt_to_indices[prompt].append(idx)
        self.prompts = list(self.prompt_to_indices)
        self.batch_size = batch_size
        self.num_prompts = max(1, min(num_prompts, batch_size, len(self.prompts)))
        self.seed = seed
        self.epoch = epoch
        self.batches = self._generate_batches()

    @staticmethod
    def _balanced_sample_allocation(total, num_parts, rng):
        quotient, remainder = divmod(total, num_parts)
        allocation = [quotient + 1] * remainder + [quotient] * (num_parts - remainder)
        rng.shuffle(allocation)
        return allocation

    def _generate_batches(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches = []
        num_batches = max(1, len(self.prompt_to_indices) * max(
            len(indices) for indices in self.prompt_to_indices.values()
        ) // self.batch_size)
        for _ in range(num_batches):
            selected = rng.sample(self.prompts, self.num_prompts)
            allocation = self._balanced_sample_allocation(
                self.batch_size, self.num_prompts, rng
            )
            batch = []
            for prompt, count in zip(selected, allocation):
                candidates = self.prompt_to_indices[prompt]
                sample = rng.sample(candidates, count) if len(candidates) >= count else rng.choices(candidates, k=count)
                batch.extend(sample)
            rng.shuffle(batch)
            all_batches.append(batch)
        return all_batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

