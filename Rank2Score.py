import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.clip as clip
from model.clip.model import build_model
import numpy as np
from torch import einsum
from model.networks import TextEncoder, PromptLearner


clip_vis = 'ViT-B/16'
temperature = 1.0


class DimensionProjector(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.mlp(x)
    
    
class Rank2Score(nn.Module):
    def __init__(self, device, args, dimensions, dimension_level):
        super(Rank2Score,self).__init__()
        self.device = device
        
        self.dimensions = dimensions
        self.dimension_level = dimension_level
        
        self.clip_model, _ = clip.load(clip_vis)
        self.clip_model = self.clip_model.to(torch.float32)
        self.dtype = self.clip_model.dtype
        
        self.text_encoder = TextEncoder(self.clip_model)

      
        self.level_prompt_learner = PromptLearner(self.device, args, self.dimensions, self.dimension_level, self.clip_model)
        self.level_tokenized_prompt = self.level_prompt_learner.tokenized_prompts


        self.feature_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Two-layer projector Ep from Eq. 2.
        self.dimension_projector = DimensionProjector(dim=512)

            
    def forward(self, texture_imgs, prompts, dataset_name, score_list):
        batch_size, num_views, channels, image_height, image_width = texture_imgs.shape
        texture_imgs = texture_imgs.reshape(batch_size * num_views, channels, image_height, image_width).type(self.dtype)


        feature_texture = self.clip_model.encode_image_allpatch(texture_imgs)
        num_patches = feature_texture.shape[1]
        feature_texture =  feature_texture.reshape(batch_size, num_views * num_patches, -1)
        
        prompts = clip.tokenize(prompts).to(self.device) 
        prompt_embedding = self.clip_model.token_embedding(prompts).to(self.device)
        _,  feature_prompt_all = self.text_encoder(prompt_embedding, prompts)

        dimensions_tokenized = clip.tokenize(self.dimensions).to(self.device)
        dimensions_tokenized_expand = dimensions_tokenized.repeat(batch_size,1,1) # dimensions_tokenized_expand 8x12x77
        dimensions_embedding = self.clip_model.token_embedding(dimensions_tokenized_expand).to(self.device) #dimensions_embedding 8x12x77x512
        dimensions_embedding = self.dimension_projector(dimensions_embedding)
        
        _, num_dimension, _, _ = dimensions_embedding.shape
        feature_dimension_expand = []
        
        for i in range(0, num_dimension):
            per_dim_dimensions_embedding = dimensions_embedding[:,i,:,:] #8x77x512   
            feature_dimension, _ = self.text_encoder(per_dim_dimensions_embedding, dimensions_tokenized_expand[:,i,:])
            feature_dimension_expand.append(feature_dimension.unsqueeze(1)) 

        feature_dimension_expand = torch.cat(feature_dimension_expand, dim=1)  
        
        
        sim_texture_prompt = einsum('b i d, b j d -> b j i', F.normalize(feature_prompt_all,dim=2), F.normalize(feature_texture,dim=2))
        sim_prompt_dimension = einsum('b i d, b j d -> b j i', F.normalize(feature_prompt_all,dim=2), F.normalize(feature_dimension_expand,dim=2))        
        patch_weight = einsum('b i d, b j d -> b j i', sim_prompt_dimension, sim_texture_prompt)
        patch_weight = F.softmax(patch_weight, dim=1)
        
        prompt_weight = einsum('b i d, b j d -> b j i', F.normalize(feature_dimension_expand,dim=2), F.normalize(feature_prompt_all,dim=2))
        prompt_weight = F.softmax(prompt_weight, dim=1)
        
        feature_texture_fused = []
        feature_prompt_fused = []
        
        for i in range(0,feature_dimension_expand.shape[1]):
            fused_texture = torch.sum(feature_texture * patch_weight[:,:,i].unsqueeze(2), dim=1) # 8x512
            feature_texture_fused.append(fused_texture.unsqueeze(1)) # 8x1x512
            
            fused_prompt = torch.sum(feature_prompt_all * prompt_weight[:,:,i].unsqueeze(2), dim=1) # 8x512
            feature_prompt_fused.append(fused_prompt.unsqueeze(1)) # 8x1x512
        
        feature_texture_fused = torch.cat(feature_texture_fused, dim=1)  # 8x4x512
        feature_prompt_fused = torch.cat(feature_prompt_fused, dim=1)  # 8x4x512
            
        # normalize
        feature_texture_fused = feature_texture_fused / feature_texture_fused.norm(dim=-1, keepdim=True)
        feature_prompt_fused = feature_prompt_fused / feature_prompt_fused.norm(dim=-1, keepdim=True)
        
        fused_feature = torch.cat([feature_texture_fused, feature_prompt_fused], dim=-1)
        feature_fusion = self.feature_proj(fused_feature)

        
        level_prompt_learner = self.level_prompt_learner() # 60x77x512
        feature_level, _ = self.text_encoder(level_prompt_learner, self.level_tokenized_prompt)#60x77

        feature_level = F.normalize(feature_level, dim=-1)

        feature_level_expand = feature_level.unsqueeze(0).expand(batch_size, -1, -1)

        dimensions_len = len(self.dimensions)
        level_len = len(self.dimension_level)
        
        
        


        quality_score = []
        
        
        for i in range(0, dimensions_len):
            
            start = i * level_len
            end = (i + 1) * level_len

            feature_fusion_subset = feature_fusion[:, i:i+1, :]
            feature_level_subset = feature_level_expand[:, start:end, :] 
            
            quality_logits = einsum('b i d, b j d -> b j i', F.normalize(feature_fusion_subset,dim=2), F.normalize(feature_level_subset,dim=2))
    
            pred_distribution = F.softmax(quality_logits / temperature, dim=1).squeeze(-1).to(self.device)
            bin_tensor = torch.tensor(score_list).to(self.device).reshape(1, level_len)
            bin_tensor = bin_tensor.expand(pred_distribution.size(0), -1)
    
            quality_score.append((pred_distribution * bin_tensor).sum(1, keepdim=True).to(self.device))
        
        quality_score = torch.cat(quality_score, dim=1)

        if dataset_name == "MATE-3D":
            quality_score = (quality_score - 1) / 8 * 10

        elif dataset_name == "3DGCQA":
            quality_score = (quality_score - 1) / 4 * 5
        
        elif dataset_name == "T23D-CompBench":
            quality_score = quality_score


        return quality_score, feature_fusion
    


