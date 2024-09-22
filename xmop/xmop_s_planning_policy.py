import torch
import torch.nn as nn
import torch.nn.functional as F
from base_models import TransformerEncoderBlock, PositionEmbedding


class SinglestepPoseModel(nn.Module):
    def __init__(self, config):
        super(SinglestepPoseModel, self).__init__()
        self.hidden_dim = config['hidden_dim']
        self.num_transformer_heads = config['num_heads']

        self.ref_pose_proj = nn.Linear(9, self.hidden_dim)
        
        self.max_dof = config['max_dof']
        self.num_tokens = (self.max_dof+1)*2 + 1 # (DOF+base) reference and current + target
        self.pos_embed = PositionEmbedding(dim=self.hidden_dim, 
                                            max_seq_len=self.num_tokens, 
                                            learned=False)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_size=self.hidden_dim,
                                             num_heads=self.num_transformer_heads)
            for _ in range(config['num_layers'])
        ])
        self.final_layer = nn.Linear(self.hidden_dim, 9)
        
        causal_mask = torch.triu(torch.ones(self.num_tokens, self.num_tokens), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask==1, float('-inf'))
        causal_mask[:,self.max_dof+1:] = 0
        self.register_buffer("causal_mask", causal_mask)

        self.initialize_weights()
    
    def forward(self, 
                ref_link_poses,
                target_pose,
                mask):
        ref_link_feat = self.ref_pose_proj(ref_link_poses)
        target_feat = self.ref_pose_proj(target_pose).unsqueeze(1)
        delta_query = torch.zeros_like(ref_link_feat)

        # sequence is important here for masking
        x = torch.cat((delta_query, ref_link_feat, target_feat), dim=1)

        x = self.pos_embed(x)
        for block in self.encoder_blocks:
            x = block(x, x_mask=mask)     
        x = self.final_layer(x)
        return x
    
    def initialize_weights(self):
        # Initialize linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        # Initialize token pos embedding
        if self.pos_embed.learned:
            torch.nn.init.normal_(self.pos_embed.pos_enc, mean=0.0, std=0.02)       

        # Zero-out output layers
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

class SinglestepPosePlanningPolicy(SinglestepPoseModel):
    def __init__(self, config):
        super(SinglestepPosePlanningPolicy, self).__init__(config['pose_model'])
    
    def get_mask(self, dof_mask):
        B = dof_mask.shape[0]
        dof_mask = dof_mask.float()
        dof_mask = dof_mask.masked_fill(dof_mask==1, float('-inf'))
        # first max_dof+1 positions are generative
        assert dof_mask.shape[1] == self.max_dof+1
        extended_dof_mask = \
            dof_mask.unsqueeze(-1).repeat(1,1,self.num_tokens).transpose(-2,-1)

        mask = self.causal_mask.detach().clone().unsqueeze(0).repeat(B,1,1)
        mask[:,:,:self.max_dof+1] += extended_dof_mask
        mask[:,:,self.max_dof+1:2*(self.max_dof+1)] += extended_dof_mask

        mask = mask.unsqueeze(1).repeat(1,self.num_transformer_heads,1,1)
        return mask
    
    def get_action(self, data_dict):
        mask = self.get_mask(data_dict['dof_mask'])
        delta_poses = super().forward(data_dict['PA_9d'],
                                        data_dict['goal_pose_9d'],
                                        mask)[:,:self.max_dof+1]
        return delta_poses
    
    def forward(self, data_dict):
        rel_poses = data_dict['TAB_9d_h']
        assert rel_poses.shape[1] == 1
        rel_poses = rel_poses.squeeze(1)

        mask = self.get_mask(data_dict['dof_mask'])
        delta_poses = super().forward(data_dict['PA_9d'],
                                        data_dict['goal_pose_9d'],
                                        mask)

        loss = F.mse_loss(delta_poses[:,:self.max_dof+1], rel_poses, reduction='none')
        loss = loss[~data_dict['dof_mask']]

        data_dict['loss'] = loss.mean()
        return data_dict
