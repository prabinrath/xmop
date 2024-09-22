import torch
import torch.nn as nn
import torch.nn.functional as F
from base_models import DiffusionTransformerEncoderBlock, \
    TimestepEmbedding, FinalLayer, PositionEmbedding
from diffusers import DDPMScheduler, DDIMScheduler
from common.rotation_torch import rot6d_to_rotmat
import numpy as np


class MultistepPoseNoiseModel(nn.Module):
    def __init__(self, config):
        super(MultistepPoseNoiseModel, self).__init__()
        self.hidden_dim = config['hidden_dim']
        self.num_transformer_heads = config['num_heads']
        self.horizon = config['horizon']

        self.ref_pose_proj = nn.Linear(9, self.hidden_dim)
        self.noisy_pose_proj = nn.Linear(9, self.hidden_dim)
        
        self.max_dof = config['max_dof']
        self.num_tokens = (self.max_dof+1)*(self.horizon+1) + 1 # (DOF+base) * (reference + horizon) + target
        self.dof_pos_embed = PositionEmbedding(dim=self.hidden_dim, 
                                            max_seq_len=self.max_dof+1, 
                                            learned=False)
        self.hor_pos_embed = nn.Embedding(self.horizon, self.hidden_dim)
        self.cond_pos_embd = PositionEmbedding(dim=self.hidden_dim, 
                                            max_seq_len=(self.max_dof+1)+1, 
                                            learned=True)
        self.t_embed = TimestepEmbedding(hidden_size=self.hidden_dim)
        self.encoder_blocks = nn.ModuleList([
            DiffusionTransformerEncoderBlock(hidden_size=self.hidden_dim,
                                             num_heads=self.num_transformer_heads)
            for _ in range(config['num_layers'])
        ])
        self.final_layer = FinalLayer(hidden_size=self.hidden_dim,
                                      out_channels=9)
        
        self.causal_mask = torch.zeros((self.num_tokens, self.num_tokens), dtype=torch.float32)
        self.causal_mask[:(self.max_dof+1)*self.horizon, :(self.max_dof+1)*self.horizon] = \
            self.create_causal_mask(self.max_dof+1, self.horizon)

        self.initialize_weights()
    
    def forward(self, 
                ref_link_poses,
                noisy_poses,
                target_pose,
                mask,
                t):
        B = t.shape[0]
        noisy_feat = self.noisy_pose_proj(noisy_poses)
        dof_pos_embeding = self.dof_pos_embed(torch.zeros((B,self.max_dof+1,self.hidden_dim), device=t.device))
        dof_pos_embeding = dof_pos_embeding.unsqueeze(1).repeat(1,self.horizon,1,1).view(B,-1,self.hidden_dim)
        noisy_feat += dof_pos_embeding
        horizon_indices = torch.arange(self.horizon, device=t.device).repeat_interleave(self.max_dof+1)\
            .unsqueeze(0).repeat(B,1)
        hor_pos_embeding = self.hor_pos_embed(horizon_indices)
        noisy_feat += hor_pos_embeding

        ref_link_feat = self.ref_pose_proj(ref_link_poses)
        target_feat = self.ref_pose_proj(target_pose).unsqueeze(1)
        cond = torch.cat((ref_link_feat, target_feat), dim=1)
        cond = self.cond_pos_embd(cond)

        # sequence is important here for masking
        x = torch.cat((noisy_feat, cond), dim=1)
        c = self.t_embed(t)
        for block in self.encoder_blocks:
            x = block(x, c, x_mask=mask)     
        x = self.final_layer(x, c)
        return x
    
    def fixed_forward(self, ref_link_poses, target_pose):
        ref_link_feat = self.ref_pose_proj(ref_link_poses)
        target_feat = self.ref_pose_proj(target_pose).unsqueeze(1)
        cond = torch.cat((ref_link_feat, target_feat), dim=1)
        cond = self.cond_pos_embd(cond)

        B = ref_link_feat.shape[0]
        device = ref_link_feat.device
        dof_pos_embeding = self.dof_pos_embed(torch.zeros((B,self.max_dof+1,self.hidden_dim), device=device))
        dof_pos_embeding = dof_pos_embeding.unsqueeze(1).repeat(1,self.horizon,1,1).view(B,-1,self.hidden_dim)
        horizon_indices = torch.arange(self.horizon, device=device).repeat_interleave(self.max_dof+1)\
            .unsqueeze(0).repeat(B,1)
        hor_pos_embeding = self.hor_pos_embed(horizon_indices)
        noise_pos_embedding = dof_pos_embeding + hor_pos_embeding

        return cond, noise_pos_embedding

    def denoising_forward(self, fixed_feats, noisy_poses, mask, t):
        cond, noise_pos_embedding = fixed_feats
        noisy_feat = self.noisy_pose_proj(noisy_poses) + noise_pos_embedding

        # sequence is important here for masking
        x = torch.cat((noisy_feat, cond), dim=1)
        c = self.t_embed(t)
        for block in self.encoder_blocks:
            x = block(x, c, x_mask=mask)     
        x = self.final_layer(x, c)
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
        torch.nn.init.normal_(self.cond_pos_embd.pos_enc, mean=0.0, std=0.02)

        # Initialize timestep embedding
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.encoder_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)          

    def create_causal_mask(self, N, H):
        # this function was generated using ChatGPT-4
        neg_inf = float('-inf')
        
        # Create matrix C
        C = torch.full((N, N), neg_inf)
        for i in range(N):
            C[i, :i+1] = 0 
        
        # Create matrix D
        D = torch.full((N, N), neg_inf)
        torch.diagonal(D).fill_(0)
        
        # Create matrix F
        F = torch.full((N, N), neg_inf)
        
        # Create the big matrix M of shape (HN x HN)
        M = torch.full((H * N, H * N), neg_inf)
        
        # Fill M with blocks
        for i in range(H):
            row_start = i * N
            col_start = i * N
            # Place C at the diagonal block
            M[row_start:row_start + N, col_start:col_start + N] = C
            # Fill preceding blocks with D and following blocks with F
            for j in range(i):
                M[row_start:row_start + N, (j * N):(j * N + N)] = D
            for j in range(i + 1, H):
                M[row_start:row_start + N, (j * N):(j * N + N)] = F
        
        return M


class MultistepPosePlanningPolicy(MultistepPoseNoiseModel):
    def __init__(self, config):
        super(MultistepPosePlanningPolicy, self).__init__(config['noise_model'])
        if config['noise_schedule_type'] == 'ddpm':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=config['train_steps'], 
                                                 beta_schedule=config['beta_schedule'])
        elif config['noise_schedule_type'] == 'ddim':
            self.noise_scheduler = DDIMScheduler(num_train_timesteps=config['train_steps'], 
                                                 beta_schedule=config['beta_schedule'])
        else:
            raise Exception('Invalid Noise Schedule')
    
    def get_mask(self, dof_mask):
        B = dof_mask.shape[0]
        dof_mask = dof_mask.float()
        dof_mask = dof_mask.masked_fill(dof_mask==1, float('-inf'))
        # first (max_dof+1)*horizon positions are generative
        assert dof_mask.shape[1] == self.max_dof+1
        extended_dof_mask = \
            dof_mask.unsqueeze(-1).repeat(1,1,self.num_tokens).transpose(-2,-1)

        mask = self.causal_mask.detach().clone().unsqueeze(0).repeat(B,1,1).to(dof_mask.device)
        mask[...,:-1] += extended_dof_mask.repeat(1,1,self.horizon+1)

        mask = mask.unsqueeze(1).repeat(1,self.num_transformer_heads,1,1)
        return mask
    
    def conditional_samples(self, data_dict, inference_steps=10):
        self.noise_scheduler.set_timesteps(inference_steps)
        ref_link_poses = data_dict['PA_9d']
        B = ref_link_poses.shape[0]
        rel_poses_p = torch.randn((B,(self.max_dof+1)*self.horizon,9), device=ref_link_poses.device)  
        mask = self.get_mask(data_dict['dof_mask'])
        fixed_feats = self.fixed_forward(ref_link_poses, data_dict['goal_pose_9d'])
        for t in self.noise_scheduler.timesteps:
            t = t.to(ref_link_poses.device)
            with torch.no_grad():
                noisy_residual = self.denoising_forward(fixed_feats, 
                                                        rel_poses_p, 
                                                        mask, t.unsqueeze(0))
            rel_poses_p = self.noise_scheduler.step(noisy_residual[:,:rel_poses_p.shape[1]], 
                                                    t, rel_poses_p).prev_sample
        return rel_poses_p
    
    def forward(self, data_dict):
        rel_poses = data_dict['TAB_9d_h']
        B = rel_poses.shape[0]
        rel_poses = rel_poses.view(B,-1,9)
        
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=rel_poses.device
        ).long()        
        noise = torch.randn(rel_poses.shape, device=rel_poses.device)
        noisy_rel_poses = self.noise_scheduler.add_noise(rel_poses, noise, timesteps)

        mask = self.get_mask(data_dict['dof_mask'])
        noisy_residual = super().forward(data_dict['PA_9d'], 
                                        noisy_rel_poses,
                                        data_dict['goal_pose_9d'],
                                        mask, timesteps)

        loss = F.mse_loss(noisy_residual[:,:rel_poses.shape[1]], noise, reduction='none')
        loss = loss[~data_dict['dof_mask'].repeat(1,self.horizon)]

        data_dict['loss'] = loss.mean()
        return data_dict


class MpcMultistepPosePlanningPolicy(MultistepPosePlanningPolicy):
    def __init__(self, config, robot_point_sampler, check_horizon=8):
        super().__init__(config)
        self.robot_point_sampler = robot_point_sampler
        self.color_map = np.array([[0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,0.0,0.0]])
        self.check_horizon = check_horizon
        self.batch_tensor = None
        self.action_horizon_dict = ((0.1, 2, 'slow'), (0.05, 3, 'normal'), (-1, 4, 'fast'))
    
    def get_collision_loss(self, surface_pts, query_indices, coll_model, inference=False):
        B, N = surface_pts.shape[:2]
        batch_coord = surface_pts[:,:,:3].view(-1,3)
        batch_feat = surface_pts.view(-1,4)
        if self.batch_tensor is None:
            self.batch_tensor = torch.arange(B).repeat_interleave(N).to(surface_pts.device)
        input_dict = dict(
            coord=batch_coord,
            feat=batch_feat,
            batch=self.batch_tensor,
            grid_size=0.01
        )

        if inference:
            with torch.no_grad():
                output_dict = coll_model(input_dict)
        else:
            output_dict = coll_model(input_dict)

        seg_logits = output_dict['feat'].view(B,N,2)
        labels_pred = torch.argmax(seg_logits, dim=-1)
        cost = labels_pred[query_indices].view(B,-1).float().mean(dim=-1)
        return cost, labels_pred + 1

    def get_action(self, data_dict, PA, obstacle_surface_pts, coll_model, inference_steps=10):
        self.noise_scheduler.set_timesteps(inference_steps)
        ref_link_poses = data_dict['PA_9d']
        B = ref_link_poses.shape[0]
        rel_poses_p = torch.randn((B,(self.max_dof+1)*self.horizon,9), device=ref_link_poses.device)  
        mask = self.get_mask(data_dict['dof_mask'])
        fixed_feats = self.fixed_forward(ref_link_poses, data_dict['goal_pose_9d'])
        for t in self.noise_scheduler.timesteps:
            t = t.to(ref_link_poses.device)
            with torch.no_grad():
                noisy_residual = self.denoising_forward(fixed_feats, 
                                                        rel_poses_p, 
                                                        mask, t.repeat(B))[:,:rel_poses_p.shape[1]]
            rel_poses_p = self.noise_scheduler.step(noisy_residual, t, rel_poses_p).prev_sample

        assert B==PA.shape[0]==obstacle_surface_pts.shape[0]
        assert self.check_horizon <= self.horizon
        interim_poses = rel_poses_p[~data_dict['dof_mask'].repeat(1,self.horizon)]
        interim_poses = interim_poses.view(B,self.horizon,self.robot_point_sampler.DOF+1,9)
        interim_poses = interim_poses[:,:self.check_horizon]
        interim_poses = interim_poses.reshape(-1,9)
        TAB_p = torch.zeros((B*self.check_horizon*(self.robot_point_sampler.DOF+1),4,4), device=ref_link_poses.device)
        TAB_p[:,:3,:3] = rot6d_to_rotmat(interim_poses[:,3:])
        TAB_p[:,:3,3] = interim_poses[:,:3]
        TAB_p[:,3,3] = 1
        TAB_p = TAB_p.view(B*self.check_horizon,self.robot_point_sampler.DOF+1,4,4)
        PB_p = TAB_p@PA.repeat(self.check_horizon,1,1,1)

        interim_robot_surface_pts = self.robot_point_sampler.sample_robot_points_batch(
            joint_config_batch=None, target_poses_batch=PB_p
        )
        surface_pts = torch.cat((interim_robot_surface_pts, obstacle_surface_pts.repeat(self.check_horizon,1,1)), dim=1)

        query_indices = surface_pts[:,:,3]!=0
        loss, labels_pred = self.get_collision_loss(surface_pts, 
                                                    query_indices, 
                                                    coll_model,
                                                    inference=True) 
        loss = loss.reshape(B,self.check_horizon).mean(dim=-1)
        best_idx = torch.argmin(loss)

        max_loss = torch.max(loss)
        act_horizon = 2
        for ah in self.action_horizon_dict:
            if max_loss > ah[0]:
                act_horizon = ah[1]
                # print(ah[2])
                break

        rel_poses_p = rel_poses_p[best_idx] 
        return rel_poses_p, act_horizon