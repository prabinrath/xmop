import torch
from training.xcod_dataset import XCoDDataset
from common import RedirectStream
from torch.utils.data import DataLoader
from xcod import XCoD
from base_models.losses import CrossEntropyLoss
from base_models.losses import LovaszLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import argparse
from pathlib import Path
import os


LOGGING = True
device = 'cuda'
experiment_name = 'XCoD_Collision_Model'
experiment_notes = 'PTv3 backbone, 1M scenes'
store_dir = ''

def main(args):
    mpinet_dataset_path = os.path.join(store_dir, 'resources/datasets/mpinet_dataset/train.hdf5')
    traj_dataset_root = os.path.join(store_dir, 'resources/datasets/traj_dataset')
    coll_dataset_root = os.path.join(store_dir, 'resources/datasets/coll_dataset')
    ckpt_dir = os.path.join(store_dir, 'checkpoints', experiment_name)
    log_dir = os.path.join(store_dir, 'log', experiment_name)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if LOGGING:
        wandb.init(
            project="neural-collision",
            config={
            "architecture": "PTv3",
            "lr": 0.0005,
            "lr_schedule": "CosineAnnealingLR"
            },
            name=experiment_name,
            notes=experiment_notes
        )

    dataset = XCoDDataset('global_solutions', 
                            mpinet_dataset_path=mpinet_dataset_path, 
                            traj_dataset_root=traj_dataset_root,
                            coll_dataset_root=coll_dataset_root,
                            panda_urdf_path='urdf/franka_panda/panda.urdf', 
                            n_dof_template_path='urdf/n_dof_template.xacro',
                            traj_index=(0, 3270000),
                            num_obstacle_points=4096*4,
                            num_robot_points=4096*2,
                            max_len=int(args.max_len),
                            )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            pin_memory=True, num_workers=args.num_workers)

    model = XCoD( 
                stride=[2, 2],
                enc_depths=[2, 4, 2],
                enc_channels=[32, 64, 128],
                enc_num_head=[2, 4, 8],
                enc_patch_size=[256, 256, 256],
                dec_depths=[2, 2],
                dec_channels=[32, 64],
                dec_num_head=[4, 8],
                dec_patch_size=[256, 256]).to('cuda')
    model.train()

    # print(model)
    # nbr_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(" The model has {:d} parameters".format(nbr_param))

    ce_loss_fn = CrossEntropyLoss(loss_weight=1.0, ignore_index=-1)
    lz_loss_fn = LovaszLoss(mode="multiclass", loss_weight=1.0, ignore_index=-1)
    optimizer = AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer=optimizer, 
                                T_max=len(dataloader),
                                eta_min=0.00005)

    with RedirectStream(os.path.join(log_dir, 'console.log')):
        step=0
        for surface_pts, labels in dataloader:
            B, N = surface_pts.shape[:2]
            batch_coord = surface_pts[:,:,:3].view(-1,3)
            batch_feat = surface_pts.view(-1,4)
            input_dict = dict(
                coord=batch_coord.to('cuda').float(),
                feat=batch_feat.to('cuda').float(),
                batch=torch.arange(B).repeat_interleave(N).to('cuda'),
                grid_size=0.01
            )
            labels = labels.to('cuda').view(-1).long()

            optimizer.zero_grad()

            output_dict = model(input_dict)
            seg_logits = output_dict['feat']
            ce_loss = ce_loss_fn(seg_logits, labels)
            lz_loss = lz_loss_fn(seg_logits, labels)
            loss = ce_loss + lz_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            step+=1
            if LOGGING and step%args.log_every==0:
                wandb.log({'net_loss': loss.item(),
                        'ce_loss': ce_loss.item(),
                        'lz_loss': lz_loss.item()})
            if step%args.ckpt_every==0:
                checkpoint = {"model": model.state_dict()}
                torch.save(checkpoint, os.path.join(ckpt_dir, f'{experiment_name}_{step}.pth'))

    checkpoint = {"model": model.state_dict()}
    torch.save(checkpoint, os.path.join(ckpt_dir, f'{experiment_name}_terminal.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=str, default=12)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=1e6)
    args = parser.parse_args()
    main(args)