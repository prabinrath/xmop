import torch
from torch.utils.data import DataLoader
from training.xmop_dataset import XMoPDataset
from xmop.xmop_planning_policy import MultistepPosePlanningPolicy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from common.redirect_stream import RedirectStream
from base_models.ema_model import EMAmodel
from pathlib import Path
import argparse
import copy
import yaml
import wandb
import os


LOGGING = True
device = 'cuda'
experiment_name = 'XMoP_Policy'
experiment_notes = 'multistep planning, hidden size 512, diffusion policy, 20 epochs'
store_dir = ''


def main(args):
    traj_dataset_root = os.path.join(store_dir, 'resources/datasets/traj_dataset')
    ckpt_dir = os.path.join(store_dir, 'checkpoints', experiment_name)
    log_dir = os.path.join(store_dir, 'log', experiment_name)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    with open("config/xmop_planning_policy.yaml") as file:
        model_config = yaml.safe_load(file)
        model = MultistepPosePlanningPolicy(model_config).to(device)
        model.train()
        ema_model = EMAmodel(model=copy.deepcopy(model))

        if LOGGING:
            wandb.init(
                project="neural-planning",
                config=model_config,
                name=experiment_name,
                notes=experiment_notes
            )

    dataset  = XMoPDataset(
                    traj_dataset_root=traj_dataset_root,
                    n_dof_template_path='urdf/n_dof_template.xacro',
                    traj_index=(0,3270000),
                    max_len=int(3270000),
                    horizon=model.horizon,
                    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            pin_memory=True, 
                            num_workers=args.num_workers)

    # print(model)
    # nbr_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(" The model has {:d} parameters".format(nbr_param))

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    lr_scheduler = LinearLR(optimizer, 1, 0.1, len(dataloader))

    with RedirectStream(os.path.join(log_dir, 'console.log')):
        step=0
        for epoch in range(args.epochs):
            print(f'Epoch {epoch+1}')
            for data_dict_batch in dataloader:
                data_dict_batch = {k: v.to(device, non_blocking=True) \
                                    for k, v in data_dict_batch.items()}
                loss = model(data_dict_batch)['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                ema_model.step(model)
                
                step+=1
                if LOGGING and step%args.log_every==0:
                    step_log = {
                        'train_loss': loss.item(),
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    wandb.log(step_log)
                    
                if step%args.ckpt_every==0:
                    checkpoint = {"ema_model": ema_model.state_dict(),
                                  "model": model.state_dict()}
                    torch.save(checkpoint, os.path.join(ckpt_dir, f'{experiment_name}_{step}.pth'))
                
    checkpoint = {"ema_model": ema_model.state_dict(),
                  "model": model.state_dict()}
    torch.save(checkpoint, os.path.join(ckpt_dir, f'{experiment_name}_terminal.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=str, default=64)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    main(args)