import torch
from benchmark import BulletCollisionChecker
from common import RealRobotPointSampler
from benchmark import PlanningBenchmarker
from xcod import XCoD
from torchmetrics import Precision, Recall, F1Score, JaccardIndex
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy
import numpy as np
import time
import yaml
from tqdm import tqdm

if __name__=='__main__':
    problem_handle = PlanningBenchmarker(robot_name='kinova7', num_obstacle_points=4096*4)
    sim_handle = BulletCollisionChecker(problem_handle.urdf_path, gui=False)

    coll_model = XCoD(
        stride=[2, 2],
        enc_depths=[2, 4, 2],
        enc_channels=[32, 64, 128],
        enc_num_head=[2, 4, 8],
        enc_patch_size=[256, 256, 256],
        dec_depths=[2, 2],
        dec_channels=[32, 64],
        dec_num_head=[4, 8],
        dec_patch_size=[256, 256],
        checkpoint_path='checkpoints/ptv3_semantic_mini_best.pth',
        ).to('cuda')
    coll_model.eval()

    with open("config/robot_point_sampler.yaml") as file:
        robot_point_sampler = RealRobotPointSampler(
            urdf_path=problem_handle.urdf_path, 
            config=yaml.safe_load(file)['xcod_ik'],
            num_robot_points=4096*2,
            device='cpu')

    average = 'macro'
    precision_metric = Precision(task="multiclass", average=average, num_classes=2).to('cuda')
    recall_metric = Recall(task="multiclass", average=average, num_classes=2).to('cuda')
    f1score_metric = F1Score(task="multiclass", average=average, num_classes=2).to('cuda')
    coll_det_precision_metric = BinaryPrecision().to('cuda')
    coll_det_recall_metric = BinaryRecall().to('cuda')
    coll_det_accuracy_metric = BinaryAccuracy().to('cuda')
    iou_metric = JaccardIndex(task="multiclass", num_classes=2).to('cuda')

    coll_preds, coll_labels = [], []
    durations = []
    for idx in tqdm(range(len(problem_handle))):
        obstacle_surface_pts, obstacle_config, _, _, _, _ = problem_handle.get_problem(idx)
        sim_handle.load_obstacles(obstacle_config)

        # sample equal number of collision and collision free trajectories
        coll_label = np.random.rand() < 0.5
        semantic_labels = np.zeros([robot_point_sampler.DOF+1,])
        if coll_label:
            # we need a collision js here
            js = np.random.uniform(robot_point_sampler.jl_limits, robot_point_sampler.ju_limits)
            while not sim_handle.in_collision(js):
                js = np.random.uniform(robot_point_sampler.jl_limits, robot_point_sampler.ju_limits)
            collision_labels = np.asarray(list(sim_handle.in_collision_complete(js)), dtype=np.int64) + 1
            semantic_labels[collision_labels] = 1
            semantic_labels[0] = 0 # base never in collision
        else:
            # we need a collision free js here
            js = np.random.uniform(robot_point_sampler.jl_limits, robot_point_sampler.ju_limits)
            while sim_handle.in_collision(js):
                js = np.random.uniform(robot_point_sampler.jl_limits, robot_point_sampler.ju_limits)
        
        manip_surface_pts = robot_point_sampler.sample_robot_points_batch(
            torch.from_numpy(js).unsqueeze(0)).squeeze()
        labels = torch.zeros_like(manip_surface_pts[:,3])
        for idx in range(robot_point_sampler.DOF):
            labels[manip_surface_pts[:,3]==idx] = semantic_labels[idx]
        labels = torch.cat((labels, torch.zeros(obstacle_surface_pts.shape[0],)))

        surface_pts = torch.cat((manip_surface_pts, 
                                torch.as_tensor(obstacle_surface_pts).float()))
        N, _ = surface_pts.shape[:2]
        batch_coord = surface_pts[:,:3]
        batch_feat = surface_pts
        input_dict = dict(
            coord=batch_coord.to('cuda').float(),
            feat=batch_feat.to('cuda').float(),
            batch=torch.arange(1).repeat_interleave(N).to('cuda'),
            grid_size=0.01
        )
        labels = labels.to('cuda').long()
        
        query_indices = surface_pts[:,3]!=0
        with torch.no_grad():
            start = time.perf_counter()
            output_dict = coll_model(input_dict)
            durations.append(time.perf_counter()-start)
            seg_logits = output_dict['feat']

        # metrics aggregation
        preds, target = seg_logits[query_indices], labels[query_indices]
        precision_metric(preds, target)
        recall_metric(preds, target)
        f1score_metric(preds, target)
        iou_metric(preds, target)

        cost_pred = torch.sum(torch.argmax(preds, dim=1))/preds.shape[0]
        cost_label = torch.sum(target)/target.shape[0]
        coll_pred = 1 if cost_pred > 0.001 else 0
        coll_label = int(coll_label)
        coll_preds.append(coll_pred)
        coll_labels.append(coll_label)

        sim_handle.remove_obstacles()

    # metrics evaluation
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1score = f1score_metric.compute()
    iou = iou_metric.compute()
    print(f'Pointwise - Precision: {precision}, Recall: {recall}, F1score: {f1score}, IoU: {iou}')

    coll_det_precision = coll_det_precision_metric(torch.tensor(coll_preds),torch.tensor(coll_labels))
    coll_det_recall = coll_det_recall_metric(torch.tensor(coll_preds),torch.tensor(coll_labels))
    coll_det_accuracy = coll_det_accuracy_metric(torch.tensor(coll_preds),torch.tensor(coll_labels))
    print(f'Collision Detection - Precesion: {coll_det_precision}, Recall: {coll_det_recall}, Accuracy: {coll_det_accuracy}')

    print(f'Average duration: {sum(durations)/len(durations)}')