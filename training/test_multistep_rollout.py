import torch
from mpinet_dataset import MpiNetDataset
from xmop.xmop_planning_policy import MultistepPosePlanningPolicy
from xmop.js_retrieval import JointStateRetriever
from common import BulletRobotEnv, TrajDataManager
from urdf import NDofGenerator
from urdfpy import URDF
import yaml
import numpy as np
import tempfile
from collections import defaultdict
from common.rotation_torch import rot6d_to_rotmat, quat_to_rotmat

device = 'cuda'

mpinet_dataset = MpiNetDataset('global_solutions', 
                               'resources/datasets/mpinet_dataset/train.hdf5', 
                               'urdf/franka_panda/panda.urdf', 
                               num_obstacle_points=None,
                               sample_color=True)
random_indices = np.random.choice(len(mpinet_dataset), 10, replace=False)

traj_mgr = TrajDataManager('resources/datasets/traj_dataset', 0, 3270000)
ndof_generator = NDofGenerator(template_path='urdf/n_dof_template.xacro',
                                    joint_gap=0.005, base_axis=2, base_offset=0.03)

with open("config/xmop_planning_policy.yaml") as file:
    model_config = yaml.safe_load(file)
    PREDICT_HORIZON = model_config['noise_model']['horizon']
    model = MultistepPosePlanningPolicy(model_config).to(device)
    checkpoint = torch.load('checkpoints/XMoP_Policy/XMoP_Policy_terminal.pth')
    model.load_state_dict(checkpoint['ema_model'])
    model.eval()

MAX_ROLLOUT_STEPS = 300
ACTION_HORIZON = 3
ALPHA = 0.25
sim_handle = BulletRobotEnv(gui=True)

for idx in random_indices:
    (obstacle_surface_pts, _), obstacle_config, eef_plan = mpinet_dataset.get_scenario(idx)
    # eef goal is non-hindsight, hence unseen during training
    goal_eef = eef_plan[-1]
    sim_handle.set_dummy_state(goal_eef[0], goal_eef[1])

    dof, kinematics, dynamics, traj = traj_mgr.retrieve_trajectory(idx)
    urdf_text = ndof_generator.get_urdf(kinematics, dynamics)
    with tempfile.NamedTemporaryFile(suffix='.urdf') as file:
        file.write(urdf_text)
        urdf_handle = URDF.load(file.name)
        sim_handle.load_robot(file.name)
    js_ret = JointStateRetriever(urdf_handle, cost_th=0.5)
    start_js = np.copy(traj[0])

    # assign robot frames
    manip_semantic_link_map = defaultdict(list)
    for idx, link in enumerate(urdf_handle.links):
        manip_semantic_link_map[int(link.name[6])].append(link)
    fk_link_indices = [np.random.choice(len(manip_semantic_link_map[key]))
                           for key in range(dof)]
    fk_link_indices.append(1) # gripper
    # fk_link_indices = [0,]*(dof+1)

    # generate start state pointcloud
    sim_handle.marionette_robot(start_js)
    curr_js = np.copy(start_js)
    
    # get target pose
    goal_ee_pose = torch.as_tensor(np.concatenate(goal_eef, axis=0), 
                    dtype=torch.float32)
    goal_rot_mat = quat_to_rotmat(goal_ee_pose[3:].unsqueeze(0)).squeeze()
    goal_pose_9d = torch.cat((goal_ee_pose[:3], 
                              goal_rot_mat[:,0],
                              goal_rot_mat[:,1]))

    # assign dof mask
    dof_mask = np.full((model.max_dof+1), True)
    dof_mask[:dof] = False
    dof_mask[-1] = False

    # prepare data dict
    data_dict_batch = dict(
        goal_pose_9d=goal_pose_9d.cuda().unsqueeze(0),
        dof_mask=torch.from_numpy(dof_mask).cuda().unsqueeze(0),
    )    

    # rollout policy
    for _ in range(MAX_ROLLOUT_STEPS):
        fk_dict = urdf_handle.link_fk(cfg=curr_js, use_names=True)
        PA = []
        for key, idx in zip(list(range(dof+1)), fk_link_indices):
            link = manip_semantic_link_map[key][idx]
            PA.append((fk_dict[link.name]@link.visuals[0].origin)[None,:,:])
        PA = np.concatenate(PA)
        PA_9d = np.zeros((model.max_dof+1,9))
        T_9d = np.concatenate((PA[:,:3,3], PA[:,:3,0], PA[:,:3,1]), axis=1)
        PA_9d[:dof] = T_9d[:dof]
        PA_9d[-1] = T_9d[dof]
        PA_9d = PA_9d.astype(np.float32)
        data_dict_batch['PA_9d'] = torch.from_numpy(PA_9d).cuda().unsqueeze(0)

        with torch.no_grad():
            rel_poses_p = model.conditional_samples(data_dict_batch)
        rel_poses_p = rel_poses_p[~data_dict_batch['dof_mask'].repeat(1,PREDICT_HORIZON)]

        TAB_p = torch.zeros(((dof+1)*PREDICT_HORIZON,4,4), device=device)
        TAB_p[:,:3,:3] = rot6d_to_rotmat(rel_poses_p[:,3:])
        TAB_p[:,:3,3] = rel_poses_p[:,:3]
        TAB_p[:,3,3] = 1
        TAB_p = TAB_p.view(PREDICT_HORIZON,dof+1,4,4)
        PB_p = TAB_p.cpu().numpy()@PA

        for h in range(ACTION_HORIZON):
            target_poses_9d = {}
            for key, idx, pose in zip(list(range(dof+1)), fk_link_indices, PB_p[h]):
                link = manip_semantic_link_map[key][idx]
                target_poses_9d[link.visuals[0].geometry] = \
                    np.concatenate((pose[:3,3], pose[:3,0], pose[:3,1]))
            joint_config_n_p = js_ret.retrieve_js(target_poses_9d, curr_js)
            curr_js = ALPHA * curr_js + (1-ALPHA) * joint_config_n_p
            sim_handle.marionette_robot(curr_js)

        fk_dict = urdf_handle.link_fk(cfg=curr_js, use_names=True)  
        curr_ee_link = manip_semantic_link_map[dof][fk_link_indices[-1]]
        curr_pose = fk_dict[curr_ee_link.name]@curr_ee_link.visuals[0].origin
        curr_pose_9d = torch.as_tensor(np.concatenate((curr_pose[:3,3], 
                                       curr_pose[:3,0],
                                       curr_pose[:3,1])), dtype=torch.float32)
        goal_ik_dist = torch.linalg.norm(goal_pose_9d-curr_pose_9d)
        if goal_ik_dist < 0.01:
            break
        
    sim_handle.clear_scene()  