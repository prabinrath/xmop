import time
import numpy as np
from training import MpiNetDataset
from common import TrajDataManager, BulletRobotEnv
from common.o3d_viz import Open3DVisualizer
from pyquaternion import Quaternion
from urdf import NDofGenerator
from urdfpy import URDF
from geometrout.primitive import Cylinder, Cuboid
from pyquaternion import Quaternion
import tempfile

mpinet_dataset_path = 'resources/datasets/mpinet_dataset/train.hdf5'
panda_urdf_path = 'urdf/franka_panda/panda.urdf'
n_dof_template_path = 'urdf/n_dof_template.xacro'

sim_handle = BulletRobotEnv(gui=True, floor=True, rotate_camera=True)
mpinet_dataset = MpiNetDataset('global_solutions', 
                               mpinet_dataset_path, 
                               panda_urdf_path, 
                               num_obstacle_points=4096*4,
                               sample_color=True)
start_idx, end_idx = 0, 3270000

## uncomment following lines to merge the dataset fragments 
# traj_mgr = TrajDataManager('resources/datasets/traj_dataset/', start_idx, end_idx, mode='w')
# traj_mgr.merge_traj_datasets('resources/datasets/traj_dataset/temp/')
# del traj_mgr

traj_mgr = TrajDataManager('resources/datasets/traj_dataset/', start_idx, end_idx)
ndof_generator = NDofGenerator(template_path=n_dof_template_path,
                                            joint_gap=0.005, base_axis=2, base_offset=0.03)

def get_primitive(link_pose, center, link_collision):
    q = Quaternion(matrix=link_pose[:3,:3])
    q = np.asarray([q.w, q.x, q.y, q.z])
    if link_collision.geometry._cylinder is not None:
        height = link_collision.geometry._cylinder.length
        radius = link_collision.geometry._cylinder.radius
        return Cylinder(center, radius, height, q)
    if link_collision.geometry._box is not None:
        dims = link_collision.geometry._box.size
        return Cuboid(center, dims, q)
    return None

viz_indices = np.random.choice(traj_mgr.success_indices, 10, replace=False)

for idx in viz_indices:
    try:
        (obstacle_surface_pts, obstacle_color_pts), obstacle_config, _ = mpinet_dataset[idx]
        dof, kinematics, dynamics, traj = traj_mgr.retrieve_trajectory(idx)
    except:
        print('Skipping this Environment')
        continue
    urdf_text = ndof_generator.get_urdf(kinematics, dynamics)

    sim_handle.load_obstacles(obstacle_config)
    with tempfile.NamedTemporaryFile(suffix='.urdf') as file:
        file.write(urdf_text)
        urdf_handle = URDF.load(file.name)
        sim_handle.load_robot(file.name)
    assert sim_handle.DOF == len(urdf_handle.actuated_joints)
    ee_pose = urdf_handle.link_fk(cfg=traj[-1], use_names=True)[f'link_g{dof}1']
    target = (ee_pose[:3,3], Quaternion(matrix=ee_pose[:3,:3]).q)
    sim_handle.set_dummy_state(target[0], target[1])
    time.sleep(0.5)

    # visualize n_dof plan
    for js in traj:
        sim_handle.marionette_robot(js)
        time.sleep(0.1)
    time.sleep(0.5)
    sim_handle.clear_scene()

    # create target pointcloud with Hindsight Goal Rivision (HGR)
    fk_dict = urdf_handle.link_fk(cfg=traj[-1])
    target_primitives = []
    semantic_labels = []
    colors = []
    for link in urdf_handle.links:
         if link.name[5] == 'g':
            link_pose = fk_dict[link]
            # see the batch fk comments below for more detail
            link_pose = link_pose @ link.collisions[0].origin
            center = link_pose[:3,3]
            link_id = 8 # can be any number greater than the max dof of the sampled robots
            target_primitives.append(get_primitive(link_pose, center, link.collisions[0]))
            semantic_labels.append(link_id)
            colors.append(np.asarray([0.0, 1.0, 0.0])) # green target
    colors = np.vstack(colors)
    target_surface_pts, target_color_pts = mpinet_dataset.construct_mixed_point_cloud(target_primitives, 
                                            1024, noise=0.0, semantic_labels=semantic_labels, colors=colors)

    # generate pointcloud observations
    viz_handle = Open3DVisualizer(window_name="Pointcoud Training Data")
    fk_dict_batch = urdf_handle.link_fk_batch(cfgs=traj)
    semantic_color_map = [
        [1.0,   0.0,   0.0],
        [0.514, 0.545, 0.682],
        [0.675, 0.882, 0.941],
        [0.427, 0.573, 0.871],
        [0.682, 0.698, 0.718],
        [0.4, 0.812, 0.949],
        [0.435, 0.471, 0.494],
        [0.635, 0.667, 0.992],
        ]
    for idx in range(traj.shape[0]):
        manip_primitives = []
        semantic_labels = []
        colors = []
        for link in urdf_handle.links:
            link_pose = fk_dict_batch[link][idx]
            # we can also use visual_geometry_fk here, it has exact same logic
            link_pose = link_pose @ link.visuals[0].origin
            center = link_pose[:3,3]
            link_id = int(link.name[6])
            manip_primitives.append(get_primitive(link_pose, center, link.visuals[0]))
            semantic_labels.append(link_id)
            colors.append(semantic_color_map[link_id])

        colors = np.vstack(colors)
        manip_surface_pts, manip_color_pts = mpinet_dataset.construct_mixed_point_cloud(manip_primitives, 
                                            4096*2, noise=0.0, semantic_labels=semantic_labels, colors=colors)
        surface_pts = np.vstack((obstacle_surface_pts, target_surface_pts, manip_surface_pts))
        color_pts = np.vstack((obstacle_color_pts, target_color_pts, manip_color_pts))

        viz_handle.update_buffers(surface_pts[:,:3], color_pts)
        time.sleep(0.05)
    print("INFO: Close the Pointcoud Training Data window for next scenario")
    viz_handle.vis.run()
    del viz_handle
