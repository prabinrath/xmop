import io
import numpy as np
from mpinet_dataset import MpiNetDataset
from common import TrajDataManager, CollisionDataManager
from urdf import NDofGenerator
from urdfpy import URDF
from geometrout.primitive import Cylinder, Cuboid
from pyquaternion import Quaternion


class XCoDDataset(MpiNetDataset):
    def __init__(self, trajectory_key, 
                 mpinet_dataset_path, 
                 traj_dataset_root, 
                 coll_dataset_root,
                 panda_urdf_path, 
                 n_dof_template_path,
                 traj_index,
                 coll_data_prob = 0.5,
                 num_obstacle_points=4096*4,
                 num_robot_points=4096*2,
                 obs_pcd_noise = 0.005,
                 max_len=None):
        super(XCoDDataset, self).__init__(trajectory_key, mpinet_dataset_path, panda_urdf_path, 
                                            pcd_noise = obs_pcd_noise,
                                            num_obstacle_points = num_obstacle_points)
        self.traj_mgr = TrajDataManager(traj_dataset_root, traj_index[0], traj_index[1])
        if max_len is not None:
            self.success_indices = np.random.choice(self.traj_mgr.success_indices, max_len, replace=False)
        else:
            self.success_indices = self.traj_mgr.success_indices
        self.coll_mgr = CollisionDataManager(coll_dataset_root, traj_index[0], traj_index[1])
        self.ndof_generator = NDofGenerator(template_path=n_dof_template_path,
                                            joint_gap=0.005, base_axis=2, base_offset=0.03)
        self.num_obstacle_points = num_obstacle_points
        self.num_robot_points = num_robot_points
        self.coll_data_prob = coll_data_prob

    def get_primitive(self, link_pose, center, link_collision):
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
    
    def __len__(self):
        return self.success_indices.shape[0]

    def __getitem__(self, index):
        mpinet_idx = self.success_indices[index]
        (obstacle_surface_pts,_), _, _ = super().get_scenario(mpinet_idx)
        dof, kinematics, dynamics, traj = self.traj_mgr.retrieve_trajectory(mpinet_idx)
        urdf_text = self.ndof_generator.get_urdf(kinematics, dynamics)
        urdf_io_handle = io.BytesIO(initial_bytes=urdf_text)
        urdf_io_handle.name = 'n_dof_robot'
        urdf_handle = URDF.load(urdf_io_handle)

        if np.random.rand() < self.coll_data_prob:
            joint_config, collision_links = self.coll_mgr.retrieve_collision(mpinet_idx)
        else:
            joint_config = np.copy(traj[np.random.randint(traj.shape[0])])
            collision_links = np.zeros((7,))

        # generate pointcloud observations
        fk_dict = urdf_handle.link_fk(cfg=joint_config[:dof])
        manip_primitives = []
        semantic_labels = []
        semantic_labels_dict = {}
        for link in urdf_handle.links:
            link_pose = fk_dict[link]
            # we can also use visual_geometry_fk here, it has exact same logic
            link_pose = link_pose @ link.collisions[0].origin
            center = link_pose[:3,3]
            link_id = int(link.name[6])
            manip_primitives.append(self.get_primitive(link_pose, center, link.collisions[0]))
            semantic_labels.append(link_id)
            if link_id>0:
                semantic_labels_dict[link_id] = collision_links[link_id-1]
        semantic_labels_dict[0] = -1 # ignore base link collision

        manip_surface_pts, _ = self.construct_mixed_point_cloud(manip_primitives, 
                                self.num_robot_points, noise=0.0, semantic_labels=semantic_labels)
        semantic_labels = manip_surface_pts[:,3]
        manip_labels = np.zeros_like(semantic_labels)
        for key in semantic_labels_dict:
            manip_labels[semantic_labels==key] = semantic_labels_dict[key]
        
        surface_pts = np.zeros((self.num_robot_points+self.num_obstacle_points,4))
        surface_pts[:self.num_robot_points] = manip_surface_pts
        surface_pts[self.num_robot_points:] = obstacle_surface_pts
        labels = np.zeros((self.num_robot_points+self.num_obstacle_points,))
        labels[:self.num_robot_points] = manip_labels
        labels[self.num_robot_points:] = -1 * np.ones((self.num_obstacle_points,)) # ignore obstacles

        # np.save('surface_pts.npy', surface_pts)
        # np.save('labels.npy', labels)
        return surface_pts, labels
