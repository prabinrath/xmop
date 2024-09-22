import numpy as np
from torch.utils.data import Dataset
import h5py
from urdfpy import URDF
from pyquaternion import Quaternion
from geometrout.primitive import Cuboid, Cylinder
from common.sample_surface_points import sample_surface_points


class MpiNetDataset(Dataset):
    # Borrowed from https://github.com/NVlabs/motion-policy-networks/blob/main/mpinets/data_loader.py
    def __init__(self, trajectory_key, dataset_path, panda_urdf_path, pcd_noise=0.005, num_obstacle_points=None, sample_color=False):
        self.panda_urdf = URDF.load(panda_urdf_path)
        self.dataset_path = dataset_path
        self.trajectory_key = trajectory_key
        self.pcd_noise = pcd_noise
        self.num_obstacle_points = num_obstacle_points
        self.sample_color = sample_color
        self.cfg = {}
        for joint in self.panda_urdf.actuated_joints:
            self.cfg[joint.name] = 0.0
        del self.cfg['panda_finger_joint1']

        # global_solutions or hybrid_solutions
        with h5py.File(self.dataset_path, "r") as f:
            self.len = f[self.trajectory_key].shape[0]

    def get_scenario(self, trajectory_idx):
        obstacle_config = []
        eef_plan = []
        with h5py.File(self.dataset_path, "r") as f:
            plan_configs= f[self.trajectory_key][trajectory_idx, :, :]
            plan_configs = np.hstack((plan_configs, np.zeros((plan_configs.shape[0],1), dtype=np.float32)))
            hand_poses = self.panda_urdf.link_fk_batch(cfgs=plan_configs, link="panda_hand", use_names=True)
            cuboid_dims = f["cuboid_dims"][trajectory_idx, ...]
            if cuboid_dims.ndim == 1:
                cuboid_dims = np.expand_dims(cuboid_dims, axis=0)

            cuboid_centers = f["cuboid_centers"][trajectory_idx, ...]
            if cuboid_centers.ndim == 1:
                cuboid_centers = np.expand_dims(cuboid_centers, axis=0)

            cuboid_quats = f["cuboid_quaternions"][trajectory_idx, ...]
            if cuboid_quats.ndim == 1:
                cuboid_quats = np.expand_dims(cuboid_quats, axis=0)
            # Entries without a shape are stored with an invalid quaternion of all zeros
            # This will cause NaNs later in the pipeline. It's best to set these to unit
            # quaternions.
            # To find invalid shapes, we just look for a dimension with size 0
            cuboid_quats[np.all(np.isclose(cuboid_quats, 0), axis=1), 0] = 1

            cuboids = [
                Cuboid(c, d, q)
                for c, d, q in zip(
                    list(cuboid_centers), list(cuboid_dims), list(cuboid_quats)
                )
            ]

            # Filter out the cuboids with zero volume
            cuboids = [c for c in cuboids if not c.is_zero_volume()]

            max_area_idx = 0
            max_area = 0
            for idx, cuboid in enumerate(cuboids):
                obstacle_config.append(dict(type='cuboid', 
                            translation=cuboid.center,
                            orientation=cuboid.pose.so3.q,
                            scale=cuboid.dims,
                            # color=np.random.rand(3,)))
                color=np.array([0.5, 0.4, 0.4])))
                if cuboid.surface_area > max_area:
                    max_area = cuboid.surface_area
                    max_area_idx = idx
            obstacle_config[max_area_idx]['color'] = np.array([0.75, 0.75, 0.75])

            if "cylinder_radii" not in f.keys():
                # Create a dummy cylinder if cylinders aren't in the hdf5 file
                cylinder_radii = np.array([[0.0]])
                cylinder_heights = np.array([[0.0]])
                cylinder_centers = np.array([[0.0, 0.0, 0.0]])
                cylinder_quats = np.array([[1.0, 0.0, 0.0, 0.0]])
            else:
                cylinder_radii = f["cylinder_radii"][trajectory_idx, ...]
                if cylinder_radii.ndim == 1:
                    cylinder_radii = np.expand_dims(cylinder_radii, axis=0)
                cylinder_heights = f["cylinder_heights"][trajectory_idx, ...]
                if cylinder_heights.ndim == 1:
                    cylinder_heights = np.expand_dims(cylinder_heights, axis=0)
                cylinder_centers = f["cylinder_centers"][trajectory_idx, ...]
                if cylinder_centers.ndim == 1:
                    cylinder_centers = np.expand_dims(cylinder_centers, axis=0)
                cylinder_quats = f["cylinder_quaternions"][trajectory_idx, ...]
                if cylinder_quats.ndim == 1:
                    cylinder_quats = np.expand_dims(cylinder_quats, axis=0)
                # Ditto to the comment above about fixing ill-formed quaternions
                cylinder_quats[np.all(np.isclose(cylinder_quats, 0), axis=1), 0] = 1
            
            cylinders = [
                Cylinder(c, r, h, q)
                for c, r, h, q in zip(
                    list(cylinder_centers),
                    list(cylinder_radii.squeeze(1)),
                    list(cylinder_heights.squeeze(1)),
                    list(cylinder_quats),
                )
            ]
            cylinders = [c for c in cylinders if not c.is_zero_volume()]
            
            for cylinder in cylinders:
                obstacle_config.append(dict(type='cylinder',
                            translation=cylinder.center,
                            orientation=cylinder.pose.so3.q,
                            radius=cylinder.radius.item(),
                            height=cylinder.height.item(),
                            # color=np.random.rand(3,)))
                color=np.array([0.5, 0.4, 0.4])))
            
            if self.num_obstacle_points is not None:
                primitives = cuboids + cylinders
                obstacle_surface_pts, _ = self.construct_mixed_point_cloud(
                    primitives, self.num_obstacle_points, noise=self.pcd_noise)
                if self.sample_color:
                    obstacle_color_pts = np.ones((obstacle_surface_pts.shape[0], 3))*np.asarray((1.0,0.0,0.0))
                else:
                    obstacle_color_pts = None
            else:
                obstacle_surface_pts = None
                obstacle_color_pts = None

            for hand_pose in hand_poses:
                target_position = hand_pose[:3,3]
                qt = Quaternion(matrix=hand_pose[:3,:3])
                target_orientation = np.asarray([qt.w, qt.x, qt.y, qt.z], dtype=np.float32)
                eef_plan.append((target_position, target_orientation))

        return (obstacle_surface_pts, obstacle_color_pts), obstacle_config, eef_plan

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.get_scenario(index)

    def construct_mixed_point_cloud(self, primitives, num_points, noise=0.005, semantic_labels=None, colors=None, separated=False):
        """
        Creates a random point cloud from a collection of primitives. The points in
        the point cloud should be fairly(-ish) distributed amongst the primitives based
        on their surface area.

        :param primitives Sequence[Union[Sphere, Cuboid, Cylinder]]: The primitives in the scene
        :param num_points int: The total number of points in the samples scene (not
                            the number of points per obstacle)
        :param noise float: Noise to add to the sampled pointcloud
        :param semantic_labels np.ndarray: Semantic labels for the individual primitives
        :param colors: Color of the pointcloud
        :param separated: Link specific pointcloud or merged semantic cloud
        :rtype (np.ndarray, np.ndarray): pointcloud (N,4), colorcloud (N,3)
        """
        # keeping this alias for backward compatibility
        return sample_surface_points(primitives, num_points, noise, semantic_labels, colors, separated)
        