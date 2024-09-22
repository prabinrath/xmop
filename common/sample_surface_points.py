import numpy as np
from collections import defaultdict


def sample_surface_points(primitives, num_points, noise=0.005, semantic_labels=None, colors=None, separated=False):
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

    total_primitives = len(primitives)
    assert total_primitives > 0
    
    if semantic_labels is None:
        semantic_labels = np.zeros((total_primitives,))

    # Allocate points based on obstacle surface area for even sampling
    surface_areas = np.array([o.surface_area for o in primitives])
    total_area = np.sum(surface_areas)
    proportions = (surface_areas / total_area).tolist()

    if separated:
        surface_points_map = defaultdict(list)
        for i, (o, prop) in enumerate(zip(primitives, proportions)):
            sample_number = int(prop * num_points) + 500
            _points = o.sample_surface(sample_number)
            surface_points_map[semantic_labels[i]].append(_points)
        surface_points = []
        for key in sorted(surface_points_map.keys()):
            link_pcd = np.concatenate(surface_points_map[key], axis=0)
            surface_points.append(link_pcd)
        return surface_points, None
    else:
        surface_points = []
        sample_numbers = []
        for i, (o, prop) in enumerate(zip(primitives, proportions)):
            sample_number = int(prop * num_points) + 500
            sample_numbers.append(sample_number)
            _points = semantic_labels[i] * np.ones((sample_number, 4))
            _points[:,:3] = o.sample_surface(sample_number)
            surface_points.append(_points)
        surface_points = np.concatenate(surface_points, axis=0)

        # Downsample to the desired number of surface_points
        indices = np.random.choice(surface_points.shape[0], num_points, replace=False)
        surface_points = surface_points[indices, :]
        # adding observation gaussian noise
        if noise > 0:
            surface_points[:,:3] += noise * np.random.randn(surface_points.shape[0],3)
        
        if colors is not None:
            color_points = []
            for i, sample_number in enumerate(sample_numbers):
                color_points.append(np.ones((sample_number,3))*np.asarray(colors[i]))
            color_points = np.concatenate(color_points, axis=0)[indices, :]
            return surface_points, color_points

        return surface_points, None