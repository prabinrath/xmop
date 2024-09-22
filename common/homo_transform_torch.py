import torch

def transform_point_cloud_torch(pc, transformation_matrix, vector=False, in_place=True):
        """
        Parameters
        ----------
        pc: A pytorch tensor pointcloud, maybe with some addition dimensions.
            This should have shape N x [3 + M] where N is the number of points
            M could be some additional mask dimensions or whatever, but the
            3 are x-y-z
        transformation_matrix: A 4x4 homography
        vector: Whether or not to apply the translation

        Returns
        -------
        Mutates the pointcloud in place and transforms x, y, z according the homography

        """
        assert isinstance(pc, torch.Tensor)
        assert type(pc) == type(transformation_matrix)
        assert pc.ndim == transformation_matrix.ndim
        if pc.ndim == 3:
            N, M = 1, 2
        elif pc.ndim == 2:
            N, M = 0, 1
        else:
            raise Exception("Pointcloud must have dimension Nx3 or BxNx3")
        xyz = pc[..., :3]
        ones_dim = list(xyz.shape)
        ones_dim[-1] = 1
        ones_dim = tuple(ones_dim)
        if vector:
            homogeneous_xyz = torch.cat(
                (xyz, torch.zeros(ones_dim, device=xyz.device)), dim=M
            )
        else:
            homogeneous_xyz = torch.cat(
                (xyz, torch.ones(ones_dim, device=xyz.device)), dim=M
            )
        transformed_xyz = torch.matmul(
            transformation_matrix, homogeneous_xyz.transpose(N, M)
        )
        if in_place:
            pc[..., :3] = transformed_xyz[..., :3, :].transpose(N, M)
            return pc
        return torch.cat((transformed_xyz[..., :3, :].transpose(N, M), pc[..., 3:]), dim=M)
