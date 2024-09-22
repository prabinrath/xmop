import open3d as o3d
import numpy as np

class Open3DVisualizer():
    def __init__(self, window_name='XMoP', point_size=1., background_color=np.asarray([0, 0, 0])):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name)
        opt = self.vis.get_render_option()
        opt.background_color = background_color
        opt.point_size = point_size
        self.pcd = None
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        self.vis.add_geometry(mesh_frame)
    
    def __del__(self):
        self.vis.destroy_window()

    def update_buffers(self, surface_pts, color_pts=None):
        if self.pcd is not None:
            self.vis.remove_geometry(self.pcd)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(surface_pts)
        if color_pts is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(color_pts)
        self.vis.add_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def show_frames(self, frame_poses):
        for pose in frame_poses:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1., origin=[0, 0, 0])
            mesh_frame.transform(pose)
            self.vis.add_geometry(mesh_frame)

if __name__=='__main__':
    import time
    pcd = np.load('surface_pts.npy')
    semantic_color_map = np.asarray([
        [1.0,   0.0,   0.0],
        [0.514, 0.545, 0.682],
        [0.675, 0.882, 0.941],
        [0.427, 0.573, 0.871],
        [0.682, 0.698, 0.718],
        [0.4, 0.812, 0.949],
        [0.435, 0.471, 0.494],
        [0.635, 0.667, 0.992],
        [0.0,   1.0,   0.0],
        ])
    
    viz_handle = Open3DVisualizer()
    for pc in pcd:
        surface_pts = pc[:,:3]
        color_pts = semantic_color_map[pc[:,3].astype(np.int32)]
        viz_handle.update_buffers(surface_pts[:,:3], color_pts)
        time.sleep(0.05)
    viz_handle.vis.run()
