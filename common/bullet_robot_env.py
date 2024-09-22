import pybullet as p
import pybullet_data
import numpy as np
from threading import Thread
import time


class BulletRobotEnv():
    # Borrowed from https://github.com/fishbotics/robofin/blob/main/robofin/bullet.py
    def __init__(self, gui=False, base_link='link_m03', floor=False, rotate_camera=False):
        self.gui = gui
        if self.gui:
            self.cid = p.connect(p.GUI)
            if floor:
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.loadURDF("plane.urdf", basePosition=[0,0,-0.1])
            visual_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.03,0.1,0.045],
            rgbaColor=[1.,0.,0.,1.],
            physicsClientId=self.cid,
            )
            self.dummy_eef_id = p.createMultiBody(
            basePosition=[0,0,0],
            baseOrientation=[0,0,0,1],
            baseVisualShapeIndex = visual_id,
            baseCollisionShapeIndex = -1,
            physicsClientId=self.cid,
            )
        else:
            self.cid = p.connect(p.DIRECT)
        self.robot_id = None
        self.obstacle_ids = None
        self.base_link = base_link
        self.viz_theta = 0

        if rotate_camera:
            th = Thread(target=self.rotate_camera, args=(1.5, 1.5, 0.01))
            th.start()
    
    def load_robot(self, urdf_path): 
        self.robot_id = p.loadURDF(urdf_path,
                    useFixedBase=True,
                    physicsClientId=self.cid,
                    flags=p.URDF_USE_SELF_COLLISION|p.URDF_MERGE_FIXED_LINKS)
        self.DOF = p.getNumJoints(self.robot_id, physicsClientId=self.cid)
        
        self.link_name_to_id = {
            p.getBodyInfo(self.robot_id, physicsClientId=self.cid)[0].decode("UTF-8"): -1
        }
        for _id in range(self.DOF):
            _name = p.getJointInfo(self.robot_id, _id, physicsClientId=self.cid)[12].decode(
                "UTF-8"
            )
            self.link_name_to_id[_name] = _id
        
        self.id_to_link_name = {}
        for k, v in self.link_name_to_id.items():
            self.id_to_link_name[v] = k
    
    def load_obstacles(self, obstacle_config):
        self.obstacle_ids = []
        for obs in obstacle_config:
            if obs['type'] == 'cuboid':
                if self.gui:
                    visual_id = p.createVisualShape(
                        shapeType=p.GEOM_BOX,
                        halfExtents=(obs['scale']/2).tolist(),
                        rgbaColor=np.append(obs['color'], 1.0).tolist(),
                        physicsClientId=self.cid,
                    )
                else:
                    visual_id = -1
                collision_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=(obs['scale']/2).tolist(),
                physicsClientId=self.cid,
                )
            elif obs['type'] == 'cylinder':
                if self.gui:
                    visual_id = p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=obs['radius'],
                    length=obs['height'],
                    rgbaColor=np.append(obs['color'], 1.0).tolist(),
                    physicsClientId=self.cid,
                    )
                else:
                    visual_id = -1
                collision_id = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=obs['radius'],
                height=obs['height'],
                physicsClientId=self.cid,
                )
            else:
                raise Exception('Only Cuboids and Cylinders are Allowed for Obstacles')
            self.obstacle_ids.append(p.createMultiBody(
                    basePosition=obs['translation'].tolist(),
                    baseOrientation=obs['orientation'][[1,2,3,0]].tolist(),
                    baseVisualShapeIndex = visual_id,
                    baseCollisionShapeIndex = collision_id,
                    physicsClientId=self.cid,
                ))
    
    def rotate_camera(self, height=1, radius=1, dtheta=0.1):
        while True:
            phi = (np.pi/2)*np.sin(self.viz_theta)
            self.viz_theta = (self.viz_theta+dtheta)%(2*np.pi)
            pos = np.array([-radius*np.cos(phi), radius*np.sin(phi), height])
            disp = np.zeros(3) - pos
            dist = np.linalg.norm(disp)
            yaw = np.arctan2(-disp[0],disp[1]) * 180/np.pi
            pitch = np.arctan2(disp[2],np.sqrt(disp[0]**2+disp[1]**2)) * 180/np.pi
            p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=[0,0,0])
            time.sleep(0.1)

    def marionette_robot(self, joint_state):
        assert joint_state.shape[0] == self.DOF
        for i in range(self.DOF):
            p.resetJointState(self.robot_id, i, joint_state[i])
    
    def perform_collision_check(self):
        # must be called before self_collision_check or env_collision_check
        p.performCollisionDetection(physicsClientId=self.cid)
    
    def self_collision_check(self):
        # should be called after perform_collision_check
        assert self.robot_id is not None
        contact_points = p.getContactPoints(self.robot_id, self.robot_id, physicsClientId=self.cid)
        for c in contact_points:
            # A link is always in collision with itself and its neighbors
            # TODO: known bug for 6-DOF UR style arms where collision between links
            # 1 and 2 or 2 and 3 is not detected
            if abs(c[3] - c[4]) <= 1:
                continue
            return True
        return False
    
    def env_collision_check(self):
        # should be called after perform_collision_check
        assert self.robot_id is not None
        assert self.obstacle_ids is not None
        for obs_id in self.obstacle_ids:
            contact_points = p.getContactPoints(self.robot_id, obs_id, physicsClientId=self.cid)            
            for c in contact_points:
                # Base is always in collision
                # TODO: optimization possible by using collision mask
                if c[3]==self.link_name_to_id[self.base_link]:
                    continue
                return True
        return False
    
    def complete_collision_check(self):
        # should be called after perform_collision_check
        assert self.robot_id is not None
        contact_points = p.getContactPoints(self.robot_id, self.robot_id, physicsClientId=self.cid)
        self_collision_links = set()
        for c in contact_points:
            # A link is always in collision with itself and its neighbors
            # TODO: known bug for 6-DOF UR style arms where collision between links
            # 1 and 2 or 2 and 3 is not detected
            if abs(c[3] - c[4]) <= 1:
                continue
            self_collision_links.add(c[3])
            self_collision_links.add(c[4])

        assert self.obstacle_ids is not None
        env_collision_links = set()
        for obs_id in self.obstacle_ids:
            contact_points = p.getContactPoints(self.robot_id, obs_id, physicsClientId=self.cid)            
            for c in contact_points:
                # Base is always in collision
                # TODO: optimization possible by using collision mask
                if c[3]==self.link_name_to_id[self.base_link]:
                    continue
                env_collision_links.add(c[3])

        return self_collision_links.union(env_collision_links)
    
    def self_collision_distance(self, collision_radius=0.1):
        assert self.robot_id is not None
        closest_points = p.getClosestPoints(self.robot_id, self.robot_id, collision_radius, physicsClientId=self.cid)
        distance = []
        for c in closest_points:
            # A link is always in collision with itself and its neighbors
            # TODO: optimization possible by using collision mask
            if abs(c[3] - c[4]) <= 1:
                continue
            distance.append(c[8])
        return distance
    
    def env_collision_distance(self, collision_radius=0.5):
        assert self.robot_id is not None
        assert self.obstacle_ids is not None
        distance = []
        for obs_id in self.obstacle_ids:
            closest_points = p.getClosestPoints(self.robot_id, obs_id, collision_radius, physicsClientId=self.cid)
            for c in closest_points:
                # Base is always in collision
                # TODO: optimization possible by using collision mask
                if c[3]==self.link_name_to_id[self.base_link]:
                    continue
                distance.append(c[8])
        return distance
    
    def set_dummy_state(self, eef_pos, eef_ori):
        p.resetBasePositionAndOrientation(self.dummy_eef_id,
                                          posObj=eef_pos.tolist(),
                                          ornObj=eef_ori[[1,2,3,0]].tolist(),
                                          physicsClientId=self.cid)

    def robot_ik(self, eef_pos, eef_ori, eef_link_name=None):
        eef_link_name = f'link_m{self.DOF}1' if eef_link_name is None else eef_link_name
        if self.gui:
            self.set_dummy_state(eef_pos, eef_ori)
        try:
            # sometimes ik fails with an unknown error
            return np.asarray(p.calculateInverseKinematics(bodyUniqueId=self.robot_id,
                                        endEffectorLinkIndex=self.link_name_to_id[eef_link_name],
                                        targetPosition=eef_pos.tolist(),
                                        targetOrientation=eef_ori[[1,2,3,0]].tolist(),
                                        maxNumIterations=1000,
                                        physicsClientId=self.cid), dtype=np.float32)
        except:
            print('IK failed. Returning zero config.')
            return np.asarray([0,]*self.DOF)

    def remove_robot(self):
        if self.robot_id is not None:
            p.removeBody(self.robot_id, physicsClientId=self.cid)
        self.robot_id = None

    def remove_obstacles(self):
        if self.obstacle_ids is not None:
            for id in self.obstacle_ids:
                p.removeBody(id, physicsClientId=self.cid)
        self.obstacle_ids = None

    def clear_scene(self):
        self.remove_robot()
        self.remove_obstacles()
