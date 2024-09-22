from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from copy import deepcopy
import math
import os
import numpy as np
import io
import xacro

PI = math.pi

ElementTree.register_namespace("xacro", "http://www.ros.org/wiki/xacro")

class NDofGenerator():
    def __init__(self, template_path, joint_gap=0.005, base_axis=2, base_offset=0.075):
        self.template_path = template_path
        self.joint_gap = joint_gap
        self.base_axis = base_axis
        self.base_offset = base_offset
        self.xacro_tree = ElementTree.parse(template_path)
        self.pattern_map = {'11': 'yzy',
                            '12': 'yxz',
                            '21': 'zxy',}
    
    def get_urdf(self, kinematics, dynamics):
        xacro_tree = deepcopy(self.xacro_tree)
        root = xacro_tree.getroot()
        config_child = Element('xacro:property', attrib={
            'name': 'joint_gap',
            'value': f'{self.joint_gap}'
        })
        root.append(config_child)
        config_child = Element('xacro:include', attrib={
            'filename': os.path.join(os.path.dirname(os.path.realpath(self.template_path)), 'common.xacro')
        })
        root.append(config_child)
        configs = []
        last_l_xyz = None
        last_axis = None
        for idx, (kin, dyn) in enumerate(zip(kinematics, dynamics)):
            m_type, axis, j_offset, ll, ul, r, lx, ly, lz, fj = kin
            el, vl, fr, dm, mx, my, mz = dyn
            axis = int(axis)
            origin_xyz = [0.0, 0.0, 0.0]
            if idx == 0:
                origin_xyz[self.base_axis] = self.base_offset
            else:
                origin_xyz[axis] = last_l_xyz[axis] - r +self.joint_gap
                configs[idx-1]['pattern'] = self.pattern_map[f'{last_axis}{axis}']
            origin_xyz = ' '.join([str(elm) for elm in origin_xyz])
            origin_rpy = [0.0, 0.0, 0.0]
            if m_type == 1.0 and fj: # override for y axis gripper
                origin_rpy[0] = -math.pi/2
                axis = 2
            origin_rpy[axis] = j_offset
            origin_rpy = ' '.join([str(elm) for elm in origin_rpy])
            axis_xyz = [0, 0, 0]
            axis_xyz[axis] = 1
            axis_xyz = ' '.join([str(elm) for elm in axis_xyz])

            configs.append(
                    dict(
                        m_num=idx+1,
                        type='member' if m_type==0.0 else 'gripper',
                        prefix=f'm{idx}3',
                        suffix=f'm{idx+1}1',
                        origin_xyz=origin_xyz,
                        origin_rpy=origin_rpy,
                        axis_xyz=axis_xyz,
                        limits=[ll, ul, el, vl],
                        dynamics=[fr, dm],
                        pattern='zxy', # default pattern
                        r=r,
                        l_xyz=[lx, ly, lz],
                        m_xyz=[mx, my, mz],
                        flip_joint=fj,
                        h=lx,
                        m_eof=[mx, my, mz],
                        s=lz
                        )
            )
            last_l_xyz = [lx, ly, lz]
            last_axis = axis

        config_child = Element('xacro:property', attrib={
            'name': 'config',
            'value': '${' + self.get_config_strs(configs) + '}'
        })
        root.append(config_child)
        config_child = Element('xacro:loop', attrib={
            'items': '${config}'
        })
        root.append(config_child)

        # process the annotated template
        xacro_io_handle = io.BytesIO()
        xacro_tree.write(xacro_io_handle)
        urdf_doc = xacro.parse(xacro_io_handle.getvalue().decode('utf-8'))
        xacro.process_doc(urdf_doc)
        return urdf_doc.toprettyxml(encoding='utf-8')
    
    def get_config_strs(self, config_list):
        str_config_list = []
        for config in config_list:
            config_str = f"dict( \
                            m_num={config['m_num']}, \
                            type='{config['type']}', \
                            prefix='{config['prefix']}', \
                            suffix='{config['suffix']}', \
                            origin_xyz='{config['origin_xyz']}', \
                            origin_rpy='{config['origin_rpy']}', \
                            axis_xyz='{config['axis_xyz']}', \
                            limits={str(config['limits'])}, \
                            dynamics={str(config['dynamics'])}, \
                            pattern='{config['pattern']}', \
                            r={str(config['r'])}, \
                            l_xyz={str(config['l_xyz'])}, \
                            m_xyz={str(config['m_xyz'])}, \
                            flip_joint={str(config['flip_joint'])}, \
                            h={str(config['h'])}, \
                            m_eof={str(config['m_eof'])}, \
                            s={str(config['s'])}, \
                            )"
            str_config_list.append(config_str)
        return '['+','.join(str_config_list)+']'


class EeGenerator():
    def __init__(self, template_path, joint_gap=0.005):
        self.template_path = template_path
        self.joint_gap = joint_gap
        self.xacro_tree = ElementTree.parse(template_path)
    
    def get_urdf(self, kinematics, dynamics):
        m_type, axis, j_offset, ll, ul, r, lx, ly, lz, fj = kinematics
        el, vl, fr, dm, mx, my, mz = dynamics
        assert m_type==1.0
        xacro_tree = deepcopy(self.xacro_tree)
        root = xacro_tree.getroot()
        config_child = Element('xacro:include', attrib={
            'filename': os.path.join(os.path.dirname(os.path.realpath(self.template_path)), 'common.xacro')
        })
        root.append(config_child)
        config_child = Element('xacro:just_gripper', attrib={
            'm_num':"1", 'mb':str(my), 'mf':str(mz), 's':str(lz)
        })
        root.append(config_child)

        # process the annotated template
        xacro_io_handle = io.BytesIO()
        xacro_tree.write(xacro_io_handle)
        urdf_doc = xacro.parse(xacro_io_handle.getvalue().decode('utf-8'))
        xacro.process_doc(urdf_doc)
        return urdf_doc.toprettyxml(encoding='utf-8')
    

class UrdfSampler():
    def __init__(self, n_dof_template_path):
        """
        Currently samples 
        7-DOF zyzyzyz style: sawyer, franka
        6-DOF zyyyzy style: ur

        mean and sigma are determined emperically to be close to real robots
        for now franka is ignored from 7-DOF sampling and only sawyer is considered
        """

        sample_constants = self._get_robot_constants()
        self.ndof_generator = NDofGenerator(n_dof_template_path, 
                                          sample_constants['joint_gap'], 
                                          sample_constants['base_axis'], 
                                          sample_constants['base_offset'])

    def _get_robot_constants(self):
        return {'joint_gap': 0.005, 'base_axis': 2, 'base_offset': 0.03}
    
    def _get_joint_limits_sample(self, jr, DOF, constraint='random'):
        if constraint=='random':
            l_limits = np.asarray([np.random.normal(-jr, PI/18) for _ in range(DOF)], dtype=np.float32)
            u_limits = np.asarray([np.random.normal(jr, PI/18) for _ in range(DOF)], dtype=np.float32)
        else:
            if constraint=='sawyer':
                l_limits = np.asarray([-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124], dtype=np.float32)
                u_limits = np.asarray([3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124], dtype=np.float32)
            if constraint=='ur5':
                l_limits = np.asarray([-6.2831, -6.2831, -3.1415, -6.2831, -6.2831, -6.2831], dtype=np.float32)
                u_limits = np.asarray([6.2831, 6.2831, 3.1415, 6.2831, 6.2831, 6.2831], dtype=np.float32)
            if constraint=='franka':
                l_limits = np.asarray([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
                u_limits = np.asarray([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)
            l_limits += np.random.normal(np.zeros(l_limits.shape), np.ones(l_limits.shape)*PI/18)
            u_limits += np.random.normal(np.zeros(u_limits.shape), np.ones(u_limits.shape)*PI/18)
        return l_limits, u_limits

    def _get_radius_sample(self, DOF, constraint='random'):
        # all same (franka) or tappering radius (sewyer, ur)
        if constraint=='random':
            radius_sample = np.ones((DOF,))*max(0.035, min(0.045+np.random.normal(0.0, 0.01), 0.055)) \
                if np.random.rand() > 0.5 \
            else np.asarray([0.9**i for i in range(DOF)], dtype=np.float32)* \
                max(0.05, min(0.06+np.random.normal(0.0, 0.01), 0.07))
        else:
            if constraint=='sawyer':
                radius_sample = np.asarray([0.06, 0.06, 0.0508, 0.0508, 0.0412, 0.0412, 0.0412], np.float32)
            if constraint=='ur5':
                radius_sample = np.asarray([0.0571, 0.0571, 0.0508, 0.0381, 0.0381, 0.0381], np.float32)
            if constraint=='franka':
                radius_sample = np.asarray([0.0571,]*7, np.float32)
            
            radius_sample *= 0.8 + abs(np.random.normal(0.0, 0.2))
            
        # print(radius_sample)
        return radius_sample
    
    def _get_length_sample(self, DOF, constraint='random'):
        if constraint=='random':
            mu = np.asarray([0.0, 0.05, 0.15], np.float32)
            sigma = np.asarray([0.025, 0.05, 0.05], np.float32)
            length_sample = abs(np.vstack([np.random.normal(mu, sigma) for _ in range(DOF-1)]))
            for i in range(DOF-1):
                # remove too short x-links with exception for 6-DOF
                if length_sample[i][0] < 0.05 and not ((i==1 or i==2) and DOF==6):
                    length_sample[i][0] = 0.0
                # lower limits on y-link and z-link
                length_sample[i][1] = max(length_sample[i][1], 0.025) 
                length_sample[i][2] = max(length_sample[i][2], 0.075) 
                # 6-DOF exception z-link and x-link
                if (i==1 or i==2) and DOF==6:
                    length_sample[i][2] += abs(np.random.normal(0.15, 0.05))
                    length_sample[i][2] = max(length_sample[i][2], 0.1) 
                    # x length is used for y
                    length_sample[i][0] = max(np.random.normal(0.05, 0.05), 0.025) 
        else:
            if constraint=='sawyer':
                mu = np.asarray([[0.0635, 0.0635, 0.2286],
                                 [0.0, 0.127, 0.127],
                                 [0.0, 0.0635, 0.2921],
                                 [0.0, 0.127, 0.127],
                                 [0.0, 0.0381, 0.2921],
                                 [0.0, 0.1016, 0.0762],
                                 ], np.float32)
            if constraint=='ur5':
                mu = np.asarray([[0.0, 0.06985, 0.0635],
                                 [0.0635, 0.0635, 0.4313],
                                 [0.0508, 0.0508, 0.3556],
                                 [0.0, 0.0508, 0.0508],
                                 [0.0, 0.0508, 0.0508],
                                 ], np.float32)
            if constraint=='franka':
                mu = np.asarray([[0.0, 0.0, 0.2032],
                                 [0.0, 0.0, 0.2032],
                                 [0.08225, 0.0, 0.127],
                                 [0.08225, 0.0, 0.1524],
                                 [0.0, 0.0, 0.254],
                                 [0.0889, 0.0, 0.114],
                                 ], np.float32)
            length_sample = []
            for m in mu:
                sample = abs(np.random.normal(m, m/10))
                for i in range(3):
                    if m[i] < 1e-6:
                        sample[i] = 0.0
                length_sample.append(sample)
            length_sample = np.vstack(length_sample)

        # print(length_sample)
        return length_sample

    def _get_gripper_sample(self, DOF, radius_sample, constraint='random'):
        # 0.2 is y length for franka gripper and we dont want the gripper to be smaller than base cylinder diameter
        scale = max(min(np.random.normal(0.75, 0.25), 1.0), 2*radius_sample[-1]/0.2) # limit gripper scaling
        if constraint=='random':
            base_cylinder_height = radius_sample[-2]*(1.0+abs(np.random.normal(0.0,0.2)))
            if DOF==6:
                # required to prevent self collision with gripper
                base_cylinder_height *= 2
        else:
            if constraint=='sawyer':
                base_cylinder_height = 0.0762
            if constraint=='ur5':
                base_cylinder_height = 0.0317
            if constraint=='franka':
                base_cylinder_height = 0.01
        base_cylinder_height *= 1.0+abs(np.random.normal(0.0,0.2))

        # print([scale, base_cylinder_height])
        return scale, base_cylinder_height
    
    def sample_robot(self, constraint='random', seed=None):
        """
        These parameters have been considered relevant for the motion planning (sequence serial)
        member type: 0(member)/ 1(gripper)
        axis: 0(x)/ 1(y)/ 2(z)
        joint offset: of
        joint angle limit: ll, ul
        member radius: r
        member structure: lx, ly, lz (member)/ h, dont_care, s (gripper)
        flip joint: 0(false)/ 1(true) 

        These parameters have been considered relevant for handling dynamics (sequence serial)
        These parameters might be used for domain randomization in RL tasks
        effort limit: el
        velocity limit: vl
        joint friction: fr
        joint damping: dm
        member mass: mx, my, mz (member)/ mc, mb, mf (gripper)
        """

        # sample DOF      
        if constraint=='random':
            DOF = 6 if np.random.rand() > 0.5 else 7
        if constraint=='sawyer':
            DOF = 7
        if constraint=='ur5':
            DOF = 6
        if constraint=='franka':
            raise NotImplementedError()
        print(constraint)
        
        # values defined here initally are junk, they are sampled appropriately ahead
        if DOF==7:
            jr = PI
            kinematics = np.asarray(
                    [(0.0,2.0,0.0,-jr,jr,0.1,0.0,0.2,0.3,0.0),
                    (0.0,1.0,0.0,-jr,jr,0.1,0.0,0.01,0.3,0.0),
                    (0.0,2.0,0.0,-jr,jr,0.1,0.2,0.2,0.2,1.0),
                    (0.0,1.0,0.0,-jr,jr,0.1,0.2,0.01,0.2,0.0),
                    (0.0,2.0,0.0,-jr,jr,0.1,0.0,0.2,0.5,1.0),
                    (0.0,1.0,0.0,-jr,jr,0.1,0.1,0.01,0.2,1.0),
                    (1.0,2.0,0.0,-jr,jr,0.1,0.1,0.0,1.75,0.0),], dtype=np.float32)
            dynamics = np.asarray(
                    [(20.0,2.5,0.13,0.0,0.0,0.02,0.03),
                    (20.0,2.5,0.13,0.0,0.0,0.001,0.03),
                    (20.0,2.5,0.13,0.0,0.02,0.02,0.02),
                    (20.0,2.5,0.13,0.0,0.02,0.001,0.02),
                    (20.0,2.5,0.13,0.0,0.0,0.02,0.05),
                    (20.0,2.5,0.13,0.0,0.01,0.001,0.02),
                    (20.0,2.5,0.13,0.0,0.01,0.01,0.001),], dtype=np.float32)
        elif DOF==6:
            jr = 2*PI
            kinematics = np.asarray(
                        [(0.0,2.0,0.0,-jr,jr,0.1,0.0,0.2,0.2,0.0),
                        (0.0,1.0,0.0,-jr,jr,0.1,0.01,0.2,0.6,1.0),
                        (0.0,1.0,PI,-jr,jr,0.1,0.01,0.2,0.6,1.0),
                        (0.0,1.0,PI,-jr,jr,0.1,0.0,0.01,0.2,0.0),
                        (0.0,2.0,0.0,-jr,jr,0.1,0.0,0.25,0.1,0.0),
                        (1.0,1.0,0.0,-jr,jr,0.1,0.1,0.0,1.75,1.0),], dtype=np.float32)
            dynamics = np.asarray(
                        [(20.0,2.5,0.13,0.0,0.0,0.02,0.02),
                        (20.0,2.5,0.13,0.0,0.001,0.02,0.06),
                        (20.0,2.5,0.13,0.0,0.001,0.02,0.06),
                        (20.0,2.5,0.13,0.0,0.0,0.001,0.02),
                        (20.0,2.5,0.13,0.0,0.0,0.025,0.01),
                        (20.0,2.5,0.13,0.0,0.01,0.01,0.001),], dtype=np.float32)

        # sample joint limits
        l_limits, u_limits = self._get_joint_limits_sample(jr, DOF, constraint=constraint)

        # sample radius
        radius_sample = self._get_radius_sample(DOF, constraint=constraint)

        # sample lengths (lx, ly, lz)
        length_sample = self._get_length_sample(DOF, constraint=constraint)
        
        # sample flip joints
        # flip_joint = (np.random.rand(4,) > 0.8).astype(np.float32) 
        # random flipping maybe too complex so hardcoded for initial experiments
        if DOF==7:
            flip_joint = np.array([0.0,0.0,0.0,0.0], dtype=np.float32) # sawyer
        elif DOF==6:
            flip_joint = np.array([1.0,1.0,0.0,0.0], dtype=np.float32) # ur
            
        # sample gripper
        scale, base_cylinder_height = self._get_gripper_sample(DOF, radius_sample, constraint=constraint)
        
        # set radius
        kinematics[:,5] = radius_sample

        # set lengths
        kinematics[:DOF-1,6:9] = length_sample

        # set flip joints
        kinematics[DOF-5:DOF-1,9] = flip_joint
        if DOF == 6:
            # special requirement for 6-DOF
            kinematics[DOF-1,9] = 1.0

        # set joint limits
        kinematics[:,3] = l_limits
        kinematics[:,4] = u_limits

        # set gripper dims
        kinematics[DOF-1,6:9] = abs(np.asarray([base_cylinder_height, 0.0, scale], dtype=np.float32))

        # set masses
        mass_density = 0.1
        dynamics[:DOF-1,4:] = length_sample*mass_density
        # assuming eef masses to be constant
            
        return kinematics, dynamics, self.ndof_generator.get_urdf(kinematics.tolist(), dynamics.tolist())