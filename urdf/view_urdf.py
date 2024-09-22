from urdfpy import URDF

urdf_handle = URDF.load('urdf/kinova/kinova_7dof_sample.urdf')
urdf_handle.show()