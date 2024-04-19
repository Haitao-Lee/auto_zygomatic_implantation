# coding = utf-8
import os
import numpy as np


# 不清楚为何直接用相对路径读不出来
cwd = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

data_name = 'patient_name'

# the directory that store implanting points
ip_dir = cwd + '/data/' + data_name + '/implant_point'

# the directory that store stl files
stl_dir = cwd + '/data/' + data_name + '/stl'

# the directory that store DICOM files
dicom_dir = cwd + '/data/' + data_name + '/dicom'

remove_radius = 2
nbh = 10

min_samples = 5

eps_dis=10

crop_distance = 42
crop_radius = 50

filled_iter = 7
voxel_size = 1

BIC_margin = 0.2

# stl outputpath
save_stl = cwd + '/export_screw_stl/' + data_name

# the directory that store npy files
# mtx_dir = cwd + '/data/data0014/trans_matrix'

# # the path of label file
# label_dir = cwd + '/data/label0014.nii.gz'

# # the path of image file
# img_dir = cwd + '/data/image0014.nii.gz'

# color space
color = [128/255,   174/255,   128/255,
         216/255,   101/255,   79/255,
         241/255,   214/255,   145/255,
         230/255,   220/255,   70/255,
         183/255,   156/255,   220/255,
         111/255,   184/255,   210/255]

# the number of neighborhoods in remove outliers
nd = 20

# the statistic ratio in remove outliers
std_rt = 2

# threhold in svm
svm_threshold = 2

# threshold in get_effect_pcd
gep_threshold = 6

# threhold in separate_point
sp_threshold = 10

# radius in separate_point
sp_radius = 60

# radius of screw
screw_radius = 1.5
screw_radius_list = [1.5, 1.75, 2, 2.25, 2.5, 3, 3.25]
devide_eps = 0

# radius of a single point
pcd_radius = 0.1

# length of screw
screw_length = 30

# the ratio of line to screw in length
line_length_rate = 20

# radius in KDTree searching while refining path_info
path_refine_radius = 20

# eps in relu_refine_dir
rrd_eps_max = 0.4
rrd_eps_min = 0.3

# max distance in ransac
ransac_eps = 1

# angle eps in estimate screw length
angle_eps = 5

# distance eps in estimate screw length
dist_eps = 4

# cone angle in get_cone
cone_angle = 5*np.pi/12

# radius resolution in get_cone
r_res = 10

# circle resolution in get_cone
c_res = 48  # 20°

# length threshold in refine_path_info
length_eps = 10*screw_radius

# point cloud down sample
voxel_size = 3

# resolution in explore
resolution = 8
