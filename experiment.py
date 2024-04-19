# coding = utf-8
import numpy as np
import data_input
import visualization
import screw_setting
import core_algorithm
import vtkmodules.all as vtk
import os
import open3d as o3d
import time
import data_process
import geometry
import paths
import matplotlib.pyplot as plt



def zygomatic_implant_planning(stl_folder, point_folder, implant_point_dir=None):
    stl_filenames = data_input.get_filenames(stl_folder, ".stl")
    point_filenames = data_input.get_filenames(point_folder, ".txt")
    stls = data_input.getSTLs(stl_filenames)
    implant_points = data_input.getImplantPoints(point_filenames)[0]
    visualization.stl_visualization_by_vtk([stls[-1]], implant_points)
    plane = core_algorithm.build_symmetrical_plane(stls[-1])
    actor = visualization.get_screw_cylinder_actor(plane[0], plane[1], 200,1,4)
    actor.GetProperty().SetOpacity(0.4)
    actor.GetProperty().SetColor(1,0,0)
    
    plane_points = core_algorithm.project_points_onto_plane(implant_points, plane[0], plane[1])
    ref_point = np.sum(plane_points, axis=0)/plane_points.shape[0]
    new_obb_poly, new_obb_actor = core_algorithm.transform_obb_box(ref_point, plane[1])
    # visualization.stl_visualization_by_vtk([stls[-1], new_obb_poly])
    clip_direc = core_algorithm.get_clip_direct(implant_points, plane[0], ref_point)
    crop_res = core_algorithm.crop_by_cylinder(stls[-1], new_obb_poly, ref_point)
    tmp_actor = visualization.get_screw_cylinder_actor(ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point), clip_direc, 60,1,12)
    tmp_actor.GetProperty().SetOpacity(0.6)
    tmp_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
    visualization.stl_visualization_by_vtk([crop_res], [ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point)],[tmp_actor])
    target_mesh = core_algorithm.connectivity_filter(core_algorithm.crop_by_cylinder(core_algorithm.seperate_maxilla_mandible(stls[-1], clip_direc, ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point)), new_obb_poly, ref_point), implant_points[0])
    zygomatic_bone1, zygomatic_bone2, zygomatic_pcd1, zygomatic_pcd2 = core_algorithm.get_zygomatic_bone(target_mesh, plane[1], ref_point)
    visualization.stl_pcd_visualization_by_vtk([target_mesh, zygomatic_bone1, zygomatic_bone2], [zygomatic_pcd1, zygomatic_pcd2])
    visualization.stl_visualization_by_vtk([core_algorithm.seperate_maxilla_mandible(crop_res, clip_direc, ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point)), target_mesh])

    visual_stls = []
    for stl in stls:
        visual_stls.append(stl)
    visual_stls.append(zygomatic_bone1)
    visual_stls.append(zygomatic_bone2)
    visul_points = np.concatenate([implant_points, plane_points])
    sphere_actors = [actor]
    for i in range(visul_points.shape[0]):
        p = visul_points[i]
        if i < 4:
            sphere_actors.append(visualization.get_sphere_actor(p, radius=1, color=[1,   0,   0]))
        else:
            sphere_actors.append(visualization.get_sphere_actor(p, radius=1, color=[0,   1,   0]))
    visualization.stl_visualization_by_vtk([stls[-1]], actors=sphere_actors, opacity=0.1)
    visualization.stl_visualization_by_vtk([core_algorithm.seperate_maxilla_mandible(stls[-1], plane[0] - ref_point, ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point))], np.concatenate([visul_points, np.array([plane[0]])], axis=0), [actor, new_obb_actor], opacity=0.8)
    
    planning_pcds = [data_input.getPCDfromSTL([target_mesh])[0], zygomatic_pcd1, zygomatic_pcd2]
    visualization.stl_pcd_visualization_by_vtk(stls, planning_pcds)
    
    planning_info = core_algorithm.regularize_info(implant_points, planning_pcds, ref_point, plane[1]) #[implant_point, planning pointcloud, target pointcloud]
    optimal_paths = core_algorithm.genrate_optimal_paths(planning_info)
    
    path_actors = []
    for path in optimal_paths:
        actor = visualization.get_screw_cylinder_actor(path[0]+path[1]/2, path[1]/np.linalg.norm(path[1]),path[2], np.linalg.norm(path[1]),12)
        actor.GetProperty().SetColor(1, 0, 0)
        actor.GetProperty().SetOpacity(1)
        path_actors.append(actor)
    visualization.stl_visualization_by_vtk([stls[-1]], None, path_actors, opacity=0.8)
    return optimal_paths

def crop_max_man(stl_dir, ip_dir, cr):
    stl_filenames = data_input.get_filenames(stl_dir, ".stl")
    point_filenames = data_input.get_filenames(ip_dir, ".txt")
    stls = data_input.getSTLs(stl_filenames)
    implant_points = data_input.getImplantPoints(point_filenames)[0]
    plane = core_algorithm.build_symmetrical_plane(stls[-1], implant_points)
    actor = visualization.get_screw_cylinder_actor(plane[0], plane[1], 200,1,4)
    actor.GetProperty().SetOpacity(0.4)
    actor.GetProperty().SetColor(1,0,0)
    
    plane_points = core_algorithm.project_points_onto_plane(implant_points, plane[0], plane[1])
    ref_point = np.sum(plane_points, axis=0)/plane_points.shape[0]
    new_obb_poly, _ = core_algorithm.transform_obb_box(ref_point, plane[1], radius=cr)
    target_mesh = core_algorithm.crop_by_cylinder(stls[-1], new_obb_poly, ref_point)
    visualization.stl_visualization_by_vtk([stls[-1], target_mesh])



def iden_zygo_bone(stl_dir, ip_dir, cr, cd):
    stl_filenames = data_input.get_filenames(stl_dir, ".stl")
    point_filenames = data_input.get_filenames(ip_dir, ".txt")
    stls = data_input.getSTLs(stl_filenames)
    implant_points = data_input.getImplantPoints(point_filenames)[0]
    plane = core_algorithm.build_symmetrical_plane(stls[-1], implant_points)
    actor = visualization.get_screw_cylinder_actor(plane[0], plane[1], 200,1,4)
    actor.GetProperty().SetOpacity(0.4)
    actor.GetProperty().SetColor(1,0,0)
    
    plane_points = core_algorithm.project_points_onto_plane(implant_points, plane[0], plane[1])
    ref_point = np.sum(plane_points, axis=0)/plane_points.shape[0]
    new_obb_poly, _ = core_algorithm.transform_obb_box(ref_point, plane[1], radius=cr)
    clip_direc = core_algorithm.get_clip_direct(implant_points, plane[0], ref_point)
    target_mesh = core_algorithm.connectivity_filter(core_algorithm.crop_by_cylinder(core_algorithm.seperate_maxilla_mandible(stls[-1], clip_direc, ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point)), new_obb_poly, ref_point), implant_points[0])
    zygomatic_bone1, zygomatic_bone2, zygomatic_pcd1, zygomatic_pcd2 = core_algorithm.get_zygomatic_bone(target_mesh, plane[1], ref_point, distance=cd)
    # visualization.stl_visualization_by_vtk([stls[-1], zygomatic_bone1, zygomatic_bone2])
    return zygomatic_bone1, zygomatic_bone2, zygomatic_pcd1, zygomatic_pcd2

def fill_zygomatic_pcd(stl_dir, ip_dir, cr, cd, iters):
    stl_filenames = data_input.get_filenames(stl_dir, ".stl")
    point_filenames = data_input.get_filenames(ip_dir, ".txt")
    stls = data_input.getSTLs(stl_filenames)
    implant_points = data_input.getImplantPoints(point_filenames)[0]
    plane = core_algorithm.build_symmetrical_plane(stls[-1], implant_points)
    actor = visualization.get_screw_cylinder_actor(plane[0], plane[1], 200,1,4)
    actor.GetProperty().SetOpacity(0.4)
    actor.GetProperty().SetColor(1,0,0)
    plane_points = core_algorithm.project_points_onto_plane(implant_points, plane[0], plane[1])
    ref_point = np.sum(plane_points, axis=0)/plane_points.shape[0]
    new_obb_poly, _ = core_algorithm.transform_obb_box(ref_point, plane[1], radius=cr)
    clip_direc = core_algorithm.get_clip_direct(implant_points, plane[0], ref_point)
    target_mesh = core_algorithm.connectivity_filter(core_algorithm.crop_by_cylinder(core_algorithm.seperate_maxilla_mandible(stls[-1], clip_direc, ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point)), new_obb_poly, ref_point), implant_points[0])
    center1 = ref_point + plane[1]*cd
    center2 = ref_point - plane[1]*cd
    plane1 = vtk.vtkPlane()
    plane1.SetOrigin(center1)
    plane1.SetNormal(plane[1])  # 以 x 轴为法线的平面
    clipper1 = vtk.vtkClipPolyData()
    clipper1.SetInputData(target_mesh)
    clipper1.SetClipFunction(plane1)
    clipper1.GenerateClippedOutputOn()  # 生成被裁剪部分的输出
    clipper1.Update()
    plane2 = vtk.vtkPlane()
    plane2.SetOrigin(center2)
    plane2.SetNormal(-plane[1])  # 以 x 轴为法线的平面
    clipper2 = vtk.vtkClipPolyData()
    clipper2.SetInputData(target_mesh)
    clipper2.SetClipFunction(plane2)
    clipper2.GenerateClippedOutputOn()  # 生成被裁剪部分的输出
    clipper2.Update()
    points_npy = np.array(target_mesh.GetPoints().GetData())[::7]
    dif_npy = points_npy - ref_point
    res = np.dot(dif_npy, plane[1])
    indices1 = np.array(np.argwhere(res > cd)).flatten()
    points1_npy = geometry.find_largest_cluster(np.array(points_npy[indices1]), min_samples=screw_setting.min_samples, eps=screw_setting.eps_dis)
    indices2 = np.array(np.argwhere(res < -cd)).flatten()
    points2_npy = geometry.find_largest_cluster(np.array(points_npy[indices2]), min_samples=screw_setting.min_samples, eps=screw_setting.eps_dis)
    PCD1 = o3d.geometry.PointCloud()
    PCD1.points = o3d.utility.Vector3dVector(points1_npy)
    visualization.stl_pcd_visualization_by_vtk([clipper1.GetOutput()], [PCD1])

    for itera in iters:
        # 将NumPy数组转换为Open3D点云
        PCD1 = o3d.geometry.PointCloud()
        filled_ps1 = core_algorithm.iterative_fill_pointcloud(np.array(points1_npy), plane[1], ref_point, iter=itera)
        PCD1.points = o3d.utility.Vector3dVector(filled_ps1)
        # 将NumPy数组转换为Open3D点云
        PCD2 = o3d.geometry.PointCloud()
        start_time = time.time()
        filled_ps1 = core_algorithm.iterative_fill_pointcloud(np.array(points1_npy), plane[1], ref_point, iter=itera)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        print("nf:", itera, " time:", execution_time*1000,'ms')
        filled_ps1 = core_algorithm.iterative_fill_pointcloud(np.array(points1_npy), plane[1], ref_point, iter=itera, rate=30)
        PCD1.points = o3d.utility.Vector3dVector(filled_ps1)
        # visualization.stl_pcd_visualization_by_vtk([clipper1.GetOutput()], [PCD1])


def zygomatic_implant_planning(stl_folder, point_folder, save_folder):
    stl_filenames = data_input.get_filenames(stl_folder, ".stl")
    point_filenames = data_input.get_filenames(point_folder, ".txt")
    stls = data_input.getSTLs(stl_filenames)
    implant_points = data_input.getImplantPoints(point_filenames)[0]
    # visualization.stl_visualization_by_vtk(stls, implant_points, opacity=0.8)
    plane = core_algorithm.build_symmetrical_plane(stls[-1], implant_points)
    actor = visualization.get_screw_cylinder_actor(plane[0], plane[1], 200,1,4)
    actor.GetProperty().SetOpacity(0.4)
    actor.GetProperty().SetColor(1,0,0)
    plane_points = core_algorithm.project_points_onto_plane(implant_points, plane[0], plane[1])
    ref_point = np.sum(plane_points, axis=0)/plane_points.shape[0]
    new_obb_poly, _ = core_algorithm.transform_obb_box(ref_point, plane[1])
    clip_direc = core_algorithm.get_clip_direct(implant_points, plane[0], ref_point)
    target_mesh = core_algorithm.connectivity_filter(core_algorithm.crop_by_cylinder(core_algorithm.seperate_maxilla_mandible(stls[-1], clip_direc, ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point)), new_obb_poly, ref_point), implant_points[0])
    _, _, zygomatic_pcd1, zygomatic_pcd2 = core_algorithm.get_zygomatic_bone(target_mesh, plane[1], ref_point)
    planning_pcds = [data_input.getPCDfromSTL([target_mesh])[0], zygomatic_pcd1, zygomatic_pcd2]
    planning_info = core_algorithm.regularize_info(implant_points, planning_pcds, ref_point, plane[1]) #[implant_point, planning pointcloud, target pointcloud]
    optimal_paths = core_algorithm.genrate_optimal_paths(planning_info)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for i in range(len(optimal_paths)):
        path = optimal_paths[i]
        pasth_stl = visualization.get_screw_cylinder_polydata(path[0]+path[1]/2, path[1]/np.linalg.norm(path[1]),path[2], np.linalg.norm(path[1]),12)
        data_process.save_polydata_to_stl(pasth_stl, save_folder+ f"/autoImplant{i+1}.stl")
    data_process.save_polydata_to_stl(stls[-1], save_folder+ f"/bone.stl")
    # for i in range(len(stls)-1):
    #     data_process.save_polydata_to_stl(pasth_stl, save_folder+ f"/manuImplant{i+1}.stl")
    # 指定保存的文件路径
    file_path = save_folder+"/autoImplant.txt"
    # 打开文件进行写入
    with open(file_path, "w") as f:
        # 将列表中的每个元素写入文件，每个元素占据一行
        for path in optimal_paths:
            for i in range(len(path)):
                if i < 2:
                    f.write(f"{path[i][0]},{path[i][1]},{path[i][2]},      ")
                if i == 2:
                    f.write(" %s\n\n" % str(path[i]))
    return optimal_paths


def get_planning_pcd(stl_folder, point_folder):
    stl_filenames = data_input.get_filenames(stl_folder, ".stl")
    point_filenames = data_input.get_filenames(point_folder, ".txt")
    stls = data_input.getSTLs(stl_filenames)
    implant_points = data_input.getImplantPoints(point_filenames)[0]
    # visualization.stl_visualization_by_vtk(stls, implant_points, opacity=0.8)
    plane = core_algorithm.build_symmetrical_plane(stls[-1], implant_points)
    plane_points = core_algorithm.project_points_onto_plane(implant_points, plane[0], plane[1])
    ref_point = np.sum(plane_points, axis=0)/plane_points.shape[0]
    new_obb_poly, _ = core_algorithm.transform_obb_box(ref_point, plane[1])
    clip_direc = core_algorithm.get_clip_direct(implant_points, plane[0], ref_point)
    target_mesh = core_algorithm.connectivity_filter(core_algorithm.crop_by_cylinder(core_algorithm.seperate_maxilla_mandible(stls[-1], clip_direc, ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point)), new_obb_poly, ref_point), implant_points[0])
    _, _, zygomatic_pcd1, zygomatic_pcd2 = core_algorithm.get_zygomatic_bone(target_mesh, plane[1], ref_point)
    return data_input.getPCDfromSTL([target_mesh])[0] + zygomatic_pcd1 + zygomatic_pcd2
    
    

def cal_BIC(path, pcd, radius=2, BIC_margin = screw_setting.BIC_margin):
    start_p = path[0]
    direct = path[1]
    length = np.linalg.norm(direct)
    direct = direct/length    
    planning_ps = pcd.points  # [n,3]
    diff = planning_ps - start_p  # [n, 3]
    diff_norm = np.linalg.norm(diff, axis=1) # [n, 1]
    prj_l = np.dot(diff, direct.T) # [n, 1]
    dist_2 = np.square(diff_norm) - np.square(prj_l) # [n, 1]
    indices = np.where((dist_2 < (radius + BIC_margin)**2) & (dist_2 > (radius - BIC_margin)**2) & (prj_l>= 0)  & (prj_l <= length)) # [k, 1]
    return indices[0].shape[0]
    
    


# # experiments of crop radius and clip distance
# cwd = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')   
# data_names = ['yangyayi','xufeng' , 'songjun' , 'majun' , 'luweilan' , 'luoqiu' , 'liuyanhua' , 'linqianhong' , 'lichanghua' , 'dujieying' , 'yangfan', 'fengwenwei' , 'chengjie' ,'hesufang' , 'tuliandi' , 'wangsuichun' , 'yujinfeng' , 'yangyayi' , 'xujun' ,] 
# ip_dirs = []
# stl_dirs = []
# results_dirs = []
# for data_name in data_names:
#     ip_dirs.append(cwd + '/data/' + data_name + '/implant_point')
#     stl_dirs.append(cwd + '/data/' + data_name + '/stl')
#     results_dirs.append(cwd + '/output/' + data_name)

# crop_radius = [35, 40, 45, 50, 55]
# clip_distance = [38, 40, 42, 44, 46]
# # for cr in crop_radius:
# #     for i in range(len(data_names)):
# #         print(cr, data_names[i])
# #         crop_max_man(stl_dirs[i], ip_dirs[i], cr)
# for cd in clip_distance:
#     for i in range(len(data_names)):
#         print('50', cd, data_names[i])
#         iden_zygo_bone(stl_dirs[i], ip_dirs[i], 50, cd)
# iters = [1, 4 , 7, 10, 13]
# fill_zygomatic_pcd(stl_dirs[0], ip_dirs[0], 50, 42, iters)
# auto_dict = {}
# for key, _ in paths.auto_dict.items():
#     print("planning ", key)
#     optimal_paths = zygomatic_implant_planning(cwd + '/data/' + key + '/stl', cwd + '/data/' + key + '/implant_point', cwd + '/output/' + key)
#     auto_dict[key] = [np.array(optimal_paths[0][0]), np.array(optimal_paths[0][1]),
#                       np.array(optimal_paths[1][0]), np.array(optimal_paths[1][1]),
#                       np.array(optimal_paths[2][0]), np.array(optimal_paths[2][1]),
#                       np.array(optimal_paths[3][0]), np.array(optimal_paths[3][1])]
    
# auto_margin = {}
# manu_margin = {}
# for key, _ in paths.auto_dict.items():
#     auto_paths = auto_dict[key]
#     manu_paths = paths.manu_dict[key]
#     control_points = paths.control_points[key]
#     auto_margin[key] = [min(min(geometry.point_to_segment_distance(control_points[0:3], auto_paths[0], auto_paths[0]+auto_paths[1]), 
#                            geometry.point_to_segment_distance(control_points[0:3], auto_paths[2], auto_paths[2]+auto_paths[3])),
#                        min(geometry.point_to_segment_distance(control_points[0:3], auto_paths[4], auto_paths[4]+auto_paths[5]), 
#                            geometry.point_to_segment_distance(control_points[0:3], auto_paths[6], auto_paths[6]+auto_paths[7])),
#                       ),
#                    min(min(geometry.point_to_segment_distance(control_points[3:6], auto_paths[0], auto_paths[0]+auto_paths[1]), 
#                            geometry.point_to_segment_distance(control_points[3:6], auto_paths[2], auto_paths[2]+auto_paths[3])),
#                        min(geometry.point_to_segment_distance(control_points[3:6], auto_paths[4], auto_paths[4]+auto_paths[5]), 
#                            geometry.point_to_segment_distance(control_points[3:6], auto_paths[6], auto_paths[6]+auto_paths[7])),
#                    ),
#                    min(min(geometry.point_to_segment_distance(control_points[6:9], auto_paths[0], auto_paths[0]+auto_paths[1]), 
#                            geometry.point_to_segment_distance(control_points[6:9], auto_paths[2], auto_paths[2]+auto_paths[3])),
#                        min(geometry.point_to_segment_distance(control_points[6:9], auto_paths[4], auto_paths[4]+auto_paths[5]), 
#                            geometry.point_to_segment_distance(control_points[6:9], auto_paths[6], auto_paths[6]+auto_paths[7])),
#                    ),
#                    min(min(geometry.point_to_segment_distance(control_points[9:12], auto_paths[0], auto_paths[0]+auto_paths[1]), 
#                            geometry.point_to_segment_distance(control_points[9:12], auto_paths[2], auto_paths[2]+auto_paths[3])),
#                        min(geometry.point_to_segment_distance(control_points[9:12], auto_paths[4], auto_paths[4]+auto_paths[5]), 
#                            geometry.point_to_segment_distance(control_points[9:12], auto_paths[6], auto_paths[6]+auto_paths[7])),
#                    )
#                   ]
#     manu_margin[key] = [min(min(geometry.point_to_segment_distance(control_points[0:3], manu_paths[0], manu_paths[0]+manu_paths[1]), 
#                            geometry.point_to_segment_distance(control_points[0:3], manu_paths[2], manu_paths[2]+manu_paths[3])),
#                        min(geometry.point_to_segment_distance(control_points[0:3], manu_paths[4], manu_paths[4]+manu_paths[5]), 
#                            geometry.point_to_segment_distance(control_points[0:3], manu_paths[6], manu_paths[6]+manu_paths[7])),
#                       ),
#                    min(min(geometry.point_to_segment_distance(control_points[3:6], manu_paths[0], manu_paths[0]+manu_paths[1]), 
#                            geometry.point_to_segment_distance(control_points[3:6], manu_paths[2], manu_paths[2]+manu_paths[3])),
#                        min(geometry.point_to_segment_distance(control_points[3:6], manu_paths[4], manu_paths[4]+manu_paths[5]), 
#                            geometry.point_to_segment_distance(control_points[3:6], manu_paths[6], manu_paths[6]+manu_paths[7])),
#                    ),
#                    min(min(geometry.point_to_segment_distance(control_points[6:9], manu_paths[0], manu_paths[0]+manu_paths[1]), 
#                            geometry.point_to_segment_distance(control_points[6:9], manu_paths[2], manu_paths[2]+manu_paths[3])),
#                        min(geometry.point_to_segment_distance(control_points[6:9], manu_paths[4], manu_paths[4]+manu_paths[5]), 
#                            geometry.point_to_segment_distance(control_points[6:9], manu_paths[6], manu_paths[6]+manu_paths[7])),
#                    ),
#                    min(min(geometry.point_to_segment_distance(control_points[9:12], manu_paths[0], manu_paths[0]+manu_paths[1]), 
#                            geometry.point_to_segment_distance(control_points[9:12], manu_paths[2], manu_paths[2]+manu_paths[3])),
#                        min(geometry.point_to_segment_distance(control_points[9:12], manu_paths[4], manu_paths[4]+manu_paths[5]), 
#                            geometry.point_to_segment_distance(control_points[9:12], manu_paths[6], manu_paths[6]+manu_paths[7])),
#                    )
#                   ]
#     planning_pcd = get_planning_pcd(cwd + '/data/' + key + '/stl', cwd + '/data/' + key + '/implant_point')
#     auto_bic = [cal_BIC(auto_paths[0:2], planning_pcd), cal_BIC(auto_paths[2:4], planning_pcd), cal_BIC(auto_paths[4:6], planning_pcd), cal_BIC(auto_paths[6:8], planning_pcd)]
#     manu_bic = [cal_BIC(manu_paths[0:2], planning_pcd), cal_BIC(manu_paths[2:4], planning_pcd), cal_BIC(manu_paths[4:6], planning_pcd), cal_BIC(manu_paths[6:8], planning_pcd)]
#     print(key," auto:", auto_margin[key][0], ' ', auto_margin[key][1], ' ', auto_margin[key][2], ' ', auto_margin[key][3])
#     print(key," manu:", manu_margin[key][0], ' ', manu_margin[key][1], ' ', manu_margin[key][2], ' ', manu_margin[key][3])
#     print("auto_bic:", auto_bic, ' manu_bic:', manu_bic)



# 设置全局字体为 Times New Roman，加粗，20 号字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 40

# all_data = [[8.2232,10.4427,10.5822,8.8245,7.9977,10.3637,8.4512,11.5014,9.5726,12.3052,11.0142,7.4281,5.4279,7.3477,10.5449,7.5126,5.4685,11.0724],
#             [6.7707,3.8667,8.195,8.2197,5.926,6.2079,7.0794,11.9289,10.3183,13.1783,7.9475,9.5039,7.9765,9.2976,7.3078,7.8051,8.2726,7.9143],
#             [10.4236,6.0241,7.6121,8.3671,5.0166,1.9104,8.4137,10.7386,8.1573,9.8856,9.0507,9.1563,8.4219,5.903,7.4539,6.2764,11.7491,9.7753],
#             [4.3075,7.6627,10.8703,5.6315,8.1949,9.0354,9.3396,5.4985,6.2487,11.1497,7.9616,4.4123,7.1154,9.0016,14.165,9.357,5.7021,9.025],
#             [7.8839,8.8771,9.564,7.6337,7.0436,6.3969,7.3831,9.23,8.6555,11.0551,8.2281,10.7313,7.7266,7.7915,9.6176,9.4356,5.2272,8.9622],
#             [5.8497,3.9007,8.7343,6.1751,4.1231,7.4142,4.5782,6.1172,7.9533,11.6751,8.5355,6.4158,6.3453,7.7081,5.8047,5.6087,8.369,7.7713],
#             [7.7134,5.39,8.2585,6.8176,5.3219,6.5324,8.139,6.7682,5.3046,11.7542,8.0469,6.8253,4.799,5.587,10.0908,6.7363,10.0728,6.6641],
#             [7.0056,6.2546,8.1241,7.8125,4.2322,4.5691,8.5874,6.5652,6.3644,8.618,9.0375,7.9095,7.0026,8.0186,6.1675,7.2978,6.2485,7.9541]]

# # 设置箱线图的位置
# positions = [1,1.5, 2.5, 3, 4, 4.5, 5.5, 6]

# #首先有图（fig），然后有轴（ax）
# fig,axes=plt.subplots(nrows=1,ncols=1)

# plot_data = [all_data[0], all_data[4], all_data[1], all_data[5], all_data[2], all_data[6], all_data[3], all_data[7]]

# # 绘制箱线图
# bplot = axes.boxplot(plot_data, widths=0.4, positions=positions, meanprops={'marker':'D', 'markerfacecolor':'r', 'color':'r', 'markersize':10}, patch_artist=True, capprops={'linewidth': 2}, showmeans=True, showfliers=True, showbox=True, medianprops=dict(color='black', linestyle='--', linewidth=2), boxprops={'linewidth': 2, 'linestyle': '-'})

# # 设置箱子的颜色
# colors = ['deepskyblue', 'gray', 'deepskyblue', 'gray','deepskyblue', 'gray','deepskyblue', 'gray','pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.6)
    
# # for i, d in enumerate(plot_data):
# #     y = np.random.normal(positions[i], 0.06, size=len(d))
# #     axes.scatter(y, d, alpha=0.6, color=colors[i%2],linewidths=0, zorder=1, s=200)


# axes.yaxis.grid(True) #在y轴上添加网格线
# axes.set_xticks([1.25, 2.75, 4.25, 5.75]) #指定x轴的轴刻度个数
# axes.set_xticklabels(['d$_{%s}$ (mm)'%1, 'd$_{%s}$ (mm)'%2, 'd$_{%s}$ (mm)'%3, 'd$_{%s}$ (mm)'%4], fontsize=40, fontweight='bold') #设置刻度标签
# axes.axvline(2,  linestyle='--', color='black')
# axes.axvline(3.5,  linestyle='--', color='black')
# axes.axvline(5,  linestyle='--', color='black')
# # 添加图例
# # fig.legend(bplot1["boxes"], ['our automatic planning results', 'manual planning results'], loc="lower center", bbox_to_anchor=(-0.5, -0.1), ncol=2)

# # 设置图表标题和坐标轴标签的字体
# plt.title('$\\varepsilon_r$ = 0.1', fontsize=48, fontweight='bold')

# # 显示图表
# plt.show()

# all_data = [[8.2232,8.8739,12.6644,8.2983,7.9977,10.3639,6.3613,11.5014,10.3183,13.1783,11.0142,5.6715,8.1711,7.7266,10.5449,4.7248,5.4832,12.3748],
#             [6.7707,3.8666,8.1950,8.7818,5.3916,6.2079,7.6621,9.5299,6.7032,11.8585,7.9475,10.9676,7.9765,9.3064,7.3078,7.8051,8.2726,6.5291],
#             [10.5793,6.0241,8.0597,8.3671,5.0881,7.9613,10.1725,10.7386,7.3065,9.7603,9.0507,9.1563,9.2818,5.9030,7.4539,6.2764,6.5476,10.5962],
#             [6.1847,7.6627,11.0092,8.3582,6.9488,5.7045,5.4029,5.4985,6.2487,11.1497,7.3756,5.5941,4.6909,9.0016,10.3178,9.3570,11.7491,6.2871],
#             [7.8839,8.8771,9.564,7.6337,7.0436,6.3969,7.3831,9.2300,8.6555,11.0551,8.2281,10.7313,7.7266,7.7915,9.6176,9.4356,5.2272,8.9622],
#             [5.8497,3.9007,8.7343,6.1751,4.1231,7.4142,4.5782,6.1172,7.9533,11.6751,8.5355,6.4158,6.3453,7.7081,5.8047,5.6087,8.369,7.7713],
#             [7.7134,5.3900,8.2585,6.8176,5.3219,6.5324,8.1390,6.7682,5.3046,11.7542,8.0469,6.8253,4.7990,5.5870,10.0908,6.7363,10.0728,6.6641],
#             [7.0056,6.2546,8.1241,7.8125,4.2322,4.5691,8.5874,6.5652,6.3644,8.6180,9.0375,7.9095,7.0026,8.0186,6.1675,7.2978,6.2485,7.9541]]


# #首先有图（fig），然后有轴（ax）
# fig,axes=plt.subplots(nrows=1,ncols=1)

# plot_data = [all_data[0], all_data[4], all_data[1], all_data[5], all_data[2], all_data[6], all_data[3], all_data[7]]

# # 绘制箱线图
# bplot = axes.boxplot(plot_data, widths=0.4, positions=positions, meanprops={'marker':'D', 'markerfacecolor':'r', 'color':'r', 'markersize':10}, patch_artist=True, capprops={'linewidth': 2}, showmeans=True, showfliers=True, showbox=True, medianprops=dict(color='black', linestyle='--', linewidth=2), boxprops={'linewidth': 2, 'linestyle': '-'})

# # 设置箱子的颜色
# colors = ['deepskyblue', 'gray', 'deepskyblue', 'gray','deepskyblue', 'gray','deepskyblue', 'gray','pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.6)
    
# # for i, d in enumerate(plot_data):
# #     y = np.random.normal(positions[i], 0.06, size=len(d))
# #     axes.scatter(y, d, alpha=0.6, color=colors[i%2],linewidths=0, zorder=1, s=200)


# axes.yaxis.grid(True) #在y轴上添加网格线
# axes.set_xticks([1.25, 2.75, 4.25, 5.75]) #指定x轴的轴刻度个数
# axes.set_xticklabels(['d$_{%s}$ (mm)'%1, 'd$_{%s}$ (mm)'%2, 'd$_{%s}$ (mm)'%3, 'd$_{%s}$ (mm)'%4], fontsize=40, fontweight='bold') #设置刻度标签
# axes.axvline(2,  linestyle='--', color='black')
# axes.axvline(3.5,  linestyle='--', color='black')
# axes.axvline(5,  linestyle='--', color='black')

# # 添加图例
# # fig.legend(bplot1["boxes"], ['our automatic planning results', 'manual planning results'], loc="lower center", bbox_to_anchor=(-0.5, -0.1), ncol=2)

# # 设置图表标题和坐标轴标签的字体
# plt.title('$\\varepsilon_r$ = 0.2', fontsize=48, fontweight='bold')

# # 显示图表
# plt.show()

# all_data = [[6.4342,9.0378,12.6644,7.4612,7.211,8.6778,6.3969,11.5014,11.0776,12.7362,11.0142,7.0263,6.8284,7.7266,10.5449,6.0847,6.4328,12.3748],
#             [7.8589,3.8667,8.195,8.7818,5.3916,6.2079,7.6621,10.5799,8.2202,12.1537,7.9475,10.9676,7.9765,9.3064,7.7282,7.8051,8.2726,6.3093],
#             [10.5794,6.0241,8.0597,8.3671,5.0881,7.6934,9.6644,9.4526,7.3065,9.7603,9.0507,9.1563,8.9349,5.903,7.4539,6.7634,11.8341,10.7131 ],
#             [6.1848,7.6627,11.0093,8.8419,6.9488,6.5215,5.4029,5.4985,6.2487,11.1497,9.4043,5.5941,6.4031,9.0016,12.165,9.5979,6.0584,6.638 ],
#             [7.8839,8.8771,9.564,7.6337,7.0436,6.3969,7.3831,9.23,8.6555,11.0551,8.2281,10.7313,7.7266,7.7915,9.6176,9.4356,5.2272,8.9622 ],
#             [5.8497,3.9007,8.7343,6.1751,4.1231,7.4142,4.5782,6.1172,7.9533,11.6751,8.5355,6.4158,6.3453,7.7081,5.8047,5.6087,8.369,7.7713],
#             [7.7134,5.3900,8.2585,6.8176,5.3219,6.5324,8.1390,6.7682,5.3046,11.7542,8.0469,6.8253,4.7990,5.5870,10.0908,6.7363,10.0728,6.6641],
#             [7.0056,6.2546,8.1241,7.8125,4.2322,4.5691,8.5874,6.5652,6.3644,8.6180,9.0375,7.9095,7.0026,8.0186,6.1675,7.2978,6.2485,7.9541]]


# #首先有图（fig），然后有轴（ax）
# fig,axes=plt.subplots(nrows=1,ncols=1)

# plot_data = [all_data[0], all_data[4], all_data[1], all_data[5], all_data[2], all_data[6], all_data[3], all_data[7]]

# # 绘制箱线图
# bplot = axes.boxplot(plot_data, widths=0.4, positions=positions, meanprops={'marker':'D', 'markerfacecolor':'r', 'color':'r', 'markersize':10}, patch_artist=True, capprops={'linewidth': 2}, showmeans=True, showfliers=True, showbox=True, medianprops=dict(color='black', linestyle='--', linewidth=2), boxprops={'linewidth': 2, 'linestyle': '-'})

# # 设置箱子的颜色
# colors = ['deepskyblue', 'gray', 'deepskyblue', 'gray','deepskyblue', 'gray','deepskyblue', 'gray','pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.6)
    
# # for i, d in enumerate(plot_data):
# #     y = np.random.normal(positions[i], 0.06, size=len(d))
# #     axes.scatter(y, d, alpha=0.6, color=colors[i%2],linewidths=0, zorder=1, s=200)


# axes.yaxis.grid(True) #在y轴上添加网格线
# axes.set_xticks([1.25, 2.75, 4.25, 5.75]) #指定x轴的轴刻度个数
# axes.set_xticklabels(['d$_{%s}$ (mm)'%1, 'd$_{%s}$ (mm)'%2, 'd$_{%s}$ (mm)'%3, 'd$_{%s}$ (mm)'%4], fontsize=40, fontweight='bold') #设置刻度标签
# axes.axvline(2,  linestyle='--', color='black')
# axes.axvline(3.5,  linestyle='--', color='black')
# axes.axvline(5,  linestyle='--', color='black')

# # 添加图例
# # fig.legend(bplot1["boxes"], ['our automatic planning results', 'manual planning results'], loc="lower center", bbox_to_anchor=(-0.5, -0.1), ncol=2)

# # 设置图表标题和坐标轴标签的字体
# plt.title('$\\varepsilon_r$ = 0.3', fontsize=48, fontweight='bold')

# # 显示图表
# plt.show()

# all_data = [[6.4342,9.0378,12.6644,8.2983,7.211,8.6778,6.8457,11.5014,11.62,12.7362,11.0142,7.4281,6.8284,7.7266,10.5449,6.0847,5.4832,11.0724],
#             [7.8589,3.8667,8.7808,8.7818,5.3916,6.2079,7.6621,10.5799,7.4381,12.1537,7.9475,10.8172,7.9765,9.3064,7.3078,7.8051,8.2726,7.7757],
#             [10.5778,6.0241,7.6121,10.1081,5.0881,7.6934,9.6644,10.9497,8.1574,9.7603,9.0507,9.1563,9.2818,5.903,7.4539,6.7634,11.8996,10.7131],
#             [4.7418,7.6627,9.5938,7.8931,6.9488,6.5215,5.4029,5.4985,6.2487,11.1497,9.4043,5.5941,4.6909,9.0016,10.3178,9.5979,6.0584,6.638],
#             [7.8839,8.8771,9.564,7.6337,7.0436,6.3969,7.3831,9.2300,8.6555,11.0551,8.2281,10.7313,7.7266,7.7915,9.6176,9.4356,5.2272,8.9622],
#             [5.8497,3.9007,8.7343,6.1751,4.1231,7.4142,4.5782,6.1172,7.9533,11.6751,8.5355,6.4158,6.3453,7.7081,5.8047,5.6087,8.369,7.7713],
#             [7.7134,5.3900,8.2585,6.8176,5.3219,6.5324,8.1390,6.7682,5.3046,11.7542,8.0469,6.8253,4.7990,5.5870,10.0908,6.7363,10.0728,6.6641],
#             [7.0056,6.2546,8.1241,7.8125,4.2322,4.5691,8.5874,6.5652,6.3644,8.6180,9.0375,7.9095,7.0026,8.0186,6.1675,7.2978,6.2485,7.9541]]


# #首先有图（fig），然后有轴（ax）
# fig,axes=plt.subplots(nrows=1,ncols=1)

# plot_data = [all_data[0], all_data[4], all_data[1], all_data[5], all_data[2], all_data[6], all_data[3], all_data[7]]

# # 绘制箱线图
# bplot = axes.boxplot(plot_data, widths=0.4, positions=positions, meanprops={'marker':'D', 'markerfacecolor':'r', 'color':'r', 'markersize':10}, patch_artist=True, capprops={'linewidth': 2}, showmeans=True, showfliers=True, showbox=True, medianprops=dict(color='black', linestyle='--', linewidth=2), boxprops={'linewidth': 2, 'linestyle': '-'})

# # 设置箱子的颜色
# colors = ['deepskyblue', 'gray', 'deepskyblue', 'gray','deepskyblue', 'gray','deepskyblue', 'gray','pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.6)
    
# # for i, d in enumerate(plot_data):
# #     y = np.random.normal(positions[i], 0.06, size=len(d))
# #     axes.scatter(y, d, alpha=0.6, color=colors[i%2],linewidths=0, zorder=1, s=200)


# axes.yaxis.grid(True) #在y轴上添加网格线
# axes.set_xticks([1.25, 2.75, 4.25, 5.75]) #指定x轴的轴刻度个数
# axes.set_xticklabels(['d$_{%s}$ (mm)'%1, 'd$_{%s}$ (mm)'%2, 'd$_{%s}$ (mm)'%3, 'd$_{%s}$ (mm)'%4], fontsize=40, fontweight='bold') #设置刻度标签
# axes.axvline(2,  linestyle='--', color='black')
# axes.axvline(3.5,  linestyle='--', color='black')
# axes.axvline(5,  linestyle='--', color='black')

# # 添加图例
# # fig.legend(bplot1["boxes"], ['our automatic planning results', 'manual planning results'], loc="lower center", bbox_to_anchor=(-0.5, -0.1), ncol=2)

# # 设置图表标题和坐标轴标签的字体
# plt.title('$\\varepsilon_r$ = 0.4', fontsize=48, fontweight='bold')

# # 显示图表
# plt.show()




#### BIC ######


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    # ax.set_ylim(0, 2.5)
    # ax.set_xlabel('Sample name')


data = [[1.184713376,1.5,0.935897436,1.072072072,1.618181818,1.121621622,1.192771084,1.814814815,1.261538462,1.320987654,1.054054054,1.691176471,2.016666667,1.366972477,1.869565217,1.469387755,1.899082569,1.218181818],
        [1.108695652,1.102564103,1.279411765,1.013422819,1.068493151,1.319148936,1.345454545,1.551724138,1.685714286,1.185185185,1.268518519,0.908256881,1.183486239,1.108695652,1.322580645,1.478873239,1.284883721,0.934210526],
        [1.079136691,1.2,1.490196078,1.093333333,1.145833333,1.055555556,1.24691358,1.384615385,1.366071429,1.296703297,1.098214286,0.912,1.376,1.403100775,1.576923077,1.295454545,1.175879397,1.051948052],
        [1.779661017,1.454545455,1.084507042,1.29245283,1.430379747,1.970588235,1.263888889,1.135135135,1.318681319,1.3625,1.11965812,1.311111111,1.1,1.764150943,1.701754386,1.875,1.30952381,1.510638298],
        ]


colors = ['cyan', 'olive', 'deeppink','sienna']




fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

positions = [1, 2, 3, 4]

# bplot = ax2.boxplot(data, widths=0.8, positions=positions, showbox=False, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=False, showfliers=False, medianprops=dict(color='black', linewidth=2), boxprops=None)

# 设置箱子的颜色
colors = ['teal', 'chocolate', 'purple', 'red', 'crimson', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

markers = ['o', 'D', 's', '^']
for i, d in enumerate(data):
    y = np.random.normal(positions[i], 0.15, size=len(d))
    ax2.scatter(y, d, alpha=0.9, marker=markers[i], color=colors[i], linewidths=0, zorder=1, s=300)


ax2.vlines(0.6, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.4, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.6, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.4, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.6, ymin=0,  ymax=np.median(np.array(data[2])), linestyle='-', linewidth=2, color='black')
ax2.vlines(3.4, ymin=0,  ymax=np.median(np.array(data[2])), linestyle='-', linewidth=2, color='black')
ax2.vlines(3.6, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
ax2.vlines(4.4, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1, ymin=np.min(np.array(data[0])),  ymax=np.max(np.array(data[0])), linestyle='-', linewidth=1, color='black')
ax2.vlines(2, ymin=np.min(np.array(data[1])),  ymax=np.max(np.array(data[1])), linestyle='-', linewidth=1, color='black')
ax2.vlines(3, ymin=np.min(np.array(data[2])),  ymax=np.max(np.array(data[2])), linestyle='-', linewidth=1, color='black')
ax2.vlines(4, ymin=np.min(np.array(data[3])),  ymax=np.max(np.array(data[3])), linestyle='-', linewidth=1, color='black')

ax2.hlines(max(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[2])), 2.8, 3.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[2])), 2.8, 3.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[3])), 3.8, 4.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[3])), 3.8, 4.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[0])), 0.6, 1.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[1])), 1.6, 2.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[2])), 2.6, 3.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[3])), 3.6, 4.4, linestyle='-', linewidth=2, color='black')
ax2.hlines([1], 0, 5, color='r', linestyle='--', lw=2)
# ax2.axvline(1.4, ymin=0,  ymax=max(np.median(np.array(data[1])), np.median(np.array(data[0]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(2.4, ymin=0,  ymax=max(np.median(np.array(data[1])), np.median(np.array(data[2]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(3.5, ymin=0,  ymax=max(np.median(np.array(data[2])), np.median(np.array(data[3]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(4.5, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
# bplot = axes.boxplot(data, widths=0.4, positions=positions, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=True, showfliers=False, showbox=True, medianprops=dict(color='black', linewidth=2), boxprops={'linewidth': 2, 'linestyle': '-'})
ax2.set_ylim(0, 2.5)
ax2.set_title('$\\varepsilon_r$ = 0.1', fontsize=48, fontweight='bold')
# parts = ax2.violinplot(
#         data, showmeans=True, showmedians=False,
#         showextrema=True)
# i = 0
# for pc in parts['bodies']:
#     pc.set_facecolor('cyan')
#     pc.set_edgecolor('black')
#     i = i + 1
    # pc.set_alpha(0.6)

# quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
# whiskers = np.array([
#     adjacent_values(sorted_array, q1, q3)
#     for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
# whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

# inds = np.arange(1, len(medians) + 1)
# ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
# ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
# ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
# set style for the axes
labels = ['BIC1', 'BIC2', 'BIC3', 'BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()



fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.1', fontsize=48, fontweight='bold')

positions = [1, 2]
data = [np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])]
# bplot = ax2.boxplot(data, widths=0.4, positions=positions, showbox=False, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=False, showfliers=False, medianprops=dict(color='black', linewidth=2), boxprops=None)

ax2.vlines(0.8, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.2, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.8, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.2, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1, ymin=np.min(np.array(data[0])),  ymax=np.max(np.array(data[0])), linestyle='-', linewidth=1, color='black')
ax2.vlines(2, ymin=np.min(np.array(data[1])),  ymax=np.max(np.array(data[1])), linestyle='-', linewidth=1, color='black')
ax2.set_ylim(0, 4)
# 设置箱子的颜色
colors = ['teal', 'chocolate', 'purple', 'red', 'crimson', 'deepskyblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

markers = ['o', 'D', 's', '^']
for i, d in enumerate(data):
    y = np.random.normal(positions[i], 0.08, size=len(d))
    ax2.scatter(y, d, alpha=0.9, marker=markers[i], color=colors[i+4], linewidths=0, zorder=1, s=300)

ax2.hlines(max(np.array(data[0])), 0.9, 1.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[0])), 0.9, 1.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[1])), 1.9, 2.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[1])), 1.9, 2.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines([2], 0, 5, color='r', linestyle='--', lw=2)

# set style for the axes
labels = ['BIC1+BIC2', 'BIC3+BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()




data = [[1.120743034,1.383458647,1.006756757,1.177339901,1.989130435,0.909722222,1.375,2,1.303278689,1.185628743,1.10701107,1.510638298,1.9140625,1.28959276,1.351851852,1.699421965,1.806167401,1.112244898],
        [1.227848101,1.163120567,1.401360544,1.049469965,1.028368794,1.169082126,0.992063492,1.344262295,1.113207547,1.139534884,1.195121951,0.821256039,0.803652968,1.015789474,1.221311475,1.340136054,0.981382979,0.845679012],
        [0.917322835,1.048275862,1.573033708,1,1.087179487,1.276836158,1.045714286,1.384615385,1.340080972,1.179775281,0.99595,0.939271255,1.296296296,1.251937984,1.170940171,1.19047619,1.026190476,1.044871795],
        [1.516981132,1.427272727,0.934640523,1.202764977,1.238993711,1.819672131,1.152777778,1.05085,1.342696629,1.14619883,1.285046729,1.311111111,0.943127962,1.657276995,1.472,1.802325581,1.25228,1.281818182],
        ]

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.2', fontsize=48, fontweight='bold')
positions = [1, 2, 3, 4]

# bplot = ax2.boxplot(data, widths=0.8, positions=positions, showbox=False, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=False, showfliers=False, medianprops=dict(color='black', linewidth=2), boxprops=None)

# 设置箱子的颜色
colors = ['teal', 'chocolate', 'purple', 'red', 'crimson', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

markers = ['o', 'D', 's', '^']
for i, d in enumerate(data):
    y = np.random.normal(positions[i], 0.15, size=len(d))
    ax2.scatter(y, d, alpha=0.9, marker=markers[i], color=colors[i], linewidths=0, zorder=1, s=300)


ax2.vlines(0.6, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.4, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.6, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.4, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.6, ymin=0,  ymax=np.median(np.array(data[2])), linestyle='-', linewidth=2, color='black')
ax2.vlines(3.4, ymin=0,  ymax=np.median(np.array(data[2])), linestyle='-', linewidth=2, color='black')
ax2.vlines(3.6, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
ax2.vlines(4.4, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1, ymin=np.min(np.array(data[0])),  ymax=np.max(np.array(data[0])), linestyle='-', linewidth=1, color='black')
ax2.vlines(2, ymin=np.min(np.array(data[1])),  ymax=np.max(np.array(data[1])), linestyle='-', linewidth=1, color='black')
ax2.vlines(3, ymin=np.min(np.array(data[2])),  ymax=np.max(np.array(data[2])), linestyle='-', linewidth=1, color='black')
ax2.vlines(4, ymin=np.min(np.array(data[3])),  ymax=np.max(np.array(data[3])), linestyle='-', linewidth=1, color='black')
ax2.hlines(max(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[2])), 2.8, 3.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[2])), 2.8, 3.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[3])), 3.8, 4.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[3])), 3.8, 4.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[0])), 0.6, 1.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[1])), 1.6, 2.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[2])), 2.6, 3.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[3])), 3.6, 4.4, linestyle='-', linewidth=2, color='black')
# ax2.axvline(1.4, ymin=0,  ymax=max(np.median(np.array(data[1])), np.median(np.array(data[0]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(2.4, ymin=0,  ymax=max(np.median(np.array(data[1])), np.median(np.array(data[2]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(3.5, ymin=0,  ymax=max(np.median(np.array(data[2])), np.median(np.array(data[3]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(4.5, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
# bplot = axes.boxplot(data, widths=0.4, positions=positions, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=True, showfliers=False, showbox=True, medianprops=dict(color='black', linewidth=2), boxprops={'linewidth': 2, 'linestyle': '-'})
ax2.set_ylim(0, 2.5)
# parts = ax2.violinplot(
#         data, showmeans=True, showmedians=False,
#         showextrema=True)
# i = 0
# for pc in parts['bodies']:
#     pc.set_facecolor('cyan')
#     pc.set_edgecolor('black')
#     i = i + 1
    # pc.set_alpha(0.6)

# quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
# whiskers = np.array([
#     adjacent_values(sorted_array, q1, q3)
#     for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
# whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

# inds = np.arange(1, len(medians) + 1)
# ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
# ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
# ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([1], 0, 5, color='r', linestyle='--', lw=2)

# set style for the axes
labels = ['BIC1', 'BIC2', 'BIC3', 'BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.2', fontsize=48, fontweight='bold')

positions = [1, 2]
data = [np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])]
# bplot = ax2.boxplot(data, widths=0.4, positions=positions, showbox=False, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=False, showfliers=False, medianprops=dict(color='black', linewidth=2), boxprops=None)

ax2.vlines(0.8, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.2, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.8, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.2, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1, ymin=np.min(np.array(data[0])),  ymax=np.max(np.array(data[0])), linestyle='-', linewidth=1, color='black')
ax2.vlines(2, ymin=np.min(np.array(data[1])),  ymax=np.max(np.array(data[1])), linestyle='-', linewidth=1, color='black')
ax2.set_ylim(0, 4)
# 设置箱子的颜色
colors = ['teal', 'chocolate', 'purple', 'red', 'crimson', 'deepskyblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

markers = ['o', 'D', 's', '^']
for i, d in enumerate(data):
    y = np.random.normal(positions[i], 0.08, size=len(d))
    ax2.scatter(y, d, alpha=0.9, marker=markers[i], color=colors[i+4], linewidths=0, zorder=1, s=300)

ax2.hlines(max(np.array(data[0])), 0.9, 1.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[0])), 0.9, 1.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[1])), 1.9, 2.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[1])), 1.9, 2.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines([2], 0, 5, color='r', linestyle='--', lw=2)


# set style for the axes
labels = ['BIC1+BIC2', 'BIC3+BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()




data = [[1.137931034,1.425414365,1.042056075,1.252631579,2.067669173,0.960591133,1.865248227,1.583333333,1.096153846,1.023255814,1.037688442,1.455399061,1.850515464,1.119241192,1.375722543,1.537162162,1.766153846,1.312977099],
        [0.944680851,1.098214286,1.4375,1.063414634,1.067357513,1.096774194,1.070351759,1.337078652,1.19269103,1.118081181,1.086826347,1.098101266,1.113207547,0.996563574,1.173684211,1.225108225,1.058303887,1.026200873],
        [1.041775457,0.981818182,1.456953642,0.941309255,1.11827957,1.212927757,0.974452555,1.301369863,1.242819843,1.209125475,1.024861878,0.989247312,1.365384615,1.194300518,1.201086957,1.141230068,1.066985646,1.004237288],
        [1.504975124,1.329479769,0.924107143,1.186335404,1.080321285,2.093023256,1.25,1.23943662,1.414728682,1.101886792,1.161849711,1.347826087,0.982817869,1.520710059,1.369791667,1.73151751,1.16032,1.214285714],
        ]

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.3', fontsize=48, fontweight='bold')
positions = [1, 2, 3, 4]

# bplot = ax2.boxplot(data, widths=0.8, positions=positions, showbox=False, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=False, showfliers=False, medianprops=dict(color='black', linewidth=2), boxprops=None)

# 设置箱子的颜色
colors = ['teal', 'chocolate', 'purple', 'red', 'crimson', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

markers = ['o', 'D', 's', '^']
for i, d in enumerate(data):
    y = np.random.normal(positions[i], 0.15, size=len(d))
    ax2.scatter(y, d, alpha=0.9, marker=markers[i], color=colors[i], linewidths=0, zorder=1, s=300)


ax2.vlines(0.6, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.4, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.6, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.4, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.6, ymin=0,  ymax=np.median(np.array(data[2])), linestyle='-', linewidth=2, color='black')
ax2.vlines(3.4, ymin=0,  ymax=np.median(np.array(data[2])), linestyle='-', linewidth=2, color='black')
ax2.vlines(3.6, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
ax2.vlines(4.4, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1, ymin=np.min(np.array(data[0])),  ymax=np.max(np.array(data[0])), linestyle='-', linewidth=1, color='black')
ax2.vlines(2, ymin=np.min(np.array(data[1])),  ymax=np.max(np.array(data[1])), linestyle='-', linewidth=1, color='black')
ax2.vlines(3, ymin=np.min(np.array(data[2])),  ymax=np.max(np.array(data[2])), linestyle='-', linewidth=1, color='black')
ax2.vlines(4, ymin=np.min(np.array(data[3])),  ymax=np.max(np.array(data[3])), linestyle='-', linewidth=1, color='black')
ax2.hlines(max(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[2])), 2.8, 3.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[2])), 2.8, 3.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[3])), 3.8, 4.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[3])), 3.8, 4.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[0])), 0.6, 1.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[1])), 1.6, 2.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[2])), 2.6, 3.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[3])), 3.6, 4.4, linestyle='-', linewidth=2, color='black')
# ax2.axvline(1.4, ymin=0,  ymax=max(np.median(np.array(data[1])), np.median(np.array(data[0]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(2.4, ymin=0,  ymax=max(np.median(np.array(data[1])), np.median(np.array(data[2]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(3.5, ymin=0,  ymax=max(np.median(np.array(data[2])), np.median(np.array(data[3]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(4.5, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
# bplot = axes.boxplot(data, widths=0.4, positions=positions, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=True, showfliers=False, showbox=True, medianprops=dict(color='black', linewidth=2), boxprops={'linewidth': 2, 'linestyle': '-'})
ax2.set_ylim(0, 2.5)
# parts = ax2.violinplot(
#         data, showmeans=True, showmedians=False,
#         showextrema=True)
# i = 0
# for pc in parts['bodies']:
#     pc.set_facecolor('cyan')
#     pc.set_edgecolor('black')
#     i = i + 1
    # pc.set_alpha(0.6)

# quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
# whiskers = np.array([
#     adjacent_values(sorted_array, q1, q3)
#     for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
# whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

# inds = np.arange(1, len(medians) + 1)
# ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
# ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
# ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([1], 0, 5, color='r', linestyle='--', lw=2)

# set style for the axes
labels = ['BIC1', 'BIC2', 'BIC3', 'BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.3', fontsize=48, fontweight='bold')

positions = [1, 2]
data = [np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])]
# bplot = ax2.boxplot(data, widths=0.4, positions=positions, showbox=False, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=False, showfliers=False, medianprops=dict(color='black', linewidth=2), boxprops=None)

ax2.vlines(0.8, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.2, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.8, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.2, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1, ymin=np.min(np.array(data[0])),  ymax=np.max(np.array(data[0])), linestyle='-', linewidth=1, color='black')
ax2.vlines(2, ymin=np.min(np.array(data[1])),  ymax=np.max(np.array(data[1])), linestyle='-', linewidth=1, color='black')
ax2.set_ylim(0, 4)
# 设置箱子的颜色
colors = ['teal', 'chocolate', 'purple', 'red', 'crimson', 'deepskyblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

markers = ['o', 'D', 's', '^']
for i, d in enumerate(data):
    y = np.random.normal(positions[i], 0.08, size=len(d))
    ax2.scatter(y, d, alpha=0.9, marker=markers[i], color=colors[i+4], linewidths=0, zorder=1, s=300)

ax2.hlines(max(np.array(data[0])), 0.9, 1.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[0])), 0.9, 1.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[1])), 1.9, 2.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[1])), 1.9, 2.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines([2], 0, 5, color='r', linestyle='--', lw=2)


# set style for the axes
labels = ['BIC1+BIC2', 'BIC3+BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()

data = [[1.166123779,1.299610895,0.985765125,1.204663212,1.844559585,0.953571429,1.281368821,1.648648649,1.190298507,0.926027397,0.953654189,1.5,1.926530612,1.08,1.31838565,1.501272265,1.759090909,1.300578035],
        [0.855421687,1.151943463,1.361313869,1.090566038,1.095785441,1.131578947,0.932539683,1.371681416,1.04047619,1.063888889,1.082589286,0.824833703,0.874720358,0.947236181,1.135135135,1.172523962,1.038410596,1.006688963],
        [0.884773663,1.006644518,1.351851852,0.993299832,1.080519481,1.216374269,0.936986301,1.32967033,1.17794971,1.203910615,1.034334764,0.932170543,1.232081911,1.123529412,1.118367347,1.150779896,1.058275058,1.059748428],
        [1.420289855,1.260504202,0.920265781,1.154566745,1.107255521,1.886178862,1.319852941,1.051282051,1.361516035,1.091954023,1.144086022,1.344023324,1.005263158,1.430406852,1.272030651,1.65129683,1.20532,1.319047619],
        ]

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.4', fontsize=48, fontweight='bold')
positions = [1, 2, 3, 4]

# bplot = ax2.boxplot(data, widths=0.8, positions=positions, showbox=False, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=False, showfliers=False, medianprops=dict(color='black', linewidth=2), boxprops=None)

# 设置箱子的颜色
colors = ['teal', 'chocolate', 'purple', 'red', 'crimson', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

markers = ['o', 'D', 's', '^']
for i, d in enumerate(data):
    y = np.random.normal(positions[i], 0.15, size=len(d))
    ax2.scatter(y, d, alpha=0.9, marker=markers[i], color=colors[i], linewidths=0, zorder=1, s=300)


ax2.vlines(0.6, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.4, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.6, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.4, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.6, ymin=0,  ymax=np.median(np.array(data[2])), linestyle='-', linewidth=2, color='black')
ax2.vlines(3.4, ymin=0,  ymax=np.median(np.array(data[2])), linestyle='-', linewidth=2, color='black')
ax2.vlines(3.6, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
ax2.vlines(4.4, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1, ymin=np.min(np.array(data[0])),  ymax=np.max(np.array(data[0])), linestyle='-', linewidth=1, color='black')
ax2.vlines(2, ymin=np.min(np.array(data[1])),  ymax=np.max(np.array(data[1])), linestyle='-', linewidth=1, color='black')
ax2.vlines(3, ymin=np.min(np.array(data[2])),  ymax=np.max(np.array(data[2])), linestyle='-', linewidth=1, color='black')
ax2.vlines(4, ymin=np.min(np.array(data[3])),  ymax=np.max(np.array(data[3])), linestyle='-', linewidth=1, color='black')
ax2.hlines(max(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[2])), 2.8, 3.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[2])), 2.8, 3.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[3])), 3.8, 4.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[3])), 3.8, 4.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[0])), 0.6, 1.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[1])), 1.6, 2.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[2])), 2.6, 3.4, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[3])), 3.6, 4.4, linestyle='-', linewidth=2, color='black')
# ax2.axvline(1.4, ymin=0,  ymax=max(np.median(np.array(data[1])), np.median(np.array(data[0]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(2.4, ymin=0,  ymax=max(np.median(np.array(data[1])), np.median(np.array(data[2]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(3.5, ymin=0,  ymax=max(np.median(np.array(data[2])), np.median(np.array(data[3]))), linestyle='-', linewidth=2, color='black')
# ax2.axvline(4.5, ymin=0,  ymax=np.median(np.array(data[3])), linestyle='-', linewidth=2, color='black')
# bplot = axes.boxplot(data, widths=0.4, positions=positions, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=True, showfliers=False, showbox=True, medianprops=dict(color='black', linewidth=2), boxprops={'linewidth': 2, 'linestyle': '-'})
ax2.set_ylim(0, 2.5)
# parts = ax2.violinplot(
#         data, showmeans=True, showmedians=False,
#         showextrema=True)
# i = 0
# for pc in parts['bodies']:
#     pc.set_facecolor('cyan')
#     pc.set_edgecolor('black')
#     i = i + 1
    # pc.set_alpha(0.6)

# quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
# whiskers = np.array([
#     adjacent_values(sorted_array, q1, q3)
#     for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
# whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

# inds = np.arange(1, len(medians) + 1)
# ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
# ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
# ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([1], 0, 5, color='r', linestyle='--', lw=2)

# set style for the axes
labels = ['BIC1', 'BIC2', 'BIC3', 'BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.4', fontsize=48, fontweight='bold')

positions = [1, 2]
data = [np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])]
# bplot = ax2.boxplot(data, widths=0.4, positions=positions, showbox=False, meanprops=dict(markerfacecolor='r', color='r', markersize = 16), patch_artist=False, capprops={'linewidth': 2}, showmeans=False, showfliers=False, medianprops=dict(color='black', linewidth=2), boxprops=None)

ax2.vlines(0.8, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.2, ymin=0,  ymax=np.median(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1.8, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2.2, ymin=0,  ymax=np.median(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.vlines(1, ymin=np.min(np.array(data[0])),  ymax=np.max(np.array(data[0])), linestyle='-', linewidth=2, color='black')
ax2.vlines(2, ymin=np.min(np.array(data[1])),  ymax=np.max(np.array(data[1])), linestyle='-', linewidth=2, color='black')
ax2.set_ylim(0, 4)
# 设置箱子的颜色
colors = ['teal', 'chocolate', 'purple', 'red', 'crimson', 'deepskyblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

markers = ['o', 'D', 's', '^']
for i, d in enumerate(data):
    y = np.random.normal(positions[i], 0.08, size=len(d))
    ax2.scatter(y, d, alpha=0.9, marker=markers[i], color=colors[i+4], linewidths=0, zorder=1, s=300)

ax2.hlines(max(np.array(data[0])), 0.9, 1.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[0])), 0.9, 1.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(max(np.array(data[1])), 1.9, 2.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(min(np.array(data[1])), 1.9, 2.1, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[0])), 0.8, 1.2, linestyle='-', linewidth=2, color='black')
ax2.hlines(np.median(np.array(data[1])), 1.8, 2.2, linestyle='-', linewidth=2, color='black')
ax2.hlines([2], 0, 5, color='r', linestyle='--', lw=2)


# set style for the axes
labels = ['BIC1+BIC2', 'BIC3+BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()


#### variation
# 设置全局字体为 Times New Roman，加粗，20 号字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 20

dis_data = [[8.2232,10.4427,10.5822,8.8245,7.9977,10.3637,8.4512,11.5014,9.5726,12.3052,11.0142,7.4281,5.4279,7.3477,10.5449,7.5126,5.4685,11.0724,
             6.7707,3.8667,8.195,8.2197,5.926,6.2079,7.0794,11.9289,10.3183,13.1783,7.9475,9.5039,7.9765,9.2976,7.3078,7.8051,8.2726,7.9143,
             10.4236,6.0241,7.6121,8.3671,5.0166,1.9104,8.4137,10.7386,8.1573,9.8856,9.0507,9.1563,8.4219,5.903,7.4539,6.2764,11.7491,9.7753,
             4.3075,7.6627,10.8703,5.6315,8.1949,9.0354,9.3396,5.4985,6.2487,11.1497,7.9616,4.4123,7.1154,9.0016,14.165,9.357,5.7021,9.025,
             7.8839,8.8771,9.564,7.6337,7.0436,6.3969,7.3831,9.23,8.6555,11.0551,8.2281,10.7313,7.7266,7.7915,9.6176,9.4356,5.2272,8.9622,
             5.8497,3.9007,8.7343,6.1751,4.1231,7.4142,4.5782,6.1172,7.9533,11.6751,8.5355,6.4158,6.3453,7.7081,5.8047,5.6087,8.369,7.7713,
             7.7134,5.39,8.2585,6.8176,5.3219,6.5324,8.139,6.7682,5.3046,11.7542,8.0469,6.8253,4.799,5.587,10.0908,6.7363,10.0728,6.6641,
             7.0056,6.2546,8.1241,7.8125,4.2322,4.5691,8.5874,6.5652,6.3644,8.618,9.0375,7.9095,7.0026,8.0186,6.1675,7.2978,6.2485,7.9541],
            
             [8.2232,8.8739,12.6644,8.2983,7.9977,10.3639,6.3613,11.5014,10.3183,13.1783,11.0142,5.6715,8.1711,7.7266,10.5449,4.7248,5.4832,12.3748,
             6.7707,3.8666,8.1950,8.7818,5.3916,6.2079,7.6621,9.5299,6.7032,11.8585,7.9475,10.9676,7.9765,9.3064,7.3078,7.8051,8.2726,6.5291,
             10.5793,6.0241,8.0597,8.3671,5.0881,7.9613,10.1725,10.7386,7.3065,9.7603,9.0507,9.1563,9.2818,5.9030,7.4539,6.2764,6.5476,10.5962,
             6.1847,7.6627,11.0092,8.3582,6.9488,5.7045,5.4029,5.4985,6.2487,11.1497,7.3756,5.5941,4.6909,9.0016,10.3178,9.3570,11.7491,6.2871,
             7.8839,8.8771,9.564,7.6337,7.0436,6.3969,7.3831,9.2300,8.6555,11.0551,8.2281,10.7313,7.7266,7.7915,9.6176,9.4356,5.2272,8.9622,
             5.8497,3.9007,8.7343,6.1751,4.1231,7.4142,4.5782,6.1172,7.9533,11.6751,8.5355,6.4158,6.3453,7.7081,5.8047,5.6087,8.369,7.7713,
             7.7134,5.3900,8.2585,6.8176,5.3219,6.5324,8.1390,6.7682,5.3046,11.7542,8.0469,6.8253,4.7990,5.5870,10.0908,6.7363,10.0728,6.6641,
             7.0056,6.2546,8.1241,7.8125,4.2322,4.5691,8.5874,6.5652,6.3644,8.6180,9.0375,7.9095,7.0026,8.0186,6.1675,7.2978,6.2485,7.9541],
            
            
             [6.4342,9.0378,12.6644,7.4612,7.211,8.6778,6.3969,11.5014,11.0776,12.7362,11.0142,7.0263,6.8284,7.7266,10.5449,6.0847,6.4328,12.3748,
             7.8589,3.8667,8.195,8.7818,5.3916,6.2079,7.6621,10.5799,8.2202,12.1537,7.9475,10.9676,7.9765,9.3064,7.7282,7.8051,8.2726,6.3093,
             10.5794,6.0241,8.0597,8.3671,5.0881,7.6934,9.6644,9.4526,7.3065,9.7603,9.0507,9.1563,8.9349,5.903,7.4539,6.7634,11.8341,10.7131,
             6.1848,7.6627,11.0093,8.8419,6.9488,6.5215,5.4029,5.4985,6.2487,11.1497,9.4043,5.5941,6.4031,9.0016,12.165,9.5979,6.0584,6.638,
             7.8839,8.8771,9.564,7.6337,7.0436,6.3969,7.3831,9.23,8.6555,11.0551,8.2281,10.7313,7.7266,7.7915,9.6176,9.4356,5.2272,8.9622,
             5.8497,3.9007,8.7343,6.1751,4.1231,7.4142,4.5782,6.1172,7.9533,11.6751,8.5355,6.4158,6.3453,7.7081,5.8047,5.6087,8.369,7.7713,
             7.7134,5.3900,8.2585,6.8176,5.3219,6.5324,8.1390,6.7682,5.3046,11.7542,8.0469,6.8253,4.7990,5.5870,10.0908,6.7363,10.0728,6.6641,
             7.0056,6.2546,8.1241,7.8125,4.2322,4.5691,8.5874,6.5652,6.3644,8.6180,9.0375,7.9095,7.0026,8.0186,6.1675,7.2978,6.2485,7.9541],
            
            
             [6.4342,9.0378,12.6644,8.2983,7.211,8.6778,6.8457,11.5014,11.62,12.7362,11.0142,7.4281,6.8284,7.7266,10.5449,6.0847,5.4832,11.0724,
             7.8589,3.8667,8.7808,8.7818,5.3916,6.2079,7.6621,10.5799,7.4381,12.1537,7.9475,10.8172,7.9765,9.3064,7.3078,7.8051,8.2726,7.7757,
             10.5778,6.0241,7.6121,10.1081,5.0881,7.6934,9.6644,10.9497,8.1574,9.7603,9.0507,9.1563,9.2818,5.903,7.4539,6.7634,11.8996,10.7131,
             4.7418,7.6627,9.5938,7.8931,6.9488,6.5215,5.4029,5.4985,6.2487,11.1497,9.4043,5.5941,4.6909,9.0016,10.3178,9.5979,6.0584,6.638,
             7.8839,8.8771,9.564,7.6337,7.0436,6.3969,7.3831,9.2300,8.6555,11.0551,8.2281,10.7313,7.7266,7.7915,9.6176,9.4356,5.2272,8.9622,
             5.8497,3.9007,8.7343,6.1751,4.1231,7.4142,4.5782,6.1172,7.9533,11.6751,8.5355,6.4158,6.3453,7.7081,5.8047,5.6087,8.369,7.7713,
             7.7134,5.3900,8.2585,6.8176,5.3219,6.5324,8.1390,6.7682,5.3046,11.7542,8.0469,6.8253,4.7990,5.5870,10.0908,6.7363,10.0728,6.6641,
             7.0056,6.2546,8.1241,7.8125,4.2322,4.5691,8.5874,6.5652,6.3644,8.6180,9.0375,7.9095,7.0026,8.0186,6.1675,7.2978,6.2485,7.9541]            
            ]


bic_data = [[1.184713376,1.5,0.935897436,1.072072072,1.618181818,1.121621622,1.192771084,1.814814815,1.261538462,1.320987654,1.054054054,1.691176471,2.016666667,1.366972477,1.869565217,1.469387755,1.899082569,1.218181818,
             1.108695652,1.102564103,1.279411765,1.013422819,1.068493151,1.319148936,1.345454545,1.551724138,1.685714286,1.185185185,1.268518519,0.908256881,1.183486239,1.108695652,1.322580645,1.478873239,1.284883721,0.934210526,
             1.779661017,1.454545455,1.084507042,1.29245283,1.430379747,1.970588235,1.263888889,1.135135135,1.318681319,1.3625,1.11965812,1.311111111,1.1,1.764150943,1.701754386,1.875,1.30952381,1.510638298,
             1.079136691,1.2,1.490196078,1.093333333,1.145833333,1.055555556,1.24691358,1.384615385,1.366071429,1.296703297,1.098214286,0.912,1.376,1.403100775,1.576923077,1.295454545,1.175879397,1.051948052],
          
          
            [1.120743034,1.383458647,1.006756757,1.177339901,1.989130435,0.909722222,1.375,2,1.303278689,1.185628743,1.10701107,1.510638298,1.9140625,1.28959276,1.351851852,1.699421965,1.806167401,1.112244898,
             1.227848101,1.163120567,1.401360544,1.049469965,1.028368794,1.169082126,0.992063492,1.344262295,1.113207547,1.139534884,1.195121951,0.821256039,0.803652968,1.015789474,1.221311475,1.340136054,0.981382979,0.845679012,
             0.917322835,1.048275862,1.573033708,1,1.087179487,1.276836158,1.045714286,1.384615385,1.340080972,1.179775281,0.99595,0.939271255,1.296296296,1.251937984,1.170940171,1.19047619,1.026190476,1.044871795,
             1.516981132,1.427272727,0.934640523,1.202764977,1.238993711,1.819672131,1.152777778,1.05085,1.342696629,1.14619883,1.285046729,1.311111111,0.943127962,1.657276995,1.472,1.802325581,1.25228,1.281818182],
           
            [1.137931034,1.425414365,1.042056075,1.252631579,2.067669173,0.960591133,1.865248227,1.583333333,1.096153846,1.023255814,1.037688442,1.455399061,1.850515464,1.119241192,1.375722543,1.537162162,1.766153846,1.312977099,
             0.944680851,1.098214286,1.4375,1.063414634,1.067357513,1.096774194,1.070351759,1.337078652,1.19269103,1.118081181,1.086826347,1.098101266,1.113207547,0.996563574,1.173684211,1.225108225,1.058303887,1.026200873,
             1.504975124,1.329479769,0.924107143,1.186335404,1.080321285,2.093023256,1.25,1.23943662,1.414728682,1.101886792,1.161849711,1.347826087,0.982817869,1.520710059,1.369791667,1.73151751,1.16032,1.214285714,
             1.041775457,0.981818182,1.456953642,0.941309255,1.11827957,1.212927757,0.974452555,1.301369863,1.242819843,1.209125475,1.024861878,0.989247312,1.365384615,1.194300518,1.201086957,1.141230068,1.066985646,1.004237288],

            [1.166123779,1.299610895,0.985765125,1.204663212,1.844559585,0.953571429,1.281368821,1.648648649,1.190298507,0.926027397,0.953654189,1.5,1.926530612,1.08,1.31838565,1.501272265,1.759090909,1.300578035,
             0.855421687,1.151943463,1.361313869,1.090566038,1.095785441,1.131578947,0.932539683,1.371681416,1.04047619,1.063888889,1.082589286,0.824833703,0.874720358,0.947236181,1.135135135,1.172523962,1.038410596,1.006688963,
             1.420289855,1.260504202,0.920265781,1.154566745,1.107255521,1.886178862,1.319852941,1.051282051,1.361516035,1.091954023,1.144086022,1.344023324,1.005263158,1.430406852,1.272030651,1.65129683,1.20532,1.319047619,
             0.884773663,1.006644518,1.351851852,0.993299832,1.080519481,1.216374269,0.936986301,1.32967033,1.17794971,1.203910615,1.034334764,0.932170543,1.232081911,1.123529412,1.118367347,1.150779896,1.058275058,1.059748428]
           ]




x = np.array([1,2,3,4])

# 创建子图
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 6))
axes[0].plot([1,2,3,4], [np.average(dis_data[0]), np.average(dis_data[1]), np.average(dis_data[2]), np.average(dis_data[3])], 'bo-', markersize=10, linewidth=2)
axes[0].errorbar([1,2,3,4], [np.average(dis_data[0]), np.average(dis_data[1]), np.average(dis_data[2]), np.average(dis_data[3])], yerr=[np.std(dis_data[0]),np.std(dis_data[1]),np.std(dis_data[2]), np.std(dis_data[3])], fmt='o', color='blue', ecolor='red', linewidth=2, capsize=10)
axes[0].set_xticks([1,2,3,4], labels=['$\\varepsilon_r$ = 0.1', '$\\varepsilon_r$ = 0.2', '$\\varepsilon_r$ = 0.3', '$\\varepsilon_r$ = 0.4'])
axes[0].set_ylim(5, 12)
# axes[0].set_title('General distance margin with different $\\varepsilon_r$', fontsize=32, fontweight='bold')

axes[1].plot([1,2,3,4], [np.average(bic_data[0]), np.average(bic_data[1]), np.average(bic_data[2]), np.average(bic_data[3])], 'bo-', markersize=10, linewidth=2)
axes[1].errorbar([1,2,3,4], [np.average(bic_data[0]), np.average(bic_data[1]), np.average(bic_data[2]), np.average(bic_data[3])], yerr=[np.std(bic_data[0]),np.std(bic_data[1]),np.std(bic_data[2]), np.std(bic_data[3])], fmt='o', color='blue', ecolor='red', linewidth=2, capsize=10)
axes[1].set_xticks([1,2,3,4], labels=['$\\varepsilon_r$ = 0.1', '$\\varepsilon_r$ = 0.2', '$\\varepsilon_r$ = 0.3', '$\\varepsilon_r$ = 0.4'])
axes[1].set_ylim(0.4, 2)
# axes[1].set_title('General BIC with different $\\varepsilon_r$', fontsize=32, fontweight='bold')4
plt.show()


# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
# 设定数据
df = pd.DataFrame({
    'group': ['$\\varepsilon_r$ = 0.1', '$\\varepsilon_r$ = 0.2', '$\\varepsilon_r$ = 0.3', '$\\varepsilon_r$ = 0.4'],
    'case1': [np.average(np.array([dis_data[0][0], dis_data[0][18], dis_data[0][36], dis_data[0][54]])), np.average(np.array([dis_data[1][0], dis_data[1][18], dis_data[1][36], dis_data[1][54]])), np.average(np.array([dis_data[2][0], dis_data[2][18], dis_data[2][36], dis_data[2][54]])), np.average(np.array([dis_data[3][0], dis_data[3][18], dis_data[3][36], dis_data[3][54]]))],
    'case2': [np.average(np.array([dis_data[0][1], dis_data[0][19], dis_data[0][37], dis_data[0][55]])), np.average(np.array([dis_data[1][1], dis_data[1][19], dis_data[1][37], dis_data[1][55]])), np.average(np.array([dis_data[2][1], dis_data[2][19], dis_data[2][37], dis_data[2][55]])), np.average(np.array([dis_data[3][1], dis_data[3][19], dis_data[3][37], dis_data[3][55]]))],
    'case3': [np.average(np.array([dis_data[0][2], dis_data[0][20], dis_data[0][38], dis_data[0][56]])), np.average(np.array([dis_data[1][2], dis_data[1][20], dis_data[1][38], dis_data[1][56]])), np.average(np.array([dis_data[2][2], dis_data[2][20], dis_data[2][38], dis_data[2][56]])), np.average(np.array([dis_data[3][2], dis_data[3][20], dis_data[3][38], dis_data[3][56]]))],
    'case4': [np.average(np.array([dis_data[0][3], dis_data[0][21], dis_data[0][39], dis_data[0][57]])), np.average(np.array([dis_data[1][3], dis_data[1][21], dis_data[1][39], dis_data[1][57]])), np.average(np.array([dis_data[2][3], dis_data[2][21], dis_data[2][39], dis_data[2][57]])), np.average(np.array([dis_data[3][3], dis_data[3][21], dis_data[3][39], dis_data[3][57]]))],
    'case5': [np.average(np.array([dis_data[0][4], dis_data[0][22], dis_data[0][40], dis_data[0][58]])), np.average(np.array([dis_data[1][4], dis_data[1][22], dis_data[1][40], dis_data[1][58]])), np.average(np.array([dis_data[2][4], dis_data[2][22], dis_data[2][40], dis_data[2][58]])), np.average(np.array([dis_data[3][4], dis_data[3][22], dis_data[3][40], dis_data[3][58]]))],
    'case6': [np.average(np.array([dis_data[0][5], dis_data[0][23], dis_data[0][41], dis_data[0][59]])), np.average(np.array([dis_data[1][5], dis_data[1][23], dis_data[1][41], dis_data[1][59]])), np.average(np.array([dis_data[2][5], dis_data[2][23], dis_data[2][41], dis_data[2][59]])), np.average(np.array([dis_data[3][5], dis_data[3][23], dis_data[3][41], dis_data[3][59]]))],
    'case7': [np.average(np.array([dis_data[0][6], dis_data[0][24], dis_data[0][42], dis_data[0][60]])), np.average(np.array([dis_data[1][6], dis_data[1][24], dis_data[1][42], dis_data[1][60]])), np.average(np.array([dis_data[2][6], dis_data[2][24], dis_data[2][42], dis_data[2][60]])), np.average(np.array([dis_data[3][6], dis_data[3][24], dis_data[3][42], dis_data[3][60]]))],
    'case8': [np.average(np.array([dis_data[0][7], dis_data[0][25], dis_data[0][43], dis_data[0][61]])), np.average(np.array([dis_data[1][7], dis_data[1][25], dis_data[1][43], dis_data[1][61]])), np.average(np.array([dis_data[2][7], dis_data[2][25], dis_data[2][43], dis_data[2][61]])), np.average(np.array([dis_data[3][7], dis_data[3][25], dis_data[3][43], dis_data[3][61]]))],
    'case9': [np.average(np.array([dis_data[0][8], dis_data[0][26], dis_data[0][44], dis_data[0][62]])), np.average(np.array([dis_data[1][8], dis_data[1][26], dis_data[1][44], dis_data[1][62]])), np.average(np.array([dis_data[2][8], dis_data[2][26], dis_data[2][44], dis_data[2][62]])), np.average(np.array([dis_data[3][8], dis_data[3][26], dis_data[3][44], dis_data[3][62]]))],
    'case10': [np.average(np.array([dis_data[0][9], dis_data[0][27], dis_data[0][45], dis_data[0][63]])), np.average(np.array([dis_data[1][9], dis_data[1][27], dis_data[1][45], dis_data[1][63]])), np.average(np.array([dis_data[2][9], dis_data[2][27], dis_data[2][45], dis_data[2][63]])), np.average(np.array([dis_data[3][9], dis_data[3][27], dis_data[3][45], dis_data[3][63]]))],
    'case11': [np.average(np.array([dis_data[0][10], dis_data[0][28], dis_data[0][46], dis_data[0][64]])), np.average(np.array([dis_data[1][10], dis_data[1][28], dis_data[1][46], dis_data[1][64]])), np.average(np.array([dis_data[2][10], dis_data[2][28], dis_data[2][46], dis_data[2][64]])), np.average(np.array([dis_data[3][10], dis_data[3][28], dis_data[3][46], dis_data[3][64]]))],
    'case12': [np.average(np.array([dis_data[0][11], dis_data[0][29], dis_data[0][47], dis_data[0][65]])), np.average(np.array([dis_data[1][11], dis_data[1][29], dis_data[1][47], dis_data[1][65]])), np.average(np.array([dis_data[2][11], dis_data[2][29], dis_data[2][47], dis_data[2][65]])), np.average(np.array([dis_data[3][11], dis_data[3][29], dis_data[3][47], dis_data[3][65]]))],
    'case13': [np.average(np.array([dis_data[0][12], dis_data[0][30], dis_data[0][48], dis_data[0][66]])), np.average(np.array([dis_data[1][12], dis_data[1][30], dis_data[1][48], dis_data[1][66]])), np.average(np.array([dis_data[2][12], dis_data[2][30], dis_data[2][48], dis_data[2][66]])), np.average(np.array([dis_data[3][12], dis_data[3][30], dis_data[3][48], dis_data[3][66]]))],
    'case14': [np.average(np.array([dis_data[0][13], dis_data[0][31], dis_data[0][49], dis_data[0][67]])), np.average(np.array([dis_data[1][13], dis_data[1][31], dis_data[1][49], dis_data[1][67]])), np.average(np.array([dis_data[2][13], dis_data[2][31], dis_data[2][49], dis_data[2][67]])), np.average(np.array([dis_data[3][13], dis_data[3][31], dis_data[3][49], dis_data[3][67]]))],
    'case15': [np.average(np.array([dis_data[0][14], dis_data[0][32], dis_data[0][50], dis_data[0][68]])), np.average(np.array([dis_data[1][14], dis_data[1][32], dis_data[1][50], dis_data[1][68]])), np.average(np.array([dis_data[2][14], dis_data[2][32], dis_data[2][50], dis_data[2][68]])), np.average(np.array([dis_data[3][14], dis_data[3][32], dis_data[3][50], dis_data[3][68]]))],
    'case16': [np.average(np.array([dis_data[0][15], dis_data[0][33], dis_data[0][51], dis_data[0][69]])), np.average(np.array([dis_data[1][15], dis_data[1][33], dis_data[1][51], dis_data[1][69]])), np.average(np.array([dis_data[2][15], dis_data[2][33], dis_data[2][51], dis_data[2][69]])), np.average(np.array([dis_data[3][15], dis_data[3][33], dis_data[3][51], dis_data[3][69]]))],
    'case17': [np.average(np.array([dis_data[0][16], dis_data[0][34], dis_data[0][52], dis_data[0][71]])), np.average(np.array([dis_data[1][16], dis_data[1][34], dis_data[1][52], dis_data[1][70]])), np.average(np.array([dis_data[2][16], dis_data[2][34], dis_data[2][52], dis_data[2][70]])), np.average(np.array([dis_data[3][16], dis_data[3][34], dis_data[3][52], dis_data[3][70]]))],
    'case18': [np.average(np.array([dis_data[0][17], dis_data[0][35], dis_data[0][53], dis_data[0][71]])), np.average(np.array([dis_data[1][17], dis_data[1][35], dis_data[1][53], dis_data[1][71]])), np.average(np.array([dis_data[2][17], dis_data[2][35], dis_data[2][53], dis_data[2][71]])), np.average(np.array([dis_data[3][17], dis_data[3][35], dis_data[3][53], dis_data[3][71]]))],

})

# number of variable
# 变量类别
categories = list(df)[1:]
# 变量类别个数
N = len(categories)

# 设置每个点的角度值
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
# 初始化极坐标网格
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels labels yet
# 设置x轴的标签
plt.xticks(angles[:-1], categories, color='grey', fontsize=20, fontweight='bold')


ax.set_rlabel_position(0)
# 设置y轴的标签
plt.yticks([3, 6, 9], ["3", "6", "9"], color="grey",  fontsize=20, fontweight='bold')
plt.ylim(0, 12)

# Plot data
# 画图
# plot the first line of the data frame.
# 绘制数据的第一行
values = df.loc[0].drop('group').values.flatten().tolist()
# 将第一个值放到最后，以封闭图形
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label='$\\varepsilon_r$ = 0.1')
ax.fill(angles, values, 'purple', alpha=0.1)

values = df.loc[1].drop('group').values.flatten().tolist()
# 将第一个值放到最后，以封闭图形
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label='$\\varepsilon_r$ = 0.2')
ax.fill(angles, values, 'b', alpha=0.1)

values = df.loc[2].drop('group').values.flatten().tolist()
# 将第一个值放到最后，以封闭图形
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label='$\\varepsilon_r$ = 0.3')
ax.fill(angles, values, 'g', alpha=0.1)

values = df.loc[3].drop('group').values.flatten().tolist()
# 将第一个值放到最后，以封闭图形
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label='$\\varepsilon_r$ = 0.4')
ax.fill(angles, values, 'saddlebrown', alpha=0.1)
plt.legend(loc='upper right')
plt.show()

df = pd.DataFrame({
    'group': ['$\\varepsilon_r$ = 0.1', '$\\varepsilon_r$ = 0.2', '$\\varepsilon_r$ = 0.3', '$\\varepsilon_r$ = 0.4'],
    'case1': [np.average(np.array([bic_data[0][0], bic_data[0][18], bic_data[0][36], bic_data[0][54]])), np.average(np.array([bic_data[1][0], bic_data[1][18], bic_data[1][36], bic_data[1][54]])), np.average(np.array([bic_data[2][0], bic_data[2][18], bic_data[2][36], bic_data[2][54]])), np.average(np.array([bic_data[3][0], bic_data[3][18], bic_data[3][36], bic_data[3][54]]))],
    'case2': [np.average(np.array([bic_data[0][1], bic_data[0][19], bic_data[0][37], bic_data[0][55]])), np.average(np.array([bic_data[1][1], bic_data[1][19], bic_data[1][37], bic_data[1][55]])), np.average(np.array([bic_data[2][1], bic_data[2][19], bic_data[2][37], bic_data[2][55]])), np.average(np.array([bic_data[3][1], bic_data[3][19], bic_data[3][37], bic_data[3][55]]))],
    'case3': [np.average(np.array([bic_data[0][2], bic_data[0][20], bic_data[0][38], bic_data[0][56]])), np.average(np.array([bic_data[1][2], bic_data[1][20], bic_data[1][38], bic_data[1][56]])), np.average(np.array([bic_data[2][2], bic_data[2][20], bic_data[2][38], bic_data[2][56]])), np.average(np.array([bic_data[3][2], bic_data[3][20], bic_data[3][38], bic_data[3][56]]))],
    'case4': [np.average(np.array([bic_data[0][3], bic_data[0][21], bic_data[0][39], bic_data[0][57]])), np.average(np.array([bic_data[1][3], bic_data[1][21], bic_data[1][39], bic_data[1][57]])), np.average(np.array([bic_data[2][3], bic_data[2][21], bic_data[2][39], bic_data[2][57]])), np.average(np.array([bic_data[3][3], bic_data[3][21], bic_data[3][39], bic_data[3][57]]))],
    'case5': [np.average(np.array([bic_data[0][4], bic_data[0][22], bic_data[0][40], bic_data[0][58]])), np.average(np.array([bic_data[1][4], bic_data[1][22], bic_data[1][40], bic_data[1][58]])), np.average(np.array([bic_data[2][4], bic_data[2][22], bic_data[2][40], bic_data[2][58]])), np.average(np.array([bic_data[3][4], bic_data[3][22], bic_data[3][40], bic_data[3][58]]))],
    'case6': [np.average(np.array([bic_data[0][5], bic_data[0][23], bic_data[0][41], bic_data[0][59]])), np.average(np.array([bic_data[1][5], bic_data[1][23], bic_data[1][41], bic_data[1][59]])), np.average(np.array([bic_data[2][5], bic_data[2][23], bic_data[2][41], bic_data[2][59]])), np.average(np.array([bic_data[3][5], bic_data[3][23], bic_data[3][41], bic_data[3][59]]))],
    'case7': [np.average(np.array([bic_data[0][6], bic_data[0][24], bic_data[0][42], bic_data[0][60]])), np.average(np.array([bic_data[1][6], bic_data[1][24], bic_data[1][42], bic_data[1][60]])), np.average(np.array([bic_data[2][6], bic_data[2][24], bic_data[2][42], bic_data[2][60]])), np.average(np.array([bic_data[3][6], bic_data[3][24], bic_data[3][42], bic_data[3][60]]))],
    'case8': [np.average(np.array([bic_data[0][7], bic_data[0][25], bic_data[0][43], bic_data[0][61]])), np.average(np.array([bic_data[1][7], bic_data[1][25], bic_data[1][43], bic_data[1][61]])), np.average(np.array([bic_data[2][7], bic_data[2][25], bic_data[2][43], bic_data[2][61]])), np.average(np.array([bic_data[3][7], bic_data[3][25], bic_data[3][43], bic_data[3][61]]))],
    'case9': [np.average(np.array([bic_data[0][8], bic_data[0][26], bic_data[0][44], bic_data[0][62]])), np.average(np.array([bic_data[1][8], bic_data[1][26], bic_data[1][44], bic_data[1][62]])), np.average(np.array([bic_data[2][8], bic_data[2][26], bic_data[2][44], bic_data[2][62]])), np.average(np.array([bic_data[3][8], bic_data[3][26], bic_data[3][44], bic_data[3][62]]))],
    'case10': [np.average(np.array([bic_data[0][9], bic_data[0][27], bic_data[0][45], bic_data[0][63]])), np.average(np.array([bic_data[1][9], bic_data[1][27], bic_data[1][45], bic_data[1][63]])), np.average(np.array([bic_data[2][9], bic_data[2][27], bic_data[2][45], bic_data[2][63]])), np.average(np.array([bic_data[3][9], bic_data[3][27], bic_data[3][45], bic_data[3][63]]))],
    'case11': [np.average(np.array([bic_data[0][10], bic_data[0][28], bic_data[0][46], bic_data[0][64]])), np.average(np.array([bic_data[1][10], bic_data[1][28], bic_data[1][46], bic_data[1][64]])), np.average(np.array([bic_data[2][10], bic_data[2][28], bic_data[2][46], bic_data[2][64]])), np.average(np.array([bic_data[3][10], bic_data[3][28], bic_data[3][46], bic_data[3][64]]))],
    'case12': [np.average(np.array([bic_data[0][11], bic_data[0][29], bic_data[0][47], bic_data[0][65]])), np.average(np.array([bic_data[1][11], bic_data[1][29], bic_data[1][47], bic_data[1][65]])), np.average(np.array([bic_data[2][11], bic_data[2][29], bic_data[2][47], bic_data[2][65]])), np.average(np.array([bic_data[3][11], bic_data[3][29], bic_data[3][47], bic_data[3][65]]))],
    'case13': [np.average(np.array([bic_data[0][12], bic_data[0][30], bic_data[0][48], bic_data[0][66]])), np.average(np.array([bic_data[1][12], bic_data[1][30], bic_data[1][48], bic_data[1][66]])), np.average(np.array([bic_data[2][12], bic_data[2][30], bic_data[2][48], bic_data[2][66]])), np.average(np.array([bic_data[3][12], bic_data[3][30], bic_data[3][48], bic_data[3][66]]))],
    'case14': [np.average(np.array([bic_data[0][13], bic_data[0][31], bic_data[0][49], bic_data[0][67]])), np.average(np.array([bic_data[1][13], bic_data[1][31], bic_data[1][49], bic_data[1][67]])), np.average(np.array([bic_data[2][13], bic_data[2][31], bic_data[2][49], bic_data[2][67]])), np.average(np.array([bic_data[3][13], bic_data[3][31], bic_data[3][49], bic_data[3][67]]))],
    'case15': [np.average(np.array([bic_data[0][14], bic_data[0][32], bic_data[0][50], bic_data[0][68]])), np.average(np.array([bic_data[1][14], bic_data[1][32], bic_data[1][50], bic_data[1][68]])), np.average(np.array([bic_data[2][14], bic_data[2][32], bic_data[2][50], bic_data[2][68]])), np.average(np.array([bic_data[3][14], bic_data[3][32], bic_data[3][50], bic_data[3][68]]))],
    'case16': [np.average(np.array([bic_data[0][15], bic_data[0][33], bic_data[0][51], bic_data[0][69]])), np.average(np.array([bic_data[1][15], bic_data[1][33], bic_data[1][51], bic_data[1][69]])), np.average(np.array([bic_data[2][15], bic_data[2][33], bic_data[2][51], bic_data[2][69]])), np.average(np.array([bic_data[3][15], bic_data[3][33], bic_data[3][51], bic_data[3][69]]))],
    'case17': [np.average(np.array([bic_data[0][16], bic_data[0][34], bic_data[0][52], bic_data[0][71]])), np.average(np.array([bic_data[1][16], bic_data[1][34], bic_data[1][52], bic_data[1][70]])), np.average(np.array([bic_data[2][16], bic_data[2][34], bic_data[2][52], bic_data[2][70]])), np.average(np.array([bic_data[3][16], bic_data[3][34], bic_data[3][52], bic_data[3][70]]))],
    'case18': [np.average(np.array([bic_data[0][17], bic_data[0][35], bic_data[0][53], bic_data[0][71]])), np.average(np.array([bic_data[1][17], bic_data[1][35], bic_data[1][53], bic_data[1][71]])), np.average(np.array([bic_data[2][17], bic_data[2][35], bic_data[2][53], bic_data[2][71]])), np.average(np.array([bic_data[3][17], bic_data[3][35], bic_data[3][53], bic_data[3][71]]))],

})

# number of variable
# 变量类别
categories = list(df)[1:]
# 变量类别个数
N = len(categories)

# 设置每个点的角度值
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
# 初始化极坐标网格
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels labels yet
# 设置x轴的标签
plt.xticks(angles[:-1], categories, color='grey',fontsize=20, fontweight='bold')


ax.set_rlabel_position(0)
# 设置y轴的标签
plt.yticks([0, 1, 2], ["0", "1", "2"], color="grey",  fontsize=20, fontweight='bold')
plt.ylim(0, 2)

# Plot data
# 画图
# plot the first line of the data frame.
# 绘制数据的第一行
values = df.loc[0].drop('group').values.flatten().tolist()
# 将第一个值放到最后，以封闭图形
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label='$\\varepsilon_r$ = 0.1')
ax.fill(angles, values, 'purple', alpha=0.1)

values = df.loc[1].drop('group').values.flatten().tolist()
# 将第一个值放到最后，以封闭图形
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label='$\\varepsilon_r$ = 0.2')
ax.fill(angles, values, 'b', alpha=0.1)

values = df.loc[2].drop('group').values.flatten().tolist()
# 将第一个值放到最后，以封闭图形
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label='$\\varepsilon_r$ = 0.3')
ax.fill(angles, values, 'g', alpha=0.1)

values = df.loc[3].drop('group').values.flatten().tolist()
# 将第一个值放到最后，以封闭图形
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label='$\\varepsilon_r$ = 0.4')
ax.fill(angles, values, 'saddlebrown', alpha=0.1)
# plt.legend(loc='upper right')
plt.show()




from scipy import stats

# 0.1
print(stats.ttest_ind(bic_data[0][0:18], bic_data[0][36:54]))
print(stats.ttest_ind(bic_data[0][18:36], bic_data[0][54:72]),'\n')
# 0.2
print(stats.ttest_ind(bic_data[1][0:18], bic_data[1][54:72]))
print(stats.ttest_ind(bic_data[1][18:36], bic_data[1][36:54]),'\n')
# 0.3
print(stats.ttest_ind(bic_data[2][0:18], bic_data[2][36:54]))
print(stats.ttest_ind(bic_data[2][18:36], bic_data[2][54:72]),'\n')
# 0.4
print(stats.ttest_ind(bic_data[3][0:18], bic_data[3][36:54]))
print(stats.ttest_ind(bic_data[3][18:36], bic_data[3][54:72]),'\n')

print('overall')
# 0.1
print(stats.ttest_ind(np.array(bic_data[0][0:18])+np.array(bic_data[0][18:36]), np.array(bic_data[0][36:54])+np.array(bic_data[0][54:72])))
# 0.2
print(stats.ttest_ind(np.array(bic_data[1][0:18])+np.array(bic_data[1][18:36]), np.array(bic_data[1][36:54])+np.array(bic_data[1][54:72])))
# 0.3
print(stats.ttest_ind(np.array(bic_data[2][0:18])+np.array(bic_data[2][18:36]), np.array(bic_data[2][36:54])+np.array(bic_data[2][54:72])))
# 0.4
print(stats.ttest_ind(np.array(bic_data[3][0:18])+np.array(bic_data[3][18:36]), np.array(bic_data[3][36:54])+np.array(bic_data[3][54:72])))






original_bic_data = [[[186,90,73,119,89,83,99,49,82,107,156,115,121,149,86,144,207,67],
                      [102,86,87,151,78,124,74,45,177,96,137,99,129,102,82,105,221,71],
                      [210,80,77,137,113,67,91,42,120,109,131,118,99,187,97,165,220,71],
                      [150,84,76,164,110,95,101,36,153,118,123,114,172,181,82,171,234,81],
                      [157,60,78,111,55,74,83,27,65,81,148,68,60,109,46,98,109,55],
                      [92,78,68,149,73,94,55,29,105,81,108,109,109,92,62,71,172,76],
                      [118,55,71,106,79,34,72,37,91,80,117,90,90,106,57,88,168,47],
                      [139,70,51,150,96,90,81,26,112,91,112,125,125,129,52,132,199,77]
                     ],
                     [[362,184,149,239,183,131,176,94,159,198,300,213,245,285,146,294,410,109],
                      [194,164,206,297,145,242,125,82,236,196,245,170,176,193,149,197,369,137],
                      [233,152,140,298,212,226,183,72,331,210,246,232,175,323,137,325,431,163],
                      [402,157,143,261,197,111,166,62,239,196,275,236,199,353,184,310,412,141],
                      [323,133,148,203,92,144,128,47,122,167,271,141,128,221,108,173,227,98],
                      [158,141,147,283,141,207,126,61,212,172,205,207,219,190,122,147,376,162],
                      [254,145,89,298,195,177,175,52,247,178,247,247,135,258,117,273,420,156],
                      [265,110,153,217,159,61,144,59,178,171,214,180,211,213,125,172,329,110]
                     ],
                     [[528,258,223,357,275,195,263,133,228,264,413,310,359,413,238,455,574,172],
                      [222,246,299,436,206,340,213,119,359,303,363,347,354,290,223,283,599,235],
                      [605,230,207,382,269,180,280,88,365,292,402,341,286,514,263,445,579,204],
                      [399,216,220,417,312,319,267,95,476,318,371,368,284,461,221,501,669,237],
                      [464,181,214,285,133,203,141,84,208,258,398,213,194,369,173,296,325,131],
                      [235,224,208,410,193,310,199,89,301,271,334,316,318,291,190,231,566,229],
                      [402,173,224,322,249,86,224,71,258,265,346,253,291,338,192,257,499,168],
                      [383,220,151,443,279,263,274,73,383,263,362,372,208,386,184,439,627,236]
                     ],
                     [[716,334,277,465,356,267,337,183,319,338,535,438,472,540,294,590,774,225],
                      [284,326,373,578,286,473,235,155,437,383,485,372,391,377,294,367,784,301],
                      [784,300,277,493,351,232,359,123,467,380,532,461,382,668,332,573,816,277],
                      [430,303,292,593,416,416,342,121,609,431,482,481,361,573,274,664,908,337],
                      [614,257,281,386,193,280,263,111,268,365,561,292,245,500,223,393,440,173],
                      [332,283,274,530,261,418,252,113,420,360,448,451,447,398,259,313,755,299],
                      [552,238,301,427,317,123,272,117,343,348,465,343,380,467,261,347,677,210],
                      [486,301,216,597,385,342,365,91,517,358,466,516,293,510,245,577,858,318]
                     ]
                    ]

from scipy import stats

# 0.1
print('threshold = 0.1')
print(stats.ttest_ind(original_bic_data[0][0], original_bic_data[0][4]))
print(stats.ttest_ind(original_bic_data[0][1], original_bic_data[0][5]))
print(stats.ttest_ind(original_bic_data[0][2], original_bic_data[0][6]))
print(stats.ttest_ind(original_bic_data[0][3], original_bic_data[0][7]), '\n')
# 0.2
print('threshold = 0.2')
print(stats.ttest_ind(original_bic_data[1][0], original_bic_data[1][4]))
print(stats.ttest_ind(original_bic_data[1][1], original_bic_data[1][5]))
print(stats.ttest_ind(original_bic_data[1][2], original_bic_data[1][6]))
print(stats.ttest_ind(original_bic_data[1][3], original_bic_data[1][7]), '\n')
# 0.3
print('threshold = 0.3')
print(stats.ttest_ind(original_bic_data[2][0], original_bic_data[2][4]))
print(stats.ttest_ind(original_bic_data[2][1], original_bic_data[2][5]))
print(stats.ttest_ind(original_bic_data[2][2], original_bic_data[2][6]))
print(stats.ttest_ind(original_bic_data[2][3], original_bic_data[2][7]), '\n')
# 0.4
print('threshold = 0.4')
print(stats.ttest_ind(original_bic_data[3][0], original_bic_data[3][4]))
print(stats.ttest_ind(original_bic_data[3][1], original_bic_data[3][5]))
print(stats.ttest_ind(original_bic_data[3][2], original_bic_data[3][6]))
print(stats.ttest_ind(original_bic_data[3][3], original_bic_data[3][7]), '\n')


import matplotlib.pyplot as plt
x = np.array(range(1, 19))
width = 1  # the width of the bars
multiplier = 0
fig, ax = plt.subplots(layout='constrained')
plt.plot(x, original_bic_data[1][0], 'rs-', linewidth=4, markersize=16) #, label='Our Planning Results')
plt.plot(x, original_bic_data[1][4], 'go-', linewidth=4, markersize=16) #, label='Manual Planning Results')
# ax.fill_between(x,original_bic_data[1][0], 0, alpha=0.4, linewidth=0, color='sandybrown')
# ax.fill_between(x, original_bic_data[1][0], original_bic_data[1][4], alpha=0.4, linewidth=0, color='deepskyblue')
plt.xticks(range(1, 19, 2), fontsize=32, weight='bold')
plt.yticks(fontsize=32, weight='bold')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Original BIC', fontsize=32, fontweight='bold')
ax.set_xlabel('Case No.', fontsize=32, fontweight='bold')
#ax.set_title('(a) Comparison of the Distance Margin', fontsize=32, fontweight='bold')
# ax.set_xticks(x + width, species)
#ax.legend(loc='upper right', ncols=1, prop={"family": "Times New Roman", "size": 36, 'weight': "bold"})

plt.show()

fig, ax = plt.subplots(layout='constrained')
plt.plot(x, original_bic_data[1][1], 'rs-', linewidth=4, markersize=16) #, label='Our Planning Results')
plt.plot(x, original_bic_data[1][5], 'go-', linewidth=4, markersize=16) #, label='Manual Planning Results')
# ax.fill_between(x,original_bic_data[1][0], 0, alpha=0.4, linewidth=0, color='sandybrown')
# ax.fill_between(x, original_bic_data[1][0], original_bic_data[1][4], alpha=0.4, linewidth=0, color='deepskyblue')
plt.xticks(range(1, 19, 2), fontsize=32, weight='bold')
plt.yticks(fontsize=32, weight='bold')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Original BIC', fontsize=32, fontweight='bold')
ax.set_xlabel('Case No.', fontsize=32, fontweight='bold')
#ax.set_title('(a) Comparison of the Distance Margin', fontsize=32, fontweight='bold')
# ax.set_xticks(x + width, species)
#ax.legend(loc='upper right', ncols=1, prop={"family": "Times New Roman", "size": 36, 'weight': "bold"})

plt.show()

fig, ax = plt.subplots(layout='constrained')
plt.plot(x, original_bic_data[1][2], 'rs-', linewidth=4, markersize=16) #, label='Our Planning Results')
plt.plot(x, original_bic_data[1][6], 'go-', linewidth=4, markersize=16) #, label='Manual Planning Results')
# ax.fill_between(x,original_bic_data[1][0], 0, alpha=0.4, linewidth=0, color='sandybrown')
# ax.fill_between(x, original_bic_data[1][0], original_bic_data[1][4], alpha=0.4, linewidth=0, color='deepskyblue')
plt.xticks(range(1, 19, 2), fontsize=32, weight='bold')
plt.yticks(fontsize=32, weight='bold')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Original BIC', fontsize=32, fontweight='bold')
ax.set_xlabel('Case No.', fontsize=32, fontweight='bold')
#ax.set_title('(a) Comparison of the Distance Margin', fontsize=32, fontweight='bold')
# ax.set_xticks(x + width, species)
#ax.legend(loc='upper right', ncols=1, prop={"family": "Times New Roman", "size": 36, 'weight': "bold"})

plt.show()

fig, ax = plt.subplots(layout='constrained')
plt.plot(x, original_bic_data[1][3], 'rs-', linewidth=4, markersize=16) #, label='Our Planning Results')
plt.plot(x, original_bic_data[1][7], 'go-', linewidth=4, markersize=16) #, label='Manual Planning Results')
# ax.fill_between(x,original_bic_data[1][0], 0, alpha=0.4, linewidth=0, color='sandybrown')
# ax.fill_between(x, original_bic_data[1][0], original_bic_data[1][4], alpha=0.4, linewidth=0, color='deepskyblue')
plt.xticks(range(1, 19, 2), fontsize=32, weight='bold')
plt.yticks(fontsize=32, weight='bold')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Original BIC', fontsize=32, fontweight='bold')
ax.set_xlabel('Case No.', fontsize=32, fontweight='bold')
#ax.set_title('(a) Comparison of the Distance Margin', fontsize=32, fontweight='bold')
# ax.set_xticks(x + width, species)
#ax.legend(loc='upper right', ncols=1, prop={"family": "Times New Roman", "size": 36, 'weight': "bold"})

plt.show()