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

# # 绘制箱线图
# bplot = axes.boxplot([all_data[0], all_data[4], all_data[1], all_data[5], all_data[2], all_data[6], all_data[3], all_data[7]], widths=0.4, positions=positions, patch_artist=True, showmeans=True)

# # 设置箱子的颜色
# colors = ['pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

# axes.yaxis.grid(True) #在y轴上添加网格线
# axes.set_xticks([1.25, 2.75, 4.25, 5.75]) #指定x轴的轴刻度个数
# axes.set_xticklabels(['d$_{%s}$ (mm)'%1, 'd$_{%s}$ (mm)'%2, 'd$_{%s}$ (mm)'%3, 'd$_{%s}$ (mm)'%4], fontsize=40, fontweight='bold') #设置刻度标签

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

# # 绘制箱线图
# bplot = axes.boxplot([all_data[0], all_data[4], all_data[1], all_data[5], all_data[2], all_data[6], all_data[3], all_data[7]], widths=0.4, positions=positions, patch_artist=True, showmeans=True)

# # 设置箱子的颜色
# colors = ['pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

# axes.yaxis.grid(True) #在y轴上添加网格线
# axes.set_xticks([1.25, 2.75, 4.25, 5.75]) #指定x轴的轴刻度个数
# axes.set_xticklabels(['d$_{%s}$ (mm)'%1, 'd$_{%s}$ (mm)'%2, 'd$_{%s}$ (mm)'%3, 'd$_{%s}$ (mm)'%4], fontsize=40, fontweight='bold') #设置刻度标签

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

# # 绘制箱线图
# bplot = axes.boxplot([all_data[0], all_data[4], all_data[1], all_data[5], all_data[2], all_data[6], all_data[3], all_data[7]], widths=0.4, positions=positions, patch_artist=True, showmeans=True)

# # 设置箱子的颜色
# colors = ['pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

# axes.yaxis.grid(True) #在y轴上添加网格线
# axes.set_xticks([1.25, 2.75, 4.25, 5.75]) #指定x轴的轴刻度个数
# axes.set_xticklabels(['d$_{%s}$ (mm)'%1, 'd$_{%s}$ (mm)'%2, 'd$_{%s}$ (mm)'%3, 'd$_{%s}$ (mm)'%4], fontsize=40, fontweight='bold') #设置刻度标签

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

# # 绘制箱线图
# bplot = axes.boxplot([all_data[0], all_data[4], all_data[1], all_data[5], all_data[2], all_data[6], all_data[3], all_data[7]], widths=0.4, positions=positions, patch_artist=True, showmeans=True)

# # 设置箱子的颜色
# colors = ['pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue', 'pink', 'lightblue']# , 'lightgreen', '#E0B0FF']
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

# axes.yaxis.grid(True) #在y轴上添加网格线
# axes.set_xticks([1.25, 2.75, 4.25, 5.75]) #指定x轴的轴刻度个数
# axes.set_xticklabels(['d$_{%s}$ (mm)'%1, 'd$_{%s}$ (mm)'%2, 'd$_{%s}$ (mm)'%3, 'd$_{%s}$ (mm)'%4], fontsize=40, fontweight='bold') #设置刻度标签

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

ax2.set_title('$\\varepsilon_r$ = 0.1', fontsize=48, fontweight='bold')
parts = ax2.violinplot(
        data, showmeans=True, showmedians=False,
        showextrema=True)
i = 0
for pc in parts['bodies']:
    pc.set_facecolor('cyan')
    pc.set_edgecolor('black')
    i = i + 1
    # pc.set_alpha(0.6)

quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([1], 0, 5, color='r', linestyle='dotted', lw=2)

# set style for the axes
labels = ['BIC1', 'BIC2', 'BIC3', 'BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.1', fontsize=48, fontweight='bold')
parts = ax2.violinplot(
        [np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], showmeans=True, showmedians=False,
        showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('sienna')
    pc.set_edgecolor('black')
    # pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile([np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip([np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([2], 0, 3, color='r', linestyle='dotted', lw=2)

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
parts = ax2.violinplot(
        data, showmeans=True, showmedians=False,
        showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('cyan')
    pc.set_edgecolor('black')
    # pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([1], 0, 5, color='r', linestyle='dotted', lw=2)

# set style for the axes
labels = ['BIC1', 'BIC2', 'BIC3', 'BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.2', fontsize=48, fontweight='bold')
parts = ax2.violinplot(
        [np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], showmeans=True, showmedians=False,
        showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('sienna')
    pc.set_edgecolor('black')
    # pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile([np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip([np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([2], 0, 3, color='r', linestyle='dotted', lw=2)

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
parts = ax2.violinplot(
        data, showmeans=True, showmedians=False,
        showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('cyan')
    pc.set_edgecolor('black')
    # pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([1], 0, 5, color='r', linestyle='dotted', lw=2)

# set style for the axes
labels = ['BIC1', 'BIC2', 'BIC3', 'BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.3', fontsize=48, fontweight='bold')
parts = ax2.violinplot(
        [np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], showmeans=True, showmedians=False,
        showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('sienna')
    pc.set_edgecolor('black')
    # pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile([np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip([np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([2], 0, 3, color='r', linestyle='dotted', lw=2)

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
parts = ax2.violinplot(
        data, showmeans=True, showmedians=False,
        showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('cyan')
    pc.set_edgecolor('black')
    # pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([1], 0, 5, color='r', linestyle='dotted', lw=2)

# set style for the axes
labels = ['BIC1', 'BIC2', 'BIC3', 'BIC4']

set_axis_style(ax2, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), sharey=True)

ax2.set_title('$\\varepsilon_r$ = 0.4', fontsize=48, fontweight='bold')
parts = ax2.violinplot(
        [np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], showmeans=True, showmedians=False,
        showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('sienna')
    pc.set_edgecolor('black')
    # pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile([np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip([np.array(data[0])+np.array(data[1]), np.array(data[2])+np.array(data[3])], quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax2.hlines([2], 0, 3, color='r', linestyle='dotted', lw=2)

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
# axes[1].set_title('General BIC with different $\\varepsilon_r$', fontsize=32, fontweight='bold')

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