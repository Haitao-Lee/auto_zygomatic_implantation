# coding = utf-8
import numpy as np
import data_input
import visualization
import screw_setting
import core_algorithm
import time


def zygomatic_implant_planning(stl_folder, point_folder, implant_point_dir=None):
    stl_filenames = data_input.get_filenames(stl_folder, ".stl")
    point_filenames = data_input.get_filenames(point_folder, ".txt")
    stls = data_input.getSTLs(stl_filenames)
    implant_points = data_input.getImplantPoints(point_filenames)[0]
    plane = core_algorithm.build_symmetrical_plane(stls[-1])
    actor = visualization.get_screw_cylinder_actor(plane[0], plane[1], 200,1,4)
    actor.GetProperty().SetOpacity(0.2)
    actor.GetProperty().SetColor(1,0,0)
    
    plane_points = core_algorithm.project_points_onto_plane(implant_points, plane[0], plane[1])
    ref_point = np.sum(plane_points, axis=0)/plane_points.shape[0]
    new_obb_poly, new_obb_actor = core_algorithm.transform_obb_box(ref_point, plane[1])
    target_mesh = core_algorithm.connectivity_filter(core_algorithm.crop_by_cylinder(core_algorithm.seperate_maxilla_mandible(stls[-1], plane[0] - ref_point, ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point)), new_obb_poly, ref_point), implant_points[0])
    zygomatic_bone1, zygomatic_bone2, zygomatic_pcd1, zygomatic_pcd2 = core_algorithm.get_zygomatic_bone(target_mesh, plane[1], ref_point, 80)
      
    visual_stls = []
    for stl in stls:
        visual_stls.append(stl)
    visual_stls.append(zygomatic_bone1)
    visual_stls.append(zygomatic_bone2)
    visul_points = np.concatenate([implant_points, plane_points])
    visualization.stl_visualization_by_vtk([core_algorithm.seperate_maxilla_mandible(stls[-1], plane[0] - ref_point, ref_point - (plane[0] - ref_point)*10/np.linalg.norm(plane[0] - ref_point))], np.concatenate([visul_points, np.array([plane[0]])], axis=0), [actor, new_obb_actor], opacity=0.8)
    
    planning_pcds = [data_input.getPCDfromSTL([target_mesh])[0], zygomatic_pcd1, zygomatic_pcd2]
    visualization.stl_pcd_visualization_by_vtk(visual_stls, planning_pcds)
    
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
    
    
zygomatic_implant_planning(screw_setting.stl_dir,screw_setting.ip_dir)