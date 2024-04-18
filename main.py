# coding = utf-8
import numpy as np
import data_input
import visualization
import screw_setting
import core_algorithm
import vtkmodules.all as vtk


def zygomatic_implant_planning(stl_folder, point_folder, implant_point_dir=None):
    stl_filenames = data_input.get_filenames(stl_folder, ".stl")
    point_filenames = data_input.get_filenames(point_folder, ".txt")
    stls = data_input.getSTLs(stl_filenames)
    implant_points = data_input.getImplantPoints(point_filenames)[0]
    visualization.stl_visualization_by_vtk([stls[-1]], implant_points)
    visualization.stl_visualization_by_vtk(stls, implant_points, opacity=0.8)
    plane = core_algorithm.build_symmetrical_plane(stls[-1], implant_points)
    actor = visualization.get_screw_cylinder_actor(plane[0], plane[1], 200,1,4)
    actor.GetProperty().SetOpacity(0.4)
    actor.GetProperty().SetColor(1,0,0)
    
    plane_points = core_algorithm.project_points_onto_plane(implant_points, plane[0], plane[1])
    ref_point = np.sum(plane_points, axis=0)/plane_points.shape[0]
    new_obb_poly, new_obb_actor = core_algorithm.transform_obb_box(ref_point, plane[1])
    visualization.stl_visualization_by_vtk([stls[-1], new_obb_poly])
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
    
zygomatic_implant_planning(screw_setting.stl_dir,screw_setting.ip_dir)