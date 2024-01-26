# coding = utf-8
import screw_setting
import numpy as np
import open3d as o3d
from tqdm import tqdm
import scipy
import scipy.spatial as spatial
import vtkmodules.all as vtk
import geometry
import visualization
from vtkbool.vtkBool import vtkPolyDataBooleanFilter
import data_input
import copy
import time



def remove_outliers(pcds, nd=screw_setting.nd, std_rt=screw_setting.std_rt):
    new_pcds = []
    for pcd in pcds:
        n_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nd, std_ratio=std_rt)
        new_pcds.append(n_pcd)
    return new_pcds


def seperate_maxilla_mandible(stl, direc, point):
    plane = vtk.vtkPlane()
    plane.SetOrigin(point)
    plane.SetNormal(direc)  # 以 x 轴为法线的平面
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(stl)
    clipper.SetClipFunction(plane)
    clipper.GenerateClippedOutputOn()  # 生成被裁剪部分的输出
    clipper.Update()
    tmp_stl = clipper.GetOutput()
    return tmp_stl


def get_rest_stl(polydata1, polydata2):
    # 创建布尔运算过滤器
    boolean_operation = vtk.vtkBooleanOperationPolyDataFilter()
    boolean_operation.SetOperationToDifference()  # 设置为差集
    # 设置输入STL文件
    boolean_operation.SetInputData(0, polydata1)
    boolean_operation.SetInputData(1, polydata2)
    # 更新过滤器
    boolean_operation.Update()
    # 获取输出的vtkPolyData
    result_polydata = boolean_operation.GetOutput()
    return result_polydata


def connectivity_filter(stl, seed_point = None, num=3):
    # 连通性过滤器
    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputData(stl)
    if seed_point is None:
        connectivity_filter.SetExtractionModeToLargestRegion()  # 提取所有连通区域
        connectivity_filter.ColorRegionsOn()
    else:
        connectivity_filter.SetExtractionModeToClosestPointRegion()  # 保留离种子点最近的区域
        connectivity_filter.SetClosestPoint(seed_point)
    connectivity_filter.Update()
    return connectivity_filter.GetOutput()


def mirrored_x(stl):
    # 创建一个Transform
    transform = vtk.vtkTransform()
    transform.Scale(-1, 1, 1)  # 在X轴方向上进行镜像，可以根据需要调整镜像方向

    # 应用变换
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(stl)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    # 获取镜像后的STL数据
    return transform_filter.GetOutput()


def project_points_onto_plane(points,  plane_point, plane_normal):
    """
    将一些点投影到给定平面上。
    参数：
    - points: 三维空间中的点,numpy数组,每一行表示一个点的坐标。
    - plane_normal: 平面的法向量,numpy数组。
    - plane_point: 平面上的一个点,numpy数组。
    返回：
    - 投影点的坐标,numpy数组。
    """
    # 确保输入是numpy数组
    points = np.array(points)
    plane_normal = np.array(plane_normal)
    plane_point = np.array(plane_point)
    # 计算平面上的投影矩阵
    projection_matrix = np.eye(3) - np.outer(plane_normal, plane_normal)
    # 计算每个点的投影点
    projected_points = []
    for point in points:
        # 计算点到平面上的向量
        vector_to_plane = point - plane_point
        # 计算投影点
        projected_point = plane_point + np.dot(projection_matrix, vector_to_plane)
        projected_points.append(projected_point)
    return np.array(projected_points)


def crop_by_cylinder(stl1, cylinder, ref_point):
    # 获取单元数量
    num_cells = cylinder.GetNumberOfCells()
    # 遍历每个单元
    plane_points = []
    for i in range(num_cells):
        # 获取当前单元
        cell = cylinder.GetCell(i)
        # 获取单元上的点的数量
        num_points_in_cell = cell.GetNumberOfPoints()
        # 提取单元上的点坐标
        cell_points = []
        for j in range(num_points_in_cell):
            point_id = cell.GetPointId(j)
            point = cylinder.GetPoint(point_id)
            cell_points.append(np.array(point))
        plane_points.append(np.array(cell_points))
    tmp_stl = stl1
    # renderer = vtk.vtkRenderer()
    # render_window = vtk.vtkRenderWindow()
    # rw_style = vtk.vtkInteractorStyleTrackballCamera()
    # rw_interactor = vtk.vtkRenderWindowInteractor()
    # rw_interactor.SetRenderWindow(render_window)
    # rw_interactor.SetInteractorStyle(rw_style)
    # render_window.SetWindowName("Plane with Specified Center and Normal")
    # render_window.SetSize(800, 600)
    for i in range(len(plane_points)):
        point0 = plane_points[i][0]
        point1 = plane_points[i][1]
        point2 = plane_points[i][2]
        normal = np.cross(point1 - point0, point2 - point0)
        normal = normal/np.linalg.norm(normal)
        if np.dot(normal, ref_point - point0) < 0:
            normal = -normal
        # if plane_points[i].shape[0] == 4:
        #     actor = visualization.get_screw_cylinder_actor(np.sum(plane_points[i], axis=0)/plane_points[i].shape[0], normal,80,1,4)
        #     actor.GetProperty().SetOpacity(0.2)
        #     actor.GetProperty().SetColor(0, 1, 1)
        #     renderer.AddActor(actor)
        # print_info(tmp_stl)
        # 创建平面裁切器
        plane = vtk.vtkPlane()
        plane.SetOrigin(point0)
        plane.SetNormal(normal)  # 以 x 轴为法线的平面
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(tmp_stl)
        clipper.SetClipFunction(plane)
        clipper.GenerateClippedOutputOn()  # 生成被裁剪部分的输出
        clipper.Update()
        tmp_stl = clipper.GetOutput()
    # mapper1 = vtk.vtkPolyDataMapper()
    # mapper1.SetInputData(stl1)
    # actor1 = vtk.vtkActor()
    # actor1.SetMapper(mapper1)
    # renderer.AddActor(actor1)
    # render_window.AddRenderer(renderer)
    # # 设置相机位置
    # renderer.GetActiveCamera().Azimuth(30)
    # renderer.GetActiveCamera().Elevation(30)
    # renderer.ResetCamera()
    # # 显示窗口
    # render_window.Render()
    # rw_interactor.Start()
    return tmp_stl



def transform_obb_box(center, normal, distance=screw_setting.crop_distance):
    cylinder_actor = visualization.get_screw_cylinder_actor(center, normal, 0.6*distance, 3*distance, 12)
    cylinder_actor.GetProperty().SetColor(1,1,1)
    cylinder_actor.GetProperty().SetOpacity(0.2)
    polydata = visualization.get_screw_cylinder_polydata(center, normal, 0.6*distance, 3*distance, 12)
    return  polydata, cylinder_actor


def regularize_info(implant_points, effect_pcds, ref_point, normal):
    directs = np.array(implant_points) - np.array(ref_point)
    res = np.dot(directs, -normal)
    # 使用argsort对数组进行排序，并输出排序后的索引
    sorted_indices = np.argsort(res)
    new_ip_points = np.array(implant_points)[sorted_indices]
    # visualization.points_visualization_by_vtk(effect_pcds[0:1],[new_ip_points[0]])
    # visualization.points_visualization_by_vtk(effect_pcds[0:1],[new_ip_points[1]])
    # visualization.points_visualization_by_vtk(effect_pcds[0:1],[new_ip_points[2]])
    # visualization.points_visualization_by_vtk(effect_pcds[0:1],[new_ip_points[3]])
    num1 = np.sum(res>0)
    num2 = np.sum(res==0)
    num3 = new_ip_points.shape[0] - num1 - num2
    effect_points = []
    for effect_pcd in effect_pcds:
        effect_points.append(np.asarray(effect_pcd.points))
    effect_points[0] = np.concatenate([effect_points[0], effect_points[1], effect_points[2]], axis=0)
    info = []
    if num2 == 0:
        if num1 == 1:
            info.append([new_ip_points[0], effect_points[0], [effect_points[1]]])
        elif num1 == 2:
            implant_center = np.sum(new_ip_points[0:num1], axis=0)/num1
            target_center = np.sum(effect_points[1], axis=0)/effect_points[1].shape[0]
            ref_dir = target_center - implant_center
            ref_dir = ref_dir/np.linalg.norm(ref_dir)
            divide_dir = new_ip_points[1] - new_ip_points[0]
            divide_dir = divide_dir/np.linalg.norm(divide_dir)
            divide_normal = np.cross(np.cross(ref_dir, divide_dir), ref_dir)
            res = np.dot(effect_points[1] - target_center, divide_normal)
            indices = np.array(np.argwhere(res <= 0)).flatten()
            info.append([new_ip_points[0], effect_points[0], [effect_points[1][indices]]])
            indices = np.array(np.argwhere(res > 0)).flatten()
            info.append([new_ip_points[1], effect_points[0], [effect_points[1][indices]]])
        if num3 == 1:
            info.append([new_ip_points[num1+num2], effect_points[0], [effect_points[2]]])
        elif num3 == 2:
            implant_center = np.sum(new_ip_points[num1+num2:], axis=0)/num3
            target_center = np.sum(effect_points[2], axis=0)/effect_points[2].shape[0]
            ref_dir = target_center - implant_center
            ref_dir = ref_dir/np.linalg.norm(ref_dir)
            divide_dir = new_ip_points[num1+num2] - new_ip_points[num1+num2+1]
            divide_dir = divide_dir/np.linalg.norm(divide_dir)
            divide_normal = np.cross(np.cross(ref_dir, divide_dir), ref_dir)
            res = np.dot(effect_points[2] - target_center, divide_normal)
            indices = np.array(np.argwhere(res <= 0)).flatten()
            info.append([new_ip_points[num1+num2+1], effect_points[0], [effect_points[2][indices]]])
            indices = np.array(np.argwhere(res > 0)).flatten()
            info.append([new_ip_points[num1+num2], effect_points[0], [effect_points[2][indices]]])
    elif num2 == 1:
        if num1 + num2 == 0:
            info.append([new_ip_points[0], effect_points[0], [effect_points[1], effect_points[2]]])
            num1 = num2 = num3 = 0
        elif num1 + num3 == 1:
            info.append([new_ip_points[0], effect_points[0], [effect_points[1]]])
            info.append([new_ip_points[1], effect_points[0], [effect_points[2]]])
            num1 = num2 = num3 = 0
        elif num1 == 1 and num3 == 1:
            info.append([new_ip_points[0], effect_points[0], [effect_points[1]]])
            info.append([new_ip_points[2], effect_points[0], [effect_points[2]]])
            info.append([new_ip_points[1], effect_points[0], [effect_points[1], effect_points[2]]])
            num1 = num2 = num3 = 0
        elif num1 == 2:
            num2 = 0
            num3 = num3 + 1
        elif num3 == 2:
            num2 = 0
            num1 = num1 + 1
        if num1 == 1:
            info.append([new_ip_points[0], effect_points[0], [effect_points[1]]])
        elif num1 == 2:
            implant_center = np.sum(new_ip_points[0:num1], axis=0)/num1
            target_center = np.sum(effect_points[1], axis=0)/effect_points[1].shape[0]
            ref_dir = target_center - implant_center
            ref_dir = ref_dir/np.linalg.norm(ref_dir)
            divide_dir = new_ip_points[1] - new_ip_points[0]
            divide_dir = divide_dir/np.linalg.norm(divide_dir)
            divide_normal = np.cross(np.cross(ref_dir, divide_dir), ref_dir)
            res = np.dot(effect_points[1] - target_center, divide_normal)
            indices = np.array(np.argwhere(res <= 0)).flatten()
            info.append([new_ip_points[0], effect_points[0], [effect_points[1][indices]]])
            indices = np.array(np.argwhere(res > 0)).flatten()
            info.append([new_ip_points[1], effect_points[0], [effect_points[1][indices]]])
        if num3 == 1:
            info.append([new_ip_points[num1+num2], effect_points[0], effect_points[2]])
        elif num3 == 2:
            implant_center = np.sum(new_ip_points[num1+num2:], axis=0)/num3
            target_center = np.sum(effect_points[2], axis=0)/effect_points[2].shape[0]
            ref_dir = target_center - implant_center
            ref_dir = ref_dir/np.linalg.norm(ref_dir)
            divide_dir = new_ip_points[num1+num2] - new_ip_points[num1+num2+1]
            divide_dir = divide_dir/np.linalg.norm(divide_dir)
            divide_normal = np.cross(np.cross(ref_dir, divide_dir), ref_dir)
            res = np.dot(effect_points[2] - target_center, divide_normal)
            indices = np.array(np.argwhere(res <= 0)).flatten()
            info.append([new_ip_points[num1+num2+1], effect_points[0], [effect_points[2][indices]]])
            indices = np.array(np.argwhere(res > 0)).flatten()
            info.append([new_ip_points[num1+num2], effect_points[0], [effect_points[2][indices]]])
    elif num2 == 2:
        if num1 + num2 == 0:
            info.append([new_ip_points[0], effect_points[0], [effect_points[1]]])
            info.append([new_ip_points[1], effect_points[0], [effect_points[2]]])
            num1 = num2 = num3 = 0
        elif num1 + num3 == 1:
            info.append([new_ip_points[0], effect_points[0], [effect_points[1]]])
            info.append([new_ip_points[2], effect_points[0], [effect_points[2]]])
            info.append([new_ip_points[1], effect_points[0], [effect_points[1], effect_points[2]]])
            num1 = num2 = num3 = 0
        elif num1 == 1 and num3 == 1:
            num2 = 0
            num1 = 2
            num3 = 2
        elif num1 == 2:
            num2 = 0
            num3 = num3 + 2
        elif num3 == 2:
            num2 = 0
            num1 = num1 + 2
        if num1 == 1:
            info.append([new_ip_points[0], effect_points[0], [effect_points[1]]])
        elif num1 == 2:
            implant_center = np.sum(new_ip_points[0:num1], axis=0)/num1
            target_center = np.sum(effect_points[1], axis=0)/effect_points[1].shape[0]
            ref_dir = target_center - implant_center
            ref_dir = ref_dir/np.linalg.norm(ref_dir)
            divide_dir = new_ip_points[1] - new_ip_points[0]
            divide_dir = divide_dir/np.linalg.norm(divide_dir)
            divide_normal = np.cross(np.cross(ref_dir, divide_dir), ref_dir)
            res = np.dot(effect_points[1] - target_center, divide_normal)
            indices = np.array(np.argwhere(res <= 0)).flatten()
            info.append([new_ip_points[0], effect_points[0], [effect_points[1][indices]]])
            indices = np.array(np.argwhere(res > 0)).flatten()
            info.append([new_ip_points[1], effect_points[0], [effect_points[1][indices]]])
        if num3 == 1:
            info.append([new_ip_points[num1+num2], effect_points[0], [effect_points[2]]])
        elif num3 == 2:
            implant_center = np.sum(new_ip_points[num1+num2:], axis=0)/num3
            target_center = np.sum(effect_points[2], axis=0)/effect_points[2].shape[0]
            ref_dir = target_center - implant_center
            ref_dir = ref_dir/np.linalg.norm(ref_dir)
            divide_dir = new_ip_points[num1+num2] - new_ip_points[num1+num2+1]
            divide_dir = divide_dir/np.linalg.norm(divide_dir)
            divide_normal = np.cross(np.cross(ref_dir, divide_dir), ref_dir)
            res = np.dot(effect_points[2] - target_center, divide_normal)
            indices = np.array(np.argwhere(res <= 0)).flatten()
            info.append([new_ip_points[num1+num2+1], effect_points[0], [effect_points[2][indices]]])
            indices = np.array(np.argwhere(res > 0)).flatten()
            info.append([new_ip_points[num1+num2], effect_points[0], [effect_points[2][indices]]])
    return info


def get_zygomatic_bone(stl, normal, ref_point, distance=screw_setting.crop_distance, sample_rate=10):
    center1 = ref_point + normal*distance/2
    center2 = ref_point - normal*distance/2
    
    plane1 = vtk.vtkPlane()
    plane1.SetOrigin(center1)
    plane1.SetNormal(normal)  # 以 x 轴为法线的平面
    clipper1 = vtk.vtkClipPolyData()
    clipper1.SetInputData(stl)
    clipper1.SetClipFunction(plane1)
    clipper1.GenerateClippedOutputOn()  # 生成被裁剪部分的输出
    clipper1.Update()
    
    plane2 = vtk.vtkPlane()
    plane2.SetOrigin(center2)
    plane2.SetNormal(-normal)  # 以 x 轴为法线的平面
    clipper2 = vtk.vtkClipPolyData()
    clipper2.SetInputData(stl)
    clipper2.SetClipFunction(plane2)
    clipper2.GenerateClippedOutputOn()  # 生成被裁剪部分的输出
    clipper2.Update()
    
    points_npy = np.array(stl.GetPoints().GetData())[::sample_rate]
    dif_npy = points_npy - ref_point
    res = np.dot(dif_npy, normal)
    indices1 = np.array(np.argwhere(res > distance/2)).flatten()
    points1_npy = np.array(points_npy[indices1])
    # 将NumPy数组转换为Open3D点云
    PCD1 = o3d.geometry.PointCloud()
    PCD1.points = o3d.utility.Vector3dVector(geometry.fill_pointcloud(points1_npy))
    
    indices2 = np.array(np.argwhere(res < -distance/2)).flatten()
    points2_npy = np.array(points_npy[indices2])
    # 将NumPy数组转换为Open3D点云
    PCD2 = o3d.geometry.PointCloud()
    PCD2.points = o3d.utility.Vector3dVector(geometry.fill_pointcloud(points2_npy))
    
    return clipper1.GetOutput(), clipper2.GetOutput(), PCD1, PCD2
    


def print_info(polydata):
    print(f"Number of Points: {polydata.GetNumberOfPoints()}")
    print(f"Number of Cells: {polydata.GetNumberOfCells()}")


def get_obb_bbox(stl, max_level=0):
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(stl)
    obbTree.SetMaxLevel(max_level)
    obbTree.BuildLocator()
    obbPolydata = vtk.vtkPolyData()
    obbTree.GenerateRepresentation(0, obbPolydata)
    obb_vertices = []
    obb_points = obbPolydata.GetPoints()
    for i in range(obb_points.GetNumberOfPoints()):
        point = obb_points.GetPoint(i)
        obb_vertices.append(point)
    
    # mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputData(obbPolydata)
    # actor = vtk.vtkActor()
    # actor.SetMapper(mapper)
    # actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 设置包围盒的颜色
    # actor.GetProperty().SetOpacity(0.4)  # 设置包围盒的颜色
    
    # mapper1 = vtk.vtkPolyDataMapper()
    # mapper1.SetInputData(stl)
    # actor1 = vtk.vtkActor()
    # actor1.SetMapper(mapper1)
    # actor1.GetProperty().SetColor(1.0, 0.0, 0.0)  # 设置包围盒的颜色
    # # 创建渲染器和窗口
    # renderer = vtk.vtkRenderer()
    # renderer.SetBackground(1.0, 1.0, 1.0)  # 设置背景颜色
    # render_window = vtk.vtkRenderWindow()
    # render_window.SetWindowName("Oriented Bounding Box")
    # render_window.SetSize(800, 600)
    # render_window_interactor = vtk.vtkRenderWindowInteractor()
    # render_window_interactor.SetRenderWindow(render_window)
    # rw_style = vtk.vtkInteractorStyleTrackballCamera()
    # render_window_interactor.SetInteractorStyle(rw_style)
    # # 创建坐标系
    # axes_actor = vtk.vtkAxesActor()
    # axes_actor.AxisLabelsOn() 
    # # 调整坐标轴标签尺寸（增加坐标轴长度）
    # axes_actor.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    # axes_actor.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    # axes_actor.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    # axes_actor.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    # axes_actor.SetTotalLength(50, 50, 50)
    # # 将Actor添加到Renderer
    # renderer.AddActor(actor)
    # renderer.AddActor(actor1)
    # renderer.AddActor(axes_actor)
    # # 将Renderer添加到RenderWindow
    # render_window.AddRenderer(renderer)
    # # 设置相机位置
    # renderer.GetActiveCamera().Azimuth(30)
    # renderer.GetActiveCamera().Elevation(30)
    # renderer.ResetCamera()
    # # 显示窗口
    # render_window.Render()
    # render_window_interactor.Start()
    return obb_vertices, obbPolydata
	

def icp_transform(source_stl, target_stl):
    # 创建ICP变换
    icp_transform = vtk.vtkIterativeClosestPointTransform()
    icp_transform.SetSource(source_stl)
    icp_transform.SetTarget(target_stl)
    icp_transform.GetLandmarkTransform().SetModeToRigidBody()  # 可以根据需要选择不同的变换模式
    icp_transform.SetMaximumNumberOfIterations(100)
    # 应用ICP变换
    icp_transform.Modified()
    icp_transform.Update()
    # 获取变换后的STL数据
    transformed_stl_data = vtk.vtkTransformPolyDataFilter()
    transformed_stl_data.SetInputData(source_stl)
    transformed_stl_data.SetTransform(icp_transform)
    transformed_stl_data.Update()
    return icp_transform, transformed_stl_data.GetOutput()


def transformPoints(transform, points):
    # 创建一些示例点
    original_points = vtk.vtkPoints()
    for point in points:       
        original_points.InsertNextPoint(point[0], point[1], point[2])
    # 创建一个PolyData对象并将点添加到其中
    original_polydata = vtk.vtkPolyData()
    original_polydata.SetPoints(original_points)
    # 创建vtkTransformPolyDataFilter进行变换
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(original_polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    # 获取变换后的点
    transformed_points = transform_filter.GetOutput().GetPoints()
    new_points = []
    for i in range(transformed_points.GetNumberOfPoints()):
        new_points.append(transformed_points.GetPoint(i))
    return new_points


def get_center(stl):
    center_of_mass = vtk.vtkCenterOfMass()
    center_of_mass.SetInputData(stl)
    center_of_mass.Update()
    return center_of_mass.GetCenter()


def fromSTL2o3d(stl):
    points = stl.GetPoints()
    num_points = points.GetNumberOfPoints()

    # 将vtkPoints的坐标转换为NumPy数组
    numpy_points = np.array([points.GetPoint(i) for i in range(num_points)])

    # 创建Open3D的点云数据结构
    open3d_point_cloud = o3d.geometry.PointCloud()
    open3d_point_cloud.points = o3d.utility.Vector3dVector(numpy_points)


def build_symmetrical_plane(stl):
    mirrored_stl = mirrored_x(stl)
    original_bb_points, original_obb_ply = get_obb_bbox(stl)
    original_bb_points = np.array(original_bb_points)
    mirrored_bb_points,_ = get_obb_bbox(mirrored_stl)
    mirrored_bb_points = np.array(mirrored_bb_points)
    orignal_center = np.sum(original_bb_points, axis=0)/original_bb_points.shape[0]
    mirrored_center = np.sum(mirrored_bb_points, axis=0)/mirrored_bb_points.shape[0]
    offset = orignal_center - mirrored_center
    # 创建一个vtkTransform对象进行平移
    translation_transform = vtk.vtkTransform()
    translation_transform.Translate(offset[0], offset[1], offset[2])  # 指定平移的偏移量
    # 创建vtkTransformFilter进行平移变换
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetInputData(mirrored_stl)
    transform_filter.SetTransform(translation_transform)
    transform_filter.Update()
    # 获取平移后的STL数据
    mirrored_stl = transform_filter.GetOutput()
    # 将NumPy数组转换为Open3D点云
    source_cloud = data_input.getPCDfromSTL([mirrored_stl], 1000)[0]
    target_cloud = data_input.getPCDfromSTL([stl], 1000)[0]

    # 执行ICP配准
    icp_result = o3d.pipelines.registration.registration_icp(
        source_cloud, 
        target_cloud,
        max_correspondence_distance=0.5,  # 设置阈值以控制收敛
        init=np.identity(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-2, relative_rmse=1e-2, max_iteration=1000)
    )

    # 获取变换矩阵
    transformation_matrix = icp_result.transformation
    transform = vtk.vtkTransform()
    transform.SetMatrix(transformation_matrix.flatten())

    # 创建TransformFilter进行变换
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(mirrored_stl)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    new_bone = transform_filter.GetOutput()
    _, new_bone = icp_transform(mirrored_stl, stl)
    # mirrored_bb_points = mirrored_bb_points + offset
    mirrored_bb_points, mirrored_obb_ply = get_obb_bbox(new_bone)  #transformPoints(transform, mirrored_bb_points) #
    mirrored_bb_points = np.array(mirrored_bb_points)
    mirrored_center = np.sum(mirrored_bb_points, axis=0)/mirrored_bb_points.shape[0] 
    original_all = np.array(np.sum(original_bb_points, axis=0))
    mirrored_all = np.array(np.sum(mirrored_bb_points, axis=0))
    direct = original_all - mirrored_all
    direct = direct/np.linalg.norm(direct)       
    center = (mirrored_center+orignal_center)/2
    points1 = []
    points2 = []
    for i in range(mirrored_bb_points.shape[0]):
        tmp_direct = mirrored_bb_points[i] - center
        if np.dot(tmp_direct, direct) > 0:
            points1.append(mirrored_bb_points[i])
        else:
            points2.append(mirrored_bb_points[i])
        tmp_direct = original_bb_points[i] - center
        if np.dot(tmp_direct, direct) > 0:
            points1.append(original_bb_points[i])
        else:
            points2.append(original_bb_points[i])
    new_direct = np.sum(np.array(points1), axis=0) - np.sum(np.array(points2), axis=0)
    new_direct = new_direct/np.linalg.norm(new_direct)  
    plane = [center, new_direct]
    
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    rw_style = vtk.vtkInteractorStyleTrackballCamera()
    rw_interactor = vtk.vtkRenderWindowInteractor()
    rw_interactor.SetRenderWindow(render_window)
    rw_interactor.SetInteractorStyle(rw_style)
    render_window.SetWindowName("Plane with Specified Center and Normal")
    render_window.SetSize(800, 600)
    # 创建Mapper和Actor
    actor = visualization.get_screw_cylinder_actor(center, new_direct, 200,1,4)
    actor.GetProperty().SetColor(1,0,0)
    actor.GetProperty().SetOpacity(0.4)
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputData(stl)
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetColor(0.8,0.8,0.8)
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(new_bone)
    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)
    actor2.GetProperty().SetColor(0,1,0)
    mapper3 = vtk.vtkPolyDataMapper()
    mapper3.SetInputData(mirrored_stl)
    actor3 = vtk.vtkActor()
    actor3.SetMapper(mapper3)
    actor3.GetProperty().SetColor(0,0,1)
    mapper4 = vtk.vtkPolyDataMapper()
    mapper4.SetInputData(original_obb_ply)
    actor4 = vtk.vtkActor()
    actor4.SetMapper(mapper4)
    actor4.GetProperty().SetOpacity(0.2)
    mapper5 = vtk.vtkPolyDataMapper()
    mapper5.SetInputData(mirrored_obb_ply)
    actor5 = vtk.vtkActor()
    actor5.SetMapper(mapper5)
    actor5.GetProperty().SetOpacity(0.2)
    
    # 将Actor添加到Renderer
    renderer.AddActor(actor)
    renderer.AddActor(actor1)
    renderer.AddActor(actor2)
    renderer.AddActor(actor3)
    renderer.AddActor(actor4)
    renderer.AddActor(actor5)
    # 将Renderer添加到RenderWindow
    render_window.AddRenderer(renderer)
    # 设置相机位置
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(30)
    renderer.ResetCamera()

    # 显示窗口
    render_window.Render()
    rw_interactor.Start()
    return plane


def get_cone(dire, angle=screw_setting.cone_angle, r_resolution=screw_setting.r_res, c_resolution=screw_setting.c_res):
    orth_dir = np.array([dire[2], 0, -dire[0]])
    orth_dir = orth_dir/np.linalg.norm(orth_dir)
    radius = np.tan(angle)
    rot_mtx = scipy.linalg.expm(np.cross(np.eye(3), dire/scipy.linalg.norm(dire)*2*np.pi/c_resolution))
    cone = [dire]
    for i in range(c_resolution):
        orth_dir = np.dot(rot_mtx, orth_dir)
        for j in range(r_resolution):
            n_dir = dire + orth_dir*radius*j/r_resolution
            n_dir = n_dir/np.linalg.norm(n_dir)
            cone.append(n_dir)
    return cone


def get_backlit_points(points, view_point, degree = 3):
    view_direc = points - view_point
    # 计算每个向量的模
    norms = np.linalg.norm(view_direc, axis=1)
    # 归一化每个向量
    normalized_view_direc = view_direc / norms[:, np.newaxis]
    conv_mtx = np.dot(normalized_view_direc, normalized_view_direc.T)
    indices = np.where(conv_mtx > np.sin((90 - degree)*np.pi/180)) #（row_npy, column_npy）
    backlit_indices = []
    for i in range(view_direc.shape[0]):
        if i in backlit_indices:
            continue
        tmp_indices = np.where(indices[0] == i)[0]
        tmp_index = np.argmax(norms[indices[1][tmp_indices]])
        tmp_index = indices[1][tmp_indices[tmp_index]]
        if tmp_index not in backlit_indices:
                backlit_indices.append(tmp_index)
    backlit_indices = np.array(backlit_indices)
    return points[backlit_indices]


def isInterference(point1, point2, point3, point4, eps=3):
    dist, _, _ = geometry.min_distance_between_segments([point1, point2], [point3, point4])
    if dist <= eps:
        return True
    return False


def genrate_optimal_paths(infos, radius=2, margin=2.5, length_interval=[30,52.5]):
    optimal_paths = []
    for info in infos:
        implant_point = info[0]
        whole_bone_npy = info[1]
        # whole_bone_pcd = o3d.geometry.PointCloud()
        # whole_bone_pcd.points = o3d.utility.Vector3dVector(whole_bone_npy)
        zygomatic_bone_npys = info[2]
        candidate_dirs = np.empty([1,3])
        for zygomatic_bone_npy in zygomatic_bone_npys:
            backlit_points = get_backlit_points(zygomatic_bone_npy, implant_point)
            candidate_dirs = np.concatenate([candidate_dirs, backlit_points-implant_point], axis=0)
            # 创建 Open3D 的 PointCloud 对象
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(backlit_points)
            # visualization.points_visualization_by_vtk([whole_bone_pcd, pcd])
        # 删除发生干涉的路径
        effect_candidate_dirs = copy.deepcopy(candidate_dirs)
        # start = time.time()
        for path in optimal_paths:
            p1 = path[0]
            p2 = p1 + path[1]
            p3 = implant_point
            numOfdirs = effect_candidate_dirs.shape[0]
            for i in range(numOfdirs):
                while isInterference(p1,p2,p3, p3+candidate_dirs[i] + 1e-3, eps=2*margin):
                    tmp_norm = np.linalg.norm(candidate_dirs[i])
                    if tmp_norm - 2 > length_interval[0]:
                        candidate_dirs[i] = candidate_dirs[i] - 2*candidate_dirs[i]/tmp_norm
                        effect_candidate_dirs[i] = candidate_dirs[i]
                    else:
                        effect_candidate_dirs[i] = (length_interval[0]-1)*candidate_dirs[i]/tmp_norm
                        break
        # end = time.time()
        # print(end - start)
        # 计算每个向量的模并归一化
        norms = np.linalg.norm(effect_candidate_dirs, axis=1) + 1e-3
        short_indices = np.where(norms < length_interval[0])[0]
        long_indices = np.where(norms > length_interval[1])[0]
        effect_candidate_dirs[long_indices] = effect_candidate_dirs[long_indices]*length_interval[1]/norms[long_indices, np.newaxis]
        effect_candidate_dirs = np.delete(effect_candidate_dirs,short_indices, axis=0)
        norms = np.linalg.norm(effect_candidate_dirs, axis=1) + 1e-3
        # 归一化每个向量
        normalized_candidate_dirs = effect_candidate_dirs / norms[:, np.newaxis]
        bic_dirs = whole_bone_npy - implant_point
        bic_norms = np.reshape(np.linalg.norm(bic_dirs, axis=1), (1, -1))
        bic_norms = np.repeat(bic_norms, effect_candidate_dirs.shape[0], axis=0)
        project_lengths = np.dot(normalized_candidate_dirs, bic_dirs.T) #(n,m)  第n行为第n个candidate_dir和各个方向的点积
        for i in range(project_lengths.shape[0]):
            tmp_indices = np.where(project_lengths[i] > norms[i])[0]
            if tmp_indices.shape[0] > 0:
                project_lengths[i][tmp_indices] = 0
            tmp_indices = np.where(project_lengths[i] < 0)[0]
            if tmp_indices.shape[0] > 0:
                project_lengths[i][tmp_indices] = 0
        distance_mtx = np.square(bic_norms) - np.square(project_lengths)
        indices = np.where(distance_mtx < radius**2)
        if indices[0].shape[0] > 0:
            # 使用 np.bincount 统计每个元素出现的次数
            counts = np.bincount(indices[0])
            # 使用 np.argmax 找到出现次数最多的元素的索引
            most_common_element = np.argmax(counts)
            optimal_paths.append([implant_point, normalized_candidate_dirs[most_common_element]*(norms[most_common_element]-2), radius]) 
    return optimal_paths





