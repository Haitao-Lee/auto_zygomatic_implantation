# coding = utf-8
import os
import vtkmodules.all as vtk
import open3d as o3d
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pydicom
import os


def get_filenames(path, filetype):  # 输入路径、文件类型例如'.csv'
    names = []
    for _, _, files in os.walk(path):
        for i in files:
            if os.path.splitext(i)[1] == filetype:
                names.append(path + '/' + i)
    return names  # 输出由有后缀的文件名组成的列表


# obtain the lists of STL files in the same folder
def getSTLs(fileNames):
    STLs = []
    for fileName in fileNames:
        stl_reader = vtk.vtkSTLReader()
        stl_reader.SetFileName(fileName)
        stl_reader.Update()
        STLs.append(stl_reader.GetOutput())
    return STLs


# obtain the lists of PCD files in the same folder
def getPCDs(fileNames):
    PCDs = []
    for fileName in fileNames:
        PCD = o3d.io.read_point_cloud(fileName)
        PCDs.append(PCD)
    return PCDs


# obtain the lists of npy files in the same folder
def getNPYs(fileNames):
    NPYs = []
    for fileName in fileNames:
        NPY = np.load(fileName)
        NPYs.append(NPY)
    return NPYs


def getNIIasNPY(fileName):
    img = nib.load(fileName).get_data() #载入
    img = np.array(img)
    return img


def readPCDfromSTL(fileNames):
    PCDs = []
    for fileName in fileNames:
        mesh_ply = o3d.io.read_triangle_mesh(fileName)
        PCD = o3d.geometry.PointCloud()
        PCD.points = mesh_ply.vertices
        PCD.normals = mesh_ply.vertex_normals
        PCD.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
        PCDs.append(PCD)
    return PCDs

def getPCDfromSTL(STLs, sample_rate=20):
    PCDs = []
    for i in range(len(STLs)):
        points_npy = np.array(STLs[i].GetPoints().GetData())[::sample_rate]
        # 将NumPy数组转换为Open3D点云
        PCD = o3d.geometry.PointCloud()
        PCD.points = o3d.utility.Vector3dVector(points_npy)
        PCDs.append(PCD)
    return PCDs


def getImplantPoints(fileNames):
    points = []
    for fileName in fileNames:
        ps = np.loadtxt(fileName, delimiter=',')
        points.append(ps)
    return np.array(points)


'''之所以用2个库来读取dicom。是因为simpleITK和pydicom在当前版本下和numpy、vtk等都有兼容问题,单个无法满足要求，只能妥协点读取时间了'''
def importDicom(path):
    # 获取文件夹中的所有文件名
    if not os.path.isdir(path):
        return None, None
    files = os.listdir(path)
    # 用于存储DICOM文件的列表
    dicom_files = []
    # 遍历文件夹中的文件
    for file_name in files:
        file_path = os.path.join(path, file_name).replace("\\","/")
        # 检查文件是否是DICOM格式
        if os.path.isfile(file_path) and file_name.lower().endswith('.dcm'):
            dicom_files.append(file_path)
    # 读取DICOM文件
    if len(dicom_files) == 0:
        return None, None
    dicom_data = [pydicom.dcmread(file) for file in dicom_files]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    return dicom_data, image



'''
# DICOM序列所在的文件夹路径
dicom_dir = '/path/to/your/dicom/folder/'

# 读取DICOM序列
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
reader.SetFileNames(dicom_names)
image = reader.Execute()

# 获取DICOM序列的元数据信息
dicom_tags = image.GetMetaDataKeys()

# 获取病人信息
patient_name = image.GetMetaData("0010|0010")  # 患者姓名
patient_id = image.GetMetaData("0010|0020")    # 患者ID
study_description = image.GetMetaData("0008|1030")  # 检查描述
study_date = image.GetMetaData("0008|0020")    # 检查日期

# 打印病人信息
print("Patient Name:", patient_name)
print("Patient ID:", patient_id)
print("Study Description:", study_description)
print("Study Date:", study_date)
 
"0010|0010"：患者姓名
"0010|0020"：患者 ID
"0008|1030"：检查描述
"0010|0040"：患者性别
"0010|1010"：患者年龄
"0008|0020"：检查日期
"0008|103e"：系列描述
"0018|0088"：扫描时间
"0018|0015"：扫描体部位
"0018|0050"：切片厚度
"0018|0060"：扫描模式
"0020|0011"：序列编号
"0020|0013"：扫描编号
"0028|0101"：比特数
"0028|0100"：位深度
"0028|0010"：行像素
"0028|0011"：列像素
'''
