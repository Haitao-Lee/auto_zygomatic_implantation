# Point2Path: an efficient zygomatic implantation planning framework


![Fig.1 The overview of our pipeline comprises three stages: (a) automatic zygomatic bone positioning through shape-prior-knowledge-based multi-planar cutting. (b) Generation of partitioned alternative trajectories guided by extracting dense 'backlit' points. (c) Utilization of a BIC-maximized algorithm to determine the optimal implanted paths.](https://github.com/Haitao-Lee/auto_zygomatic_implantation/blob/main/fig/overview.png)
Fig.1 The overview of our pipeline comprises three stages: (a) automatic zygomatic bone positioning through shape-prior-knowledge-based multi-planar cutting. (b) Generation of partitioned alternative trajectories guided by extracting dense 'backlit' points. (c) Utilization of a BIC-maximized algorithm to determine the optimal implanted paths.




![Fig.2 Comparison of the time consuming with different \(N_f\) in point cloud filling of 3 cases](https://github.com/Haitao-Lee/auto_zygomatic_implantation/blob/main/fig/surface.png)
Fig.2 Comparison of the time consuming with different \(N_f\) in point cloud filling of 3 cases.




![Fig. 3 (a-i) present the qualitative implantation planning results from 9 cases by our method. (j-m) respectively plot the original BIC of \(P_1\)-\(P_4\) in 18 cases between our planning results and those of manual planning.](https://github.com/Haitao-Lee/auto_zygomatic_implantation/blob/main/fig/results.png)
Fig. 3 (a-i) present the qualitative implantation planning results from 9 cases by our method. (j-m) respectively plot the original BIC of \(P_1\)-\(P_4\) in 18 cases between our planning results and those of manual planning.




![Fig.4 (a) Comparison of \(d_1\) to \(d_4\) between our planning results and manual results by expert doctors with different \(\varepsilon_r\), where \(d_1\)-\(d_4\) denotes the distance from the implant paths \(P_1\)-\(P_4\) to either the infraorbital margin or the lower zygomatic bone edge. (b) The relative BIC of path 1 to path 4 planned by our method compared with the corresponding manual design, denoted as BIC1 to BIC4. (c) The relative overall BIC in the left or right zygomatic bone with different \(\varepsilon_r\).
](https://github.com/Haitao-Lee/auto_zygomatic_implantation/blob/main/fig/d1-d4.png)
Fig.4 (a) Comparison of \(d_1\) to \(d_4\) between our planning results and manual results by expert doctors with different \(\varepsilon_r\), where \(d_1\)-\(d_4\) denotes the distance from the implant paths \(P_1\)-\(P_4\) to either the infraorbital margin or the lower zygomatic bone edge. (b) The relative BIC of path 1 to path 4 planned by our method compared with the corresponding manual design, denoted as BIC1 to BIC4. (c) The relative overall BIC in the left or right zygomatic bone with different \(\varepsilon_r\).




# The project can be reproduced on window 11 with the following command:

-- git clone  https://github.com/Haitao-Lee/auto_zygomatic_implantation.git

-- conda create --name zygo_implant_planning python=3.9

-- conda create zygo_implant_planning

-- pip install open3d

-- pip install vtk

-- pip install numpy

-- pip install matplotlib

-- pip install scikit-learn

-- pip install scipy

-- pip install tqdm

-- pip install random2



# Then open 'main.py', set the 'stl_folder' in function *zygomatic_implant_planning* to the directory that only contains the STL of the patient's skull. Set the 'point_folder' in function *zygomatic_implant_planning* to the directory that only contains the txt file that stores the implantation points. The txt file is written in the following format:

x1 y1 z1

x2 y1 z2

x3 y3 z3

x4 y4 z4
