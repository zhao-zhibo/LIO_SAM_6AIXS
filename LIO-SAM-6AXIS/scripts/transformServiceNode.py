#!/usr/bin/env python
import os
import time
import rospy
import numpy as np
import re
import sys
import math
from scipy.spatial.transform import Rotation as R, Slerp
import rospkg


# 动态添加当前脚本所在目录的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)  # [!++ 确保能访问同级geoFunc目录]

# 获取当前包的路径
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('lio_sam_6axis')

# 添加 ROS 生成的 Python 模块路径
sys.path.append(f"{pkg_path}/devel/lib/python3/dist-packages")  # [!++ 关键路径]
sys.path.append(f"{pkg_path}/devel/lib/python3/dist-packages/lio_sam_6axis/srv")  # [!++ 可选]

from lio_sam_6axis.srv import transformInterp, transformInterpRequest,transformInterpResponse
import geoFunc.trans as trans

class TransformInterpolator:
    def __init__(self, ie_path, ref_xyz, Ti0i1):
        # 预加载所有真值数据
        self.all_data = {}
        self.groundTruth = loadIE(ie_path, self.all_data, Ti0i1, ref_xyz)
        
        # 提取时间戳和变换矩阵
        self.groundTruthTimestamps = []
        self.groundTruthTnl = []
        for timestamp, data in self.groundTruth.items():
            self.groundTruthTimestamps.append(timestamp)
            self.groundTruthTnl.append(data['T'])
        
        # 转换为numpy数组加速后续操作
        self.gt_timestamps = np.array(self.groundTruthTimestamps)
        self.gt_matrices = np.array(self.groundTruthTnl)

    def get_transform(self, timestamp):
        """保持原有插值逻辑不变"""
        interpolated = interpolate_SE3(self.gt_timestamps, self.gt_matrices, [timestamp])
        return interpolated[0][1] if interpolated else None

# ----------------------------- 保留原有函数 -----------------------------
# Ti0i1为i1到i0的变换矩阵，也就是lidar frame to 大惯导 frame，最终返回的是lidar frame to n frame的变换矩阵
def loadIE(path : str, all_data : dict, Ti0i1 : np.ndarray, ref_xyz : np.ndarray = None):
    fp = open(path,'rt')

    if ref_xyz is None:
        is_ref_set  = False
    else:
        is_ref_set = True
        Ten0 = np.eye(4,4)
        # n frame  to e frame
        Ten0[0:3,0:3] = trans.Cen(ref_xyz)
        Ten0[0:3,3] = ref_xyz

    while True:
        line = fp.readline().strip()
        if line == '':break
        if line[0] == '#' :continue
        line = re.sub('\s\s+',' ',line)
        elem = line.split(' ')
        sod = float(elem[1])
        if sod not in all_data.keys():
            all_data[sod] ={}
        all_data[sod]['X0']   = float(elem[2])
        all_data[sod]['Y0']   = float(elem[3])
        all_data[sod]['Z0']   = float(elem[4])
        all_data[sod]['VX0']  = float(elem[15])
        all_data[sod]['VY0']  = float(elem[16])
        all_data[sod]['VZ0']  = float(elem[17])
        all_data[sod]['ATTX0']= float(elem[25])
        all_data[sod]['ATTY0']= float(elem[26])
        all_data[sod]['ATTZ0']= -float(elem[24])
        # 每次都重新计算了Ren，也就是认为n系也在动，但是小范围内认为n系不动，距离近的Ren的结果变化很小
        Ren = trans.Cen([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']])
        ani0 = [all_data[sod]['ATTX0']/180*math.pi,\
                all_data[sod]['ATTY0']/180*math.pi,\
                all_data[sod]['ATTZ0']/180*math.pi]
        Rni0 = trans.att2m(ani0) # 旋转矩阵 ,大惯导 frame to n frame

        # 接下来将得到的真值(i0)转换到i1系下，因为本身的真值结果是在大惯导坐标系下(i0为大惯导坐标系)。
        Rni1 = np.matmul(Rni0,Ti0i1[0:3,0:3]) # 得到lidar frame to n frame的旋转矩阵
        Rei1 = np.matmul(Ren,Rni1) # 得到i1传感器在e系下的姿态，也就是lidar frame to e frame的旋转矩阵

        tei0 = np.array([all_data[sod]['X0'],all_data[sod]['Y0'],all_data[sod]['Z0']]) # 大惯导在e系下的位置
        tei1 = tei0 + Ren @ Rni0 @ Ti0i1[0:3,3] # 得到lidar在e系下的位置
        # 最终得到lidar frame to e frame的变换矩阵
        Tei1 = np.eye(4,4)
        Tei1[0:3,0:3] = Rei1
        Tei1[0:3,3] = tei1

        # 如果参考坐标系没有设置，则设置第一个坐标点为n系的原点
        if not is_ref_set:
            is_ref_set = True
            Ten0 = np.eye(4,4)
            Ten0[0:3,0:3] = trans.Cen(tei1)
            Ten0[0:3,3] = tei1
        # 计算i1传感器在n系下的位姿，Ten0是n系到e系的变换矩阵，因此需要求逆
        Tni1 = np.matmul(np.linalg.inv(Ten0),Tei1)
        all_data[sod]['T'] = Tni1 # lidar frame to n frame的变换矩阵

    fp.close()
    return all_data
def interpolate_SE3(timeGroudTruth, SE3GroudTruth, timeCalc):
    # 将输入转换为numpy数组
    time_groudTruth = np.array(timeGroudTruth)
    SE3_groudTruth = np.array(SE3GroudTruth)
    time_calc = np.array(timeCalc)  # 先将timeCalc转换为numpy数组

    # 检查输入的有效性
    assert time_groudTruth.ndim == 1 and len(time_groudTruth) >= 2, "Invalid time_groudTruth."
    assert np.all(np.diff(time_groudTruth) > 0), "time_groudTruth must be strictly increasing."
    assert SE3_groudTruth.shape == (len(time_groudTruth), 4, 4), "SE3_groudTruth shape mismatch."

    # 初始化插值结果列表
    interp_results = []

    # 提取旋转矩阵并创建Slerp对象
    rotations = R.from_matrix(SE3_groudTruth[:, :3, :3])
    slerp = Slerp(time_groudTruth, rotations)

    for idx, t in enumerate(time_calc):
        # 处理边界
        t1_idx = np.searchsorted(time_groudTruth, t, side='right') - 1
        t1_idx = np.clip(t1_idx, 0, len(time_groudTruth)-2)  # 确保t2_idx有效
        t2_idx = t1_idx + 1

        t1, t2 = time_groudTruth[t1_idx], time_groudTruth[t2_idx]
        T1, T2 = SE3_groudTruth[t1_idx], SE3_groudTruth[t2_idx]

        # 平移插值
        if np.isscalar(t) and np.isscalar(t1) and np.isscalar(t2):
            alpha = (t - t1) / (t2 - t1) if t2 != t1 else 0.0
        else:
            raise ValueError("t, t1, 或 t2 不是标量值。")
        interp_translation = (1 - alpha) * T1[:3, 3] + alpha * T2[:3, 3]

        # 旋转插值
        interp_rotation = slerp(t).as_matrix()

        # 将插值结果填入SE(3)矩阵
        interp_SE3 = np.eye(4)
        interp_SE3[:3, :3] = interp_rotation
        interp_SE3[:3, 3] = interp_translation
        interp_results.append((t, interp_SE3))

    return interp_results

# ----------------------------- ROS服务封装 -----------------------------
class TransformService:
    def __init__(self):
        start_time = time.time()  # 记录初始化开始时间
        rospy.loginfo("Loading IE file, please wait...")
        # 参数从ROS参数服务器读取
        ie_path = rospy.get_param('~ie_path', '/media/zhao/ZhaoZhibo1T/AllData/tunnelRoadside/data_2025220163953/Result/Reference/IE.txt')
        # 下面这个坐标是从标定路侧ldiar的n系原点得到的，也就是路侧ldiar与高斯克吕格投影坐标标定时，n系的原点在ECEF框架下的坐标
        # 从UTM2WGS84.py文件中获取的，print(f"转换后的ECEF XYZ坐标:\n  X = {x_pyproj:.3f}米\n  Y = {y_pyproj:.3f}米\n  Z = {z_pyproj:.3f}米")
        ref_xyz = rospy.get_param('~ref_xyz', [-2251811.584, 5024460.661, 3208642.318])
        Ti0i1 = rospy.get_param('~Ti0i1', [
            [0.99888562,  0.02594205, -0.0394274,   0.03],
            [-0.02613833, 0.99964834, -0.00447079,   0.48],
            [0.03929756,  0.00549637,  0.99921244,   0.33],
            [0.0,         0.0,         0.0,          1.0]
        ])
        
        # 初始化插值器
        self.interpolator = TransformInterpolator(
            ie_path, 
            np.array(ref_xyz), 
            np.array(Ti0i1)
        )
        
        # 启动ROS服务
        self.service = rospy.Service(
            'transform_interp_service', 
            transformInterp, 
            self.handle_request
        )
        end_time = time.time()  # 记录初始化结束时间
        rospy.loginfo("Transform Interpolation Service Ready, Initialization took %.3f seconds", end_time - start_time)
    def handle_request(self, req):
        try:
            matrix = self.interpolator.get_transform(req.timestamp)
            response = transformInterpResponse()
            # 3. 正确转换数据类型 [!++]
            if matrix is not None:
                # 将numpy数组展平为一维列表（顺序需与客户端解析一致）
                response.matrix = matrix.flatten().tolist()  
            else:
                # 返回空列表保持结构正确
                response.matrix = []
            return response
        
        except Exception as e:
            rospy.logerr(f"Service error: {str(e)}")
            # 异常时仍需返回正确类型的响应 [!++]
            error_resp = transformInterpResponse()
            error_resp.matrix = np.eye(4).flatten().tolist()
            return error_resp

if __name__ == "__main__":
    rospy.init_node('transform_interpolation_server')
    server = TransformService()
    rospy.spin()