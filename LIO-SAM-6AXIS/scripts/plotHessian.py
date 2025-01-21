#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import signal
import sys
from matplotlib.ticker import ScalarFormatter

# 初始化全局变量
time_data = []
Sr_data = []
St_data = []
eigvals_rotation_data = [[] for _ in range(3)]
eigvals_translation_data = [[] for _ in range(3)]

# 更新动态绘图
def update_plot(frame):
    if len(time_data) > 1:
        ax1.clear()
        ax2.clear()
        ax3.clear()

        # 绘制 Sr 和 St  Degeneracy Factor
        ax1.plot(time_data, Sr_data, label="Rotation Degeneracy Factor", color='b')
        ax1.plot(time_data, St_data, label="Translation Degeneracy Factor", color='g')
        ax1.set_title("Degeneracy Factor vs. Time")
        ax1.set_ylabel("Degeneracy Factor")
        ax1.legend()
        ax1.grid(True)

        # 设置纵轴格式为实际数值
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax1.ticklabel_format(style='plain', axis='y')

        # 绘制旋转矩阵特征值
        colors = ['r', 'g', 'b']
        for i in range(3):
            ax2.plot(time_data, eigvals_rotation_data[i], label=f"Rotation Eigenvalue {i+1}", color=colors[i])
        ax2.set_title("Rotation Eigenvalues vs. Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Rotation Eigenvalues")
        ax2.legend()
        ax2.grid(True)

        # 设置纵轴格式为实际数值
        ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax2.ticklabel_format(style='plain', axis='y')

        # 绘制平移矩阵特征值
        for i in range(3):
            ax3.plot(time_data, eigvals_translation_data[i], label=f"Translation Eigenvalue {i+1}", color=colors[i], linestyle='--')
        ax3.set_title("Translation Eigenvalues vs. Time")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Translation Eigenvalues")
        ax3.legend()
        ax3.grid(True)

        # 设置纵轴格式为实际数值
        ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax3.ticklabel_format(style='plain', axis='y')

        plt.tight_layout()

# 处理接收到的矩阵消息
def hessian_matrix_callback(msg):
    # 获取矩阵行列信息
    rows = msg.layout.dim[0].size
    cols = msg.layout.dim[1].size
    timestamp_s = msg.layout.data_offset * 1e-4  # 时间戳转为秒

    # 将一维数组转换为二维矩阵
    matrix = np.array(msg.data).reshape((rows, cols))

    # 提取旋转矩阵（左上3x3）和平移矩阵（右下3x3）
    rotation_matrix = matrix[:3, :3]
    translation_matrix = matrix[3:6, 3:6]

    # 计算特征值和特征值比
    eigvals_rotation = np.linalg.eigvals(rotation_matrix)
    eigvals_translation = np.linalg.eigvals(translation_matrix)

    Sr = max(eigvals_rotation) / min(eigvals_rotation)
    St = max(eigvals_translation) / min(eigvals_translation)

    # 更新全局数据
    time_data.append(timestamp_s)
    Sr_data.append(Sr.real)  # 取实部
    St_data.append(St.real)  # 取实部
    for i in range(3):
        eigvals_rotation_data[i].append(eigvals_rotation[i].real)  # 取实部
        eigvals_translation_data[i].append(eigvals_translation[i].real)  # 取实部
    rospy.loginfo(f"Timestamp: {timestamp_s:.4f}s, Rotation Eigenvalues: {eigvals_rotation}, Translation Eigenvalues: {eigvals_translation}")


def hessian_matrix_listener():
    # 保持ROS节点运行
    rospy.spin()

def signal_handler(sig, frame):
    rospy.signal_shutdown('Shutting down')
    plt.close('all')
    sys.exit(0)

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('hessian_matrix_listener', anonymous=True)

    # 订阅话题
    rospy.Subscriber("lio_sam_6axis/mapping/hessian_matrix", Float32MultiArray, hessian_matrix_callback)

    rospy.loginfo("Listening to Hessian Matrix topic and plotting figures...")

    # 初始化动态绘图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    ani = FuncAnimation(fig, update_plot, interval=100)

    # 启动ROS监听线程
    listener_thread = threading.Thread(target=hessian_matrix_listener)
    listener_thread.start()

    # 捕获SIGINT信号（Ctrl+C）
    signal.signal(signal.SIGINT, signal_handler)

    # 启动Matplotlib的绘图
    plt.show()

    # 等待ROS监听线程结束
    listener_thread.join()