#!/usr/bin/env python
import subprocess
import time

def run_command(command):
    return subprocess.Popen(command, shell=True)

if __name__ == '__main__':
    # 启动 plotHessian.py 绘制海塞矩阵特征值的动态图
    # plot_hessian_process = run_command("rosrun lio_sam_6axis plotHessian.py")

    # #播放自己采集的硃山路隧道的rosbag包
    # 启动 GREAT_oursdata_mems.launch
    launch_process = run_command("roslaunch lio_sam_6axis GREAT_oursdata_mems.launch")
    time.sleep(3) # 等待一段时间，确保前面的命令已经启动
    # 第一次采集数据的rosbag包，和郭紫祎一起采集的数据，硃山路隧道，春节前采集的
    # # rosbag_process = run_command("rosbag play /media/zhao/ZhaoZhibo1T/AllData/tunnelRoadside/data_2025220163953/AfterPreProcess/mergeVehicleRoad_withGNSS.bag\
    # #                              -r 3  -s 450")

    # 第二次采集数据的rosbag包，和许思，袁诗睿一起采集的数据
    rosbag_process = run_command("rosbag play /media/zhao/ZhaoZhibo1T/AllData/tunnelRoadside/data_2025220163953/AfterPreProcess/AfterPreProcess_mems.bag\
                                  -r 3 -s 450 ")

    # # # 播放开源数据集中街道口隧道的包
    # # 启动 GREAT_oursdata_mems.launch
    # launch_process = run_command("roslaunch lio_sam_6axis GREAT_urban02.launch")
    # time.sleep(3)     # 等待一段时间，确保前面的命令已经启动
    # rosbag_process = run_command("rosbag play /media/zhao/ZhaoZhibo1/AllData/GREAT/urban-02/urban-02tunnel.bag\
    #                               -r 5")

    # 等待所有进程结束
    # plot_hessian_process.wait()
    launch_process.wait()
    # rosbag_process.wait()