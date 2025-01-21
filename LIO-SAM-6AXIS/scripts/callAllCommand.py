#!/usr/bin/env python
import subprocess
import time

def run_command(command):
    return subprocess.Popen(command, shell=True)

if __name__ == '__main__':
    # 启动 plotHessian.py
    plot_hessian_process = run_command("rosrun lio_sam_6axis plotHessian.py")

    # 启动 GREAT_oursdata_mems.launch
    launch_process = run_command("roslaunch lio_sam_6axis GREAT_oursdata_mems.launch")

    # 等待一段时间，确保前面的命令已经启动
    time.sleep(3)

    # 播放 rosbag
    rosbag_process = run_command("rosbag play /media/zhao/ZhaoZhibo/AllData/CalibrationData/2025_0116/2025_0116mems.bag -r 6 -s 130")

    # 等待所有进程结束
    plot_hessian_process.wait()
    launch_process.wait()
    rosbag_process.wait()