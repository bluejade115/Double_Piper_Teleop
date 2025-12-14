from piper_sdk import *
import time
# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface_V2()
    piper.ConnectPort()
    piper.MotionCtrl_1(0x01,0,0)
    print("机械臂已重置，等到复位!!!!")
    time.sleep(2)
    piper.MotionCtrl_1(0x02,0,0)#恢复
    print("机械臂已恢复，等到使能!!!!")
    # 强制待机
    piper.ModeCtrl(ctrl_mode=0x00,move_mode=0x00) # ctrl_mode=0x01: CAN模式  move_mode=0x01:关节控制模式 0x00 点位控制
    time.sleep(0.1)
    print("已切换成待机模式")
    print(piper.GetArmStatus())
    time.sleep(0.1)
    # 设置从臂 无需使能
    piper.MasterSlaveConfig(0xFC, 0, 0, 0)
    time.sleep(0.1)
    print(piper.GetArmStatus())
    print("已设置从臂!!!!")
 