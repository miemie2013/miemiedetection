

use_gpu = False
use_gpu = True

import os
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 使用gpu
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 强制使用cpu执行神经网络运算

import numpy as np
import time
import autopy
import pyautogui
import win32gui
import win32api
import win32con
from tools import *
import datetime as dt

import shutil
import cv2


# 查找游戏窗口，返回窗口起始坐标
def find_flash_window(name):
    hwnd = win32gui.FindWindow(None, name)
    if (hwnd):
        win32gui.SetForegroundWindow(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        return rect
    return None



# 游戏分辨率调整为1280x720
print("finding...")
pos = find_flash_window("原神")
if(pos == None):
    print("unfound!")
    exit()
print("get!")
xbase = pos[0]
ybase = pos[1]
xend = pos[2]
yend = pos[3]
w = xend-xbase   # 游戏宽度
h = yend-ybase   # 游戏高度


dataset_name = 'yuanshen_wakuang'


position_name = 'mengde_wangfengshandi'


dir_name = 'D://%s/%s/'%(dataset_name, position_name)
if not os.path.exists(dir_name): os.mkdir(dir_name)


from pynput import keyboard
break_program = False
# 按下End键时停止程序
def on_press(key):
    global break_program
    print(key)
    if key == keyboard.Key.end:
        print('end pressed')
        break_program = True
        return False


with keyboard.Listener(on_press=on_press) as listener:
    print('采集图片...')
    while break_program == False:
        now = dt.datetime.now()
        img = pyautogui.screenshot(region=[xbase, ybase, xend - xbase, yend - ybase])  # x,y,w,h
        screen_shot = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        image_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        cv2.imwrite('D://%s/%s/%s.png'%(dataset_name, position_name, image_name), screen_shot)
        time.sleep(1.0)
    listener.join()



