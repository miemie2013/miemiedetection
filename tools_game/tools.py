



import os
import shutil
import cv2

import cv2
import shutil
import numpy as np
import time
import pyautogui
import win32gui
import win32api
import win32con
import datetime as dt



# 查找游戏窗口，返回窗口起始坐标
def find_flash_window(name):
    hwnd = win32gui.FindWindow(None, name)
    if (hwnd):
        win32gui.SetForegroundWindow(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        return rect
    return None


# 模拟鼠标点击
def mouse_click(x, y):
    x = int(x)
    y = int(y)
    win32api.SetCursorPos([x, y])
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.01)

# 模拟鼠标长按
def mouse_long_press(x, y, t):
    x = int(x)
    y = int(y)
    win32api.SetCursorPos([x, y])
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(t)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.01)


def get_key_value(key):
    v = 0
    if key == 'w':
        v = 87
    elif key == 'a':
        v = 65
    elif key == 's':
        v = 83
    elif key == 'd':
        v = 68
    elif key == '->':
        v = 39
    elif key == '<-':
        v = 37
    elif key == 'up':
        v = 38
    elif key == 'down':
        v = 40
    return v


# 模拟键盘长按
def key_long_press(key, t=0):
    v = get_key_value(key)
    win32api.keybd_event(v, 0, 0, 0)  # 按下
    if t > 0:
        time.sleep(t)
    win32api.keybd_event(v, 0, win32con.KEYEVENTF_KEYUP, 0)  # 抬起
    time.sleep(0.01)

# 模拟键盘按下
def key_press(key):
    v = get_key_value(key)
    win32api.keybd_event(v, 0, 0, 0)  # 按下
    time.sleep(0.01)

# 模拟键盘松开
def key_release(key):
    v = get_key_value(key)
    win32api.keybd_event(v, 0, win32con.KEYEVENTF_KEYUP, 0)  # 抬起
    time.sleep(0.01)


# 当前位置
def where_is_here_letu(data):
    here = 'unknow'

    # 用来确定是否在主界面
    have_renwu = 0
    have_huodong = 0
    have_fuli = 0
    have_youjian = 0
    have_gonggao = 0
    have_shancheng = 0
    have_jiayuan = 0

    # 用来确定是否在乐土主界面
    have_letuzjm1 = 0
    have_letuzjm2 = 0
    have_letuzjm3 = 0
    have_letuzjm4 = 0

    # 用来确定是否在战斗准备界面
    have_zhandouzhunbei1 = 0
    have_zhandouzhunbei2 = 0

    # 用来确定是否在战斗准备界面2
    have_zhandouzhunbei21 = 0
    have_zhandouzhunbei22 = 0
    have_zhandouzhunbei23 = 0
    have_zhandouzhunbei24 = 0

    # 用来确定是否在继续上次界面
    have_jixushangci1 = 0
    have_jixushangci2 = 0

    # 用来确定是否在个人信息页面
    have_gerenxinxi1 = 0
    have_gerenxinxi2 = 0
    have_gerenxinxi3 = 0

    # 用来确定是否在加载页面
    have_jiazai1 = 0
    have_jiazai2 = 0

    # 用来确定是否在对话页面
    have_duihua1 = 0
    have_duihua2 = 0
    have_duihua3 = 0
    have_duihua4 = 0

    # 用来确定是否在选择刻印页面
    have_xuanzekeyin1 = 0
    have_xuanzekeyin2 = 0
    have_xuanzekeyin3 = 0

    # 用来确定是否在战斗页面
    have_zhandou1 = 0
    have_zhandou2 = 0

    # 用来确定是否在乐土结算页面
    have_letujs1 = 0

    # 用来确定是否在乐土退出结算页面
    have_letutcjs1 = 0
    have_letutcjs2 = 0
    have_letutcjs3 = 0

    # 用来确定是否在菲莉丝商店
    have_feilisishangdian1 = 0
    have_feilisishangdian2 = 0
    have_feilisishangdian3 = 0


    for d in data:
        text = d[1]
        text_box_position = d[0]
        x0 = text_box_position[0][0]
        y0 = text_box_position[0][1]
        x1 = text_box_position[2][0]
        y1 = text_box_position[2][1]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        if '任务' in text:
            if 100 < cx and cx < 145 and 145 < cy and cy < 169:
                have_renwu = 1
        if '活动' in text:
            if 147 < cx and cx < 183 and 217 < cy and cy < 238:
                have_huodong = 1
        if '福利' in text:
            if 100 < cx and cx < 145 and 281 < cy and cy < 308:
                have_fuli = 1
        if '邮件' in text:
            if 147 < cx and cx < 183 and 353 < cy and cy < 373:
                have_youjian = 1
        if '公告' in text:
            if 100 < cx and cx < 145 and 421 < cy and cy < 441:
                have_gonggao = 1
        if '商城' in text:
            if 147 < cx and cx < 183 and 491 < cy and cy < 513:
                have_shancheng = 1
        if '家园' in text:
            if 1015 < cx and cx < 1110 and 658 < cy and cy < 690:
                have_jiayuan = 1

        if '命定的' in text:
            if 1167 < cx and cx < 1257 and 271 < cy and cy < 295:
                have_letuzjm1 = 1
        if '装甲强化' in text:
            if 1176 < cx and cx < 1250 and 362 < cy and cy < 386:
                have_letuzjm2 = 1
        if '追忆之' in text:
            if 1174 < cx and cx < 1248 and 452 < cy and cy < 476:
                have_letuzjm3 = 1
        if '快速出击' in text:
            if 1134 < cx and cx < 1222 and 690 < cy and cy < 720:
                have_letuzjm4 = 1

        if '出战位' in text:
            if 96 < cx and cx < 165 and 178 < cy and cy < 205:
                have_zhandouzhunbei1 = 1
        if '支援位' in text:
            if 457 < cx and cx < 526 and 178 < cy and cy < 205:
                have_zhandouzhunbei2 = 1

        if '安全' in text:
            if 100 < cx and cx < 250 and 228 < cy and cy < 257:
                have_zhandouzhunbei21 = 1
        if '普通' in text:
            if 100 < cx and cx < 250 and 310 < cy and cy < 337:
                have_zhandouzhunbei22 = 1
        if '危险' in text:
            if 100 < cx and cx < 250 and 394 < cy and cy < 420:
                have_zhandouzhunbei23 = 1
        if '真实' in text:
            if 100 < cx and cx < 250 and 470 < cy and cy < 500:
                have_zhandouzhunbei24 = 1

        if '直接结算' in text:
            if 669 < cx and cx < 756 and 541 < cy and cy < 566:
                have_jixushangci1 = 1
        if '继续挑战' in text:
            if 947 < cx and cx < 1036 and 541 < cy and cy < 566:
                have_jixushangci2 = 1

        if '圣痕馆' in text:
            if 673 < cx and cx < 787 and 636 < cy and cy < 667:
                have_gerenxinxi1 = 1
        if '勋章馆' in text:
            if 888 < cx and cx < 1000 and 636 < cy and cy < 667:
                have_gerenxinxi2 = 1
        if '获得成就' in text:
            if 1102 < cx and cx < 1241 and 636 < cy and cy < 667:
                have_gerenxinxi3 = 1

        if 'Now' in text:
            if 988 < cx and cx < 1208 and 677 < cy and cy < 715:
                have_jiazai1 = 1
        if 'Load' in text:
            if 988 < cx and cx < 1208 and 677 < cy and cy < 715:
                have_jiazai2 = 1

        if '隐藏' in text:
            if 61 < cx and cx < 113 and 51 < cy and cy < 78:
                have_duihua1 = 1
        if '历史' in text:
            if 198 < cx and cx < 289 and 51 < cy and cy < 78:
                have_duihua2 = 1
        if '自动' in text:
            if 1032 < cx and cx < 1122 and 51 < cy and cy < 78:
                have_duihua3 = 1
        if '跳过' in text:
            if 1158 < cx and cx < 1247 and 51 < cy and cy < 78:
                have_duihua4 = 1

        if '刻印' in text:
            if 592 < cx and cx < 780 and 121 < cy and cy < 156:
                have_xuanzekeyin1 = 1
        if '刻印' in text:
            if 837 < cx and cx < 936 and 681 < cy and cy < 710:
                have_xuanzekeyin2 = 1
        if '刻印' in text:
            if 1090 < cx and cx < 1188 and 681 < cy and cy < 710:
                have_xuanzekeyin3 = 1

        if '层数' in text:
            if 139 < cx and cx < 253 and 38 < cy and cy < 58:
                have_zhandou1 = 1
        if '得分' in text:
            if 80 < cx and cx < 131 and 112 < cy and cy < 138:
                have_zhandou2 = 1

        if '挑战' in text:
            if 790 < cx and cx < 905 and 388 < cy and cy < 425:
                have_letujs1 = 1

        if '层数' in text:
            if 584 < cx and cx < 676 and 220 < cy and cy < 250:
                have_letutcjs1 = 1
        if '难度' in text:
            if 797 < cx and cx < 892 and 220 < cy and cy < 250:
                have_letutcjs2 = 1
        if '评价' in text:
            if 1012 < cx and cx < 1102 and 220 < cy and cy < 250:
                have_letutcjs3 = 1

        if '商店' in text:
            if 48 < cx and cx < 262 and 103 < cy and cy < 137:
                have_feilisishangdian1 = 1
        if '购买' in text:
            if 772 < cx and cx < 917 and 111 < cy and cy < 140:
                have_feilisishangdian2 = 1
        if '升级' in text:
            if 1038 < cx and cx < 1177 and 111 < cy and cy < 140:
                have_feilisishangdian3 = 1




    if (have_renwu + have_huodong + have_fuli + have_youjian + have_gonggao + have_shancheng + have_jiayuan) > 2:
        here = 'zhujiemian'
    if (have_letuzjm1 + have_letuzjm2 + have_letuzjm3 + have_letuzjm4) > 2:
        here = 'letuzhujiemian'
    if (have_zhandouzhunbei1 + have_zhandouzhunbei2) > 0:
        here = 'zhandouzhunbei'
    if (have_zhandouzhunbei21 + have_zhandouzhunbei22 + have_zhandouzhunbei23 + have_zhandouzhunbei24) > 2:
        here = 'zhandouzhunbei2'
    if (have_jixushangci1 + have_jixushangci2) > 0:
        here = 'jixushangci'
    if (have_duihua1 + have_duihua2 + have_duihua3 + have_duihua4) > 2:
        here = 'duihua'
    if (have_gerenxinxi1 + have_gerenxinxi2 + have_gerenxinxi3) > 1:
        here = 'gerenxinxi'
    if (have_xuanzekeyin1 + have_xuanzekeyin2 + have_xuanzekeyin3) > 1:
        here = 'xuanzekeyin'
    if (have_zhandou1 + have_zhandou2) > 0:
        here = 'zhandou'
    if (have_jiazai1 + have_jiazai2) > 0:
        here = 'jiazai'
    if (have_letujs1 + 0) > 0:
        here = 'letu_jiesuan'
    if (have_letutcjs1 + have_letutcjs2 + have_letutcjs3) > 1:
        here = 'letu_tcjiesuan'
    if (have_feilisishangdian1 + have_feilisishangdian2 + have_feilisishangdian3) > 1:
        here = 'feilisishangdian'
    return here, data


# 当前位置
def where_is_here_weituo(result):
    data = result['data']
    here = 'unknow'

    # 用来确定是否在主界面
    have_renwu = 0
    have_huodong = 0
    have_fuli = 0
    have_youjian = 0
    have_gonggao = 0
    have_shancheng = 0
    have_jiayuan = 0

    # 用来确定是否在委托主界面
    have_weituozjm1 = 0
    have_weituozjm2 = 0
    have_weituozjm3 = 0
    have_weituozjm4 = 0

    # 用来确定是否在个人信息页面
    have_gerenxinxi1 = 0
    have_gerenxinxi2 = 0
    have_gerenxinxi3 = 0

    # 用来确定是否在加载页面
    have_jiazai1 = 0
    have_jiazai2 = 0

    # 用来确定是否在战斗页面
    have_zhandou1 = 0
    have_zhandou2 = 0


    for d in data:
        text = d['text']
        text_box_position = d['text_box_position']
        x0 = text_box_position[0][0]
        y0 = text_box_position[0][1]
        x1 = text_box_position[2][0]
        y1 = text_box_position[2][1]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        if '任务' in text:
            if 100 < cx and cx < 145 and 145 < cy and cy < 169:
                have_renwu = 1
        if '活动' in text:
            if 147 < cx and cx < 183 and 217 < cy and cy < 238:
                have_huodong = 1
        if '福利' in text:
            if 100 < cx and cx < 145 and 281 < cy and cy < 308:
                have_fuli = 1
        if '邮件' in text:
            if 147 < cx and cx < 183 and 353 < cy and cy < 373:
                have_youjian = 1
        if '公告' in text:
            if 100 < cx and cx < 145 and 421 < cy and cy < 441:
                have_gonggao = 1
        if '商城' in text:
            if 147 < cx and cx < 183 and 491 < cy and cy < 513:
                have_shancheng = 1
        if '家园' in text:
            if 1015 < cx and cx < 1110 and 658 < cy and cy < 690:
                have_jiayuan = 1

        if '委托重置时间' in text:
            if 1134 < cx and cx < 1255 and 152 < cy and cy < 172:
                have_weituozjm1 = 1
        if '剧情进度' in text:
            if 1153 < cx and cx < 1235 and 405 < cy and cy < 430:
                have_weituozjm2 = 1
        if '地图选择' in text:
            if 742 < cx and cx < 869 and 668 < cy and cy < 699:
                have_weituozjm3 = 1
        if '开始冒险' in text:
            if 1048 < cx and cx < 1177 and 668 < cy and cy < 699:
                have_weituozjm4 = 1

        if '圣痕馆' in text:
            if 673 < cx and cx < 787 and 636 < cy and cy < 667:
                have_gerenxinxi1 = 1
        if '勋章馆' in text:
            if 888 < cx and cx < 1000 and 636 < cy and cy < 667:
                have_gerenxinxi2 = 1
        if '获得成就' in text:
            if 1102 < cx and cx < 1241 and 636 < cy and cy < 667:
                have_gerenxinxi3 = 1

        if 'Now' in text:
            if 988 < cx and cx < 1208 and 677 < cy and cy < 715:
                have_jiazai1 = 1
        if 'Load' in text:
            if 988 < cx and cx < 1208 and 677 < cy and cy < 715:
                have_jiazai2 = 1

        if '0' in text or '1' in text or '2' in text or '3' in text or '4' in text or '5' in text or '6' in text or '7' in text or '8' in text or '9' in text:
            if 377 < cx and cx < 468 and 667 < cy and cy < 694:
                have_zhandou1 = 1
        if '0' in text or '1' in text or '2' in text or '3' in text or '4' in text or '5' in text or '6' in text or '7' in text or '8' in text or '9' in text:
            if 771 < cx and cx < 833 and 718 < cy and cy < 738:
                have_zhandou2 = 1




    if (have_renwu + have_huodong + have_fuli + have_youjian + have_gonggao + have_shancheng + have_jiayuan) > 2:
        here = 'zhujiemian'
    if (have_weituozjm1 + have_weituozjm2 + have_weituozjm3 + have_weituozjm4) > 2:
        here = 'weituozhujiemian'
    if (have_gerenxinxi1 + have_gerenxinxi2 + have_gerenxinxi3) > 1:
        here = 'gerenxinxi'
    if (have_zhandou1 + have_zhandou2) > 0:
        here = 'zhandou'
    if (have_jiazai1 + have_jiazai2) > 0:
        here = 'jiazai'
    return here, data


def getText(easyocr_reader, xbase, ybase, w, h):
    # 截图识别字符
    img_rgb = pyautogui.screenshot(region=[xbase, ybase, w, h])
    img_rgb = np.asarray(img_rgb)
    img_bgr = img_rgb[:, :, [2, 1, 0]]
    results = easyocr_reader.readtext(img_bgr)
    return results

def getObjs(dir_name, xbase, xend, ybase, yend, _decode, draw_image, draw_thresh):
    # 截图识别物体
    now = dt.datetime.now()
    fname = now.strftime("%Y-%m-%d_%H-%M-%S")
    autopy.bitmap.capture_screen().save('D://%s/temp.png' % (dir_name,))
    img = cv2.imread('D://%s/temp.png' % (dir_name,))
    screen_shot = img[ybase:yend, xbase:xend, ...]
    image = screen_shot
    pimage, im_size = _decode.process_image(np.copy(image))
    image, boxes, scores, classes = _decode.detect_image(image, pimage, im_size, draw_image,
                                                         draw_thresh)
    return image, boxes, scores, classes



