'''
pip install easyocr -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install easyocr


手动下载预训练模型教程：
https://blog.csdn.net/juzicode00/article/details/122243330

下载craft_mlt_25k.zip、english_g2.zip、zh_sim_g2.zip，将3个压缩包移动到
C:\Users\yourname\.EasyOCR\model
其中yourname是登录用户名
解压3个压缩包到当前目录，得到craft_mlt_25k.pth、english_g2.pth、zh_sim_g2.pth
'''
import easyocr
import cv2

reader = easyocr.Reader(['ch_sim','en'])  # this needs to run only once to load the model into memory
img_bgr = cv2.imread('aaa.jpg')
result = reader.readtext(img_bgr)

print(result)
print(easyocr.__version__)
print()
