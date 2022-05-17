'''
pip install easyocr -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install easyocr

'''
import easyocr
import cv2

reader = easyocr.Reader(['ch_sim','en'])  # this needs to run only once to load the model into memory
img_bgr = cv2.imread('aaa.jpg')
result = reader.readtext(img_bgr)

print(result)
print(easyocr.__version__)
print()
