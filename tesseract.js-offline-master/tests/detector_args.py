import cv2 as cv
import numpy as np
import sys

def main(sys.argv[1:])
file = '/home/sagnik/Vs Code/maphe.github.io/tesseract.js-offline-master/images/shots/left_hidden,decrease_stat,some_other_true.jpg'
original_im = cv.imread(file, 0)
img_rgb = cv.imread(file)
print(img_rgb.shape)
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
templatel = cv.imread('/home/sagnik/Vs Code/maphe.github.io/tesseract.js-offline-master/images/templates/top_left.jpg', 0)
templater = cv.imread('/home/sagnik/Vs Code/maphe.github.io/tesseract.js-offline-master/images/templates/top_right.jpg', 0)
wi, hi = original_im.shape[::-1]
wl, hl = templatel.shape[::-1]
wr, hr = templater.shape[::-1]
resl = cv.matchTemplate(img_gray, templatel, cv.TM_CCOEFF_NORMED)
resr = cv.matchTemplate(img_gray, templater, cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc_l = np.where(resl >= threshold)
loc_r = np.where(resr >= threshold)