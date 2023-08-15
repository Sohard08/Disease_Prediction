import cv2 as cv
import numpy as np

img = cv.imread(
    'spider-man-tom-holland-tobey-maguire-andrew-garfield-1645636899_1646814400124_1646814415063.webp')

resize = cv.resize(img, (480, 360), interpolation=cv.INTER_AREA)


gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)

blurred = cv.medianBlur(gray, 3)
cv.imshow('gb', blurred)

edge = cv.adaptiveThreshold(
    blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 3)
cv.imshow('edges', edge)

reblur = cv.medianBlur(resize, 5)
cv.imshow('nb', reblur)

noise = cv.bilateralFilter(reblur, 15, 75, 75)
cv.imshow('noised', noise)

k = np.ones((1, 1), np.uint8)
erode = cv.erode(noise, k, iterations=3)
dilate = cv.dilate(erode, k, iterations=3)
cv.imshow('dilated', dilate)

ci = cv.stylization(resize, sigma_s=150, sigma_r=0.25)
cv.imshow('resize', resize)
cv.imshow('cartoon', ci)

cv.waitKey(0)
