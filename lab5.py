# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:08:51 2024

@author: Vanya
"""

import sys
sys.path.append('../')
# %matplotlib inline
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

"""Загружаем изображение. Преобразуем в модели RGB и HSV"""
image = cv.imread('5.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)

"""Вывод изображения"""
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.imshow(image_rgb)
plt.subplot(1, 3, 3)
plt.imshow(image_hsv)
plt.show()

"""Распределение цветов в RGB"""
r, g, b = cv.split(image_rgb)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = image_rgb.reshape((np.shape(image_rgb)[0]*np.shape(image_rgb)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

"""Распределение компонент в HSV"""
h, s, v = cv.split(image_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

"""Диапозон тёмных тонов оливок"""
light1 = (0, 0, 30)
dark1 = (150, 150, 40)

"""Диапозон светлых тонов оливок"""
light2 = (0, 25, 35)
dark2 = (40, 110, 100)

"""Формирование масок и результирующей маски"""
mask1 = cv.inRange(image_hsv, light1, dark1)
mask2 = cv.inRange(image_hsv, light2, dark2)
mask = mask1 + mask2

"""Наложение маски на изображение"""
result = cv.bitwise_and(image_rgb, image_rgb, mask=mask)

plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(result)
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.show()
