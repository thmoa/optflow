#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import optflow
import numpy as np


def color_code(flow, maxmag=10):
    x, y = flow[:, :, 0].astype(np.float32), flow[:, :, 1].astype(np.float32)
    magnitude, angle = cv2.cartToPolar(x, y, angleInDegrees=True)
    magnitude = np.clip(magnitude, 0, maxmag) / maxmag

    hsv = np.ones((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[:, :, 0] = angle
    hsv[:, :, 1] = magnitude
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


im0 = cv2.imread('examples/frame10.png')
im1 = cv2.imread('examples/frame11.png')

im0_g = cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY)
im1_g = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

f = optflow.brox(im0_g / 255., im1_g / 255.)
cv2.imshow('brox', color_code(f))

f = optflow.eppm(im0 / 255., im1 / 255.)
cv2.imshow('eppm', color_code(f))

cv2.waitKey(0)
