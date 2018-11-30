import numpy as np
from skimage import transform
from skimage import filters
import cv2
import matplotlib.pyplot as plt


def determineClosestVLine(middle_pt, image):
	y_len, x_len, depth_len = image.shape
	quad_line = [x_len/3, x_len*2/3, x_len, 0]
	x_key = np.argmin(abs(np.asarray(quad_line) - middle_pt[1]))
	return quad_line, quad_line[x_key], (middle_pt[1] < quad_line[1] and middle_pt[1] > quad_line[0])

def determineClosestHLine(middle_pt, image):
	y_len, x_len, depth_len = image.shape
	h_third_lines = [y_len/3, y_len*2/3, y_len, 0]
	y_key = np.argmin(abs(np.asarray(h_third_lines) - middle_pt[0]))
	isMiddle = middle_pt[0] < h_third_lines[1] and middle_pt[0] > h_third_lines[0]
	return h_third_lines, h_third_lines[y_key], isMiddle

def deleteLines(im, num, direction, mask, bbox, side):
	# Energy Map
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	mag = filters.sobel(gray)
	# print("dimensions", mag.shape, mask.shape)

	# Maniplate Energy Regions
	if side == 'up':
		mag[bbox[1][0]:, bbox[0][1]:bbox[1][1] + 1] = 1.0
	elif side == 'bottom':
		mag[:bbox[0][0], bbox[0][1]:bbox[1][1] + 1] = 1.0
	elif side == 'left':
		mag[bbox[0][0]:bbox[1][0] + 1, bbox[1][1]:] = 1.0
	else:
		mag[bbox[0][0]:bbox[1][0] + 1, 0:bbox[0][1]] = 1.0


	# Mask Energy Image
	mag[mask] = 1.0
	

	# Calculate New Mask
	if side == 'up':
		mask = mask[num:,:]
		bbox[0][0] -= num
		bbox[1][0] -= num
	elif side == 'bottom':
		mask = mask[:-num,:]
	elif side == 'left':
		mask = mask[:,num:]
		bbox[0][1] -= num
		bbox[1][1] -= num
	else:
		mask = mask[:,:-num]


	# New Carved Image3
	carved = transform.seam_carve(im, mag, direction, num)
	return (carved * 255).astype('uint8'), mask, bbox
