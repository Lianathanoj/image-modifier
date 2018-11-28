import numpy as np
from skimage import transform
from skimage import filters
import cv2


def determineClosestVLine(middle_pt, image):
	y_len, x_len, depth_len = image.shape
	quad_line = [x_len/3, x_len*2/3, x_len, 0]
	x_key = np.argmin(abs(np.asarray(quad_line) - middle_pt[1]))
	print (quad_line, middle_pt)
	return quad_line, quad_line[x_key], (middle_pt[1] < quad_line[1] and middle_pt[1] > quad_line[0])


def deleteLines(im, direction, mask, bbox, side):
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	mag = filters.sobel(gray)
	carved = transform.seam_carve(im, mag, 'vertical', 3)
	return (carved * 255).astype('uint8')
