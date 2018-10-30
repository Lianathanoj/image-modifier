import numpy as np
from skimage import transform
from skimage import filters
import cv2


def determineClosestVLine(middle_pt, image):
	# print middle_pt
	y_len, x_len, depth_len = image.shape
	quad_line = [x_len/3, x_len*2/3, x_len, 0]
	x_key = np.argmin(abs(np.asarray(quad_line) - middle_pt[1]))
	return quad_line, quad_line[x_key]

def combine(leftIm, rightIm):
	return np.hstack((leftIm, rightIm))



def segmentImage(im, topleft, topright, direction):
	if direction == "left":
		return im[:, 0:topleft[1], :], im[:, topleft[1]:, :]
	else: 
		return im[:, topright[1]:, :], im[:, 0:topright[1], :]


def deleteLines(im):
	
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	mag = filters.sobel(gray.astype("float"))
	carved = transform.seam_carve(im, mag, 'vertical', 3)
	# print (carved)
	return (carved * 255).astype('uint8')
	## segment image
	## delete from segment
	## append back to image
	## yay.

def generateLines():
	print ("generate")


