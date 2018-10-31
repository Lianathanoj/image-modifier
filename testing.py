# # from matplotlib.widgets import Cursor
import numpy as np
import matplotlib.pyplot as plt
# # from scipy import misc as misc
from skimage import filters
import cv2
import rule
import seam_adding

f, axarr = plt.subplots(1, 2)

im = plt.imread("images/railroad_girl.jpg", format='jpeg')
print (im.shape)
# im = im.astype('uint8')
# print (im.shape)
topleft = [56, 870]
topright = [56, 1144]
bottomleft = [644, 870]
bottomright = [644, 1144]

axarr[0].imshow(im)

middle_pt = [(topleft[0]+bottomleft[0])/2, (topright[1]+topleft[1])/2]

# Determine closest vertical line
quad, line, isMiddle = rule.determineClosestVLine(middle_pt, im)

axarr[0].axvline(x=line)
print (isMiddle, line)
if isMiddle:
	if line is quad[0]:
		segmentingLine = quad[1]
	else:
		segmentingLine = quad[0]
else:
	segmentingLine = line

print (im.shape[1], line)

if segmentingLine == quad[0]:
	# while middle point not near the vertical line, do operation and recalculate stuff
	while abs(middle_pt[1] - line) > 5:
		print ("delete on right side")
		print ((middle_pt[1] - line))
		segmentIm, boundingIm = rule.segmentImage(im, topleft, topright, "right")
		segmentIm = rule.deleteLines(segmentIm)
		im = rule.combine(boundingIm, segmentIm)
		im = im.astype('uint8')
		if line > im.shape[1]/2:
			line -= 2
		else:
			line -=1
		print (im.shape[1], line)

		## bounding and midpoint no change
		if segmentIm.shape[1] < 5:
			break

elif segmentingLine == quad[1]:
	while abs(middle_pt[1] - line) > 5:
		print ("delete on left side")

		segmentIm, boundingIm = rule.segmentImage(im, topleft, topright, "left")
		segmentIm = rule.deleteLines(segmentIm)

		im = rule.combine(segmentIm, boundingIm)
		if line > im.shape[1]/2:
			line -= 2
		else:
			line -= 1
		## bounding and midpoint change:
		topleft[1] -= 3
		topright[1] -= 3
		bottomleft[1] -= 3
		bottomright[1] -= 3

		middle_pt[1] -= 3

		if segmentIm.shape[1] < 5:
			break

elif segmentingLine == quad[2]:
	img_temp = None
	line = quad[1]
	onepass = False
	while abs(middle_pt[1] - line) > 10 and (img_temp is None or img_temp.shape[1] > 5):
		# regen middle point
		print ("generate on right")
		print (middle_pt[1] - line)
		segmentIm, boundingIm = rule.segmentImage(im, topleft, topright, "right")

		segmentIm, img_temp = seam_adding.addLines(segmentIm, img_temp)

		im = rule.combine(boundingIm, segmentIm)

		line += 1
		if img_temp.shape[1] <= 5 and not onepass:
			img_temp = None
			onepass = True
		#bounding and midpoint no change

	while abs(middle_pt[1] - line) > 5:
		print ("delete on left side after")

		segmentIm, boundingIm = rule.segmentImage(im, topleft, topright, "left")
		segmentIm = rule.deleteLines(segmentIm)

		im = rule.combine(segmentIm, boundingIm)

		line -= 1
		## bounding and midpoint change:
		topleft[1] -= 3
		topright[1] -= 3
		bottomleft[1] -= 3
		bottomright[1] -= 3

		middle_pt[1] -= 3

else:
	img_temp = None
	line = quad[0]
	while abs(middle_pt[1] - line) > 10 and (img_temp is None or img_temp.shape[1] > 5):
		## regen middle point
		print ("generate on left")
		segmentIm, boundingIm = rule.segmentImage(im, topleft, topright, "left")
		segmentIm, img_temp = seam_adding.addLines(segmentIm, img_temp)

		im = rule.combine(segmentIm, boundingIm)

		line += 1

		topleft[1] +=3
		topright[1] += 3
		bottomleft[1] += 3
		bottomleft[1] += 3

		middle_pt[1] += 3


	while abs(middle_pt[1] - line) > 5:
		# print ("delete on right side")

		segmentIm, boundingIm = rule.segmentImage(im, topleft, topright, "right")
		segmentIm = rule.deleteLines(segmentIm)
		im = rule.combine(boundingIm, segmentIm)
		im = im.astype('uint8')

		line -= 1
		## bounding and midpoint no change
axarr[1].imshow(im)
axarr[1].axvline(x=line)

plt.show()




