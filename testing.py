# # from matplotlib.widgets import Cursor
import numpy as np
import matplotlib.pyplot as plt
# # from scipy import misc as misc

import rule

f, axarr = plt.subplots(1, 2)

im = plt.imread("butterfly.jpg", format='jpeg')
im = im.astype('uint8')
print (im.shape)
topleft = [10, 400]
topright = [10, 550]
bottomleft = [20, 400]
bottomright = [20, 550]

axarr[0].imshow(im)

middle_pt = [(topleft[0]+bottomleft[0])/2, (topright[1]+topleft[1])/2]

# Determine closest vertical line
quad, line = rule.determineClosestVLine(middle_pt, im)
# print (quad)
# print (line)
axarr[0].axvline(x=line)

# for x in quad:
# 	axarr[0].axvline(x=x)
# print (im)
i = 0

if line == quad[0]:
	# while middle point not near the vertical line, do operation and recalculate stuff
	while abs(middle_pt[1] - line) > 5:
		# print ("delete on right side")

		segmentIm, boundingIm = rule.segmentImage(im, topleft, topright, "right")
		segmentIm = rule.deleteLines(segmentIm)
		im = rule.combine(boundingIm, segmentIm)
		im = im.astype('uint8')

		line -= 1
		## bounding and midpoint no change

elif line == quad[1]:
	while abs(middle_pt[1] - line) > 5:
		print ("delete on left side")
		print (i)

		segmentIm, boundingIm = rule.segmentImage(im, topleft, topright, "left")
		segmentIm = rule.deleteLines(segmentIm)

		if i in range (7, 9):
			print ("SEGMENT")
			print (segmentIm)
		im = rule.combine(segmentIm.astype('uint8'), boundingIm)
		if i in range (7, 9):
			print (im)
		im = im.astype('uint8')
		if i in range (7, 9):
			print ("AFTER")
			print (im)

		line -= 1
		## bounding and midpoint change:
		topleft[1] -= 3
		topright[1] -= 3
		bottomleft[1] -= 3
		bottomright[1] -= 3

		middle_pt[1] -= 3
		i+= 1

		# print (im.shape)




elif line == quad[2]:
	while abs(middle_pt[1] - line) > 5:
		# regen middle point
		print ("generate on right")
else:
	while abs(middle_pt[1] - line) > 5:
		## regen middle point
		print ("generate on left")


# im = im.astype('uint8')
# axarr[1].imshow(im)
# print (im)
# Else remove 



# # ## Script used to run all code when writing, uncomment for specific functions.

# f, axarr = plt.subplots(1, 2)
# origImg = plt.imread("fish.jpg", format='jpeg')
# origImg = origImg.astype('uint8')

# # img_1_size = origImg.shape

# # Image to show
# axarr[0].imshow(origImg)

# k=5

# # #1
# # outputImg, meanColors = quantizeRGB.quantizeRGB(origImg, k)
# # print (outputImg)
# # print (meanColors)
# # axarr[1].imshow(outputImg)

# #2
# # outputImg, meanHues = quantizeHSV.quantizeHSV(origImg, k)
# # axarr[1].imshow(outputImg)

# #3
# # error = computeQuantizationError.computeQuantizationError(origImg,outputImg)
# # print ("error=", error)

# #4
# histEqual, histClustered = getHueHists.getHueHists(origImg, k)
# # axarr[1].show(hist)

plt.show()



