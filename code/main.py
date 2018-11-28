# # from matplotlib.widgets import Cursor
import numpy as np
import matplotlib.pyplot as plt
# # from scipy import misc as misc
from skimage import filters
import cv2
import seam_removing
import seam_adding
import create_mrcnn

if __name__ == "__main__":
	f, axarr = plt.subplots(1, 2)

	# Read in the image to modify
	im = plt.imread("images/outdoor.jpg", format='jpeg')

	# Run the MaskRCNN to detect objects
	model, class_names = create_mrcnn.create_model()
	results = model.detect([im], verbose=1)
	r = results[0]
	print (r.keys())

	for i in range(len(r['class_ids'])):
		# class_ids = 1 is the label for people
		if r['class_ids'][i] == 1:
			# get bounding box
			mask = r['masks'][:,:,i]
			bbox = r['rois'][i]
			topleft = [bbox[0], bbox[1]]
			topright = [bbox[0], bbox[3]]
			bottomleft = [bbox[2], bbox[1]]
			bottomright = [bbox[2], bbox[3]]
			print("mask", mask.shape)
			break

	axarr[0].imshow(im)

	# Calculate the center coordinates of the bounding box
	middle_pt = [(topleft[0]+bottomleft[0])/2, (topright[1]+topleft[1])/2]

	# Determine closest vertical line
	# quad[0] is left line, quad[1] is right line, 
	# quad[2] is the right edge, quad[3] is left edge
	# line is the vertical line thats closest
	quad, line, isMiddle = seam_removing.determineClosestVLine(middle_pt, im)
	h_thirds, h_line, isHMiddle = seam_removing.determineClosestHLine(middle_pt, im)

	bbox = np.zeros((2,2))
	bbox[0] = topleft
	bbox[1] = bottomright
	bbox = bbox.astype('int')

	axarr[0].axvline(x=line)
	axarr[0].axhline(y=h_thirds[0])
	axarr[0].axhline(y=h_thirds[1])
	# if line is quad[2]:
	# 	axarr[0].axvline(x=quad[1])
	# elif line is quad[3]:
	# 	axarr[0].axvline(x=quad[0])
	# else:
	# 	axarr[0].axvline(x=line)

	if isMiddle:
		print("Object in middle")
		# closest line is left line, delete from the left
		if line is quad[0]:
			while abs(middle_pt[1] - line) > 5:
				im, mask, bbox = seam_removing.deleteLines(im, 3, 'vertical', mask, bbox, 'left')
				line -= 1
				middle_pt[1] -= 3
		# closest line is right line, delete from the right
		elif line is quad[1]:
			while abs(middle_pt[1] - line) > 5:
				im, mask, bbox = seam_removing.deleteLines(im, 3, 'vertical', mask, bbox, 'right')
				line -= 2
				# middle_pt[1] -= 3
	else:
		# closest edge is right edge
		if line is quad[2]:
			print("hi")
		#closest edge is left edge
		elif line is quad[3]:
			print("bye")

	axarr[0].set_title('Original')
	axarr[1].imshow(im)
	axarr[1].set_title('Modified')
	axarr[1].axvline(x=line)

	plt.show()

	exit()

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
			segmentIm, boundingIm = seam_removing.segmentImage(im, topleft, topright, "right")
			segmentIm = seam_removing.deleteLines(segmentIm)
			im = seam_removing.combine(boundingIm, segmentIm)
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

			segmentIm, boundingIm = seam_removing.segmentImage(im, topleft, topright, "left")
			segmentIm = seam_removing.deleteLines(segmentIm)

			im = seam_removing.combine(segmentIm, boundingIm)
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
			segmentIm, boundingIm = seam_removing.segmentImage(im, topleft, topright, "right")

			segmentIm, img_temp = seam_adding.addLines(segmentIm, img_temp)

			im = seam_removing.combine(boundingIm, segmentIm)

			if line > im.shape[1]/2:
				line += 2
			else:
				line += 1
			if img_temp.shape[1] <= 5 and not onepass:
				img_temp = None
				onepass = True
			#bounding and midpoint no change

		while abs(middle_pt[1] - line) > 5:
			print ("delete on left side after")

			segmentIm, boundingIm = seam_removing.segmentImage(im, topleft, topright, "left")
			segmentIm = seam_removing.deleteLines(segmentIm)

			im = seam_removing.combine(segmentIm, boundingIm)

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

	else:
		img_temp = None
		line = quad[0]
		while abs(middle_pt[1] - line) > 10 and (img_temp is None or img_temp.shape[1] > 5):
			## regen middle point
			print ("generate on left")
			segmentIm, boundingIm = seam_removing.segmentImage(im, topleft, topright, "left")
			segmentIm, img_temp = seam_adding.addLines(segmentIm, img_temp)

			im = seam_removing.combine(segmentIm, boundingIm)

			if line > im.shape[1]/2:
				line += 2
			else:
				line += 1

			topleft[1] +=3
			topright[1] += 3
			bottomleft[1] += 3
			bottomleft[1] += 3

			middle_pt[1] += 3


		while abs(middle_pt[1] - line) > 5:
			# print ("delete on right side")

			segmentIm, boundingIm = seam_removing.segmentImage(im, topleft, topright, "right")
			segmentIm = seam_removing.deleteLines(segmentIm)
			im = seam_removing.combine(boundingIm, segmentIm)
			im = im.astype('uint8')

			if line > im.shape[1]/2:
				line -= 2
			else:
				line -= 1
			## bounding and midpoint no change
	axarr[0].set_title('Original')
	axarr[1].imshow(im)
	axarr[1].set_title('Modified')
	axarr[1].axvline(x=line)

	plt.show()




