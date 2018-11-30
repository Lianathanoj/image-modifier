# # from matplotlib.widgets import Cursor
import numpy as np
import matplotlib.pyplot as plt
# # from scipy import misc as misc
from skimage import filters
from scipy import ndimage
import cv2
import seam_removing
import seam_adding
import create_mrcnn
from PIL import Image

if __name__ == "__main__":

	# Read in the image to modify
	im = plt.imread("images/teddybear.jpg", format='jpeg')

	# Size Image
	while im.shape[0] > 1000 or im.shape[1] > 1000:
		im = cv2.resize(im, (0, 0), fx=.55, fy=0.55)
	im_copy = im.copy()

	# Objects that we care about
	important_classes = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
		           'bus', 'train', 'boat', 'fire hydrant', 'bird', 'cat', 'dog',
				   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
		           'wine glass', 'banana', 'apple', 'orange', 'cake', 'potted plant',
		           'book', 'clock', 'vase', 'teddy bear']

	# Run the MaskRCNN to detect objects
	model, class_names = create_mrcnn.create_model()
	results = model.detect([im], verbose=1)
	r = results[0]
	print(r['class_ids'])
	print (r.keys())

	for i in range(len(r['class_ids'])): 


		im = im_copy
		# class_ids = 1 is the label for people
		if class_names[r['class_ids'][i]] in important_classes:
			# get bounding box
			mask = r['masks'][:,:,i]
			mask = ndimage.binary_dilation(mask, iterations=10)
			bbox = r['rois'][i]
			topleft = [bbox[0], bbox[1]]
			topright = [bbox[0], bbox[3]]
			bottomleft = [bbox[2], bbox[1]]
			bottomright = [bbox[2], bbox[3]]
			print("mask", mask.shape)

			# Display Image
			f, axarr = plt.subplots(1, 2)
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

			axarr[0].axvline(x=quad[0])
			axarr[0].axvline(x=quad[1])
			axarr[0].axhline(y=h_thirds[0])
			axarr[0].axhline(y=h_thirds[1])
			axarr[0].plot(middle_pt[1], middle_pt[0], marker='o', color='red')
			# if line is quad[2]:
			# 	axarr[0].axvline(x=quad[1])
			# elif line is quad[3]:
			# 	axarr[0].axvline(x=quad[0])
			# else:
			# 	axarr[0].axvline(x=line)

			# Vertical Seams
			count = 0
			im_temp = None
			orig_mask = None
			if isMiddle:
				print("Object in middle")
				# closest line is left line, delete from the left
				if line is quad[0]:
					dist = abs(middle_pt[1] - line) - 5
					half = dist/2
					while dist >= half:
						im, mask, bbox = seam_removing.deleteLines(im, 3, 'vertical', mask, bbox, 'left')
						# line -= 1
						# middle_pt[1] -= 3
						# count += 1
						dist -= 2
					while dist >= 0:
						im, im_temp, mask, bbox, orig_mask = seam_adding.addLines(im, 'vertical', mask, bbox, 'right', im_temp, orig_mask)
						dist -= 2
						count += 3
				# closest line is right line, delete from the right
				elif line is quad[1]:
					dist = abs(middle_pt[1] - line) - 5
					half = dist/2
					while dist >= half:
						im, mask, bbox = seam_removing.deleteLines(im, 3, 'vertical', mask, bbox, 'right')
						# line -= 2
						dist -= 2
					while dist >= 0:
						im, im_temp, mask, bbox, orig_mask = seam_adding.addLines(im, 'vertical', mask, bbox, 'left', im_temp, orig_mask, count)
						dist -= 2
						count += 3
			else:
				# object is on right side
				if line is quad[2] or line is quad[1]:
					line = quad[1]
					print("Object is on right side")
					dist = abs(middle_pt[1] - line) - 5
					half = dist/2
					while dist >= half:
						print(mask.shape, im.shape, bbox.shape)
						im, im_temp, mask, bbox, orig_mask = seam_adding.addLines(im, 'vertical', mask, bbox, 'right', im_temp, orig_mask, count)
						# line +=2
						count += 3
						dist -= 3
					mask = orig_mask
					while dist >= 0:
						im, mask, bbox = seam_removing.deleteLines(im, 3, 'vertical', mask, bbox, 'left')
						dist -= 1

				# object is on left side
				elif line is quad[3] or line is quad[0]:
					line = quad[0]
					print("Object is on left side")
					dist = abs(middle_pt[1] - line) - 5
					half = dist/2
					while dist >= half:
						print(abs(middle_pt[1] - line), middle_pt[1], line)
						im, im_temp, mask, bbox, orig_mask = seam_adding.addLines(im, 'vertical', mask, bbox, 'left', im_temp, orig_mask, count)
						# line +=1
						# middle_pt[1] += 3
						dist -= 2
						count += 3
					mask = orig_mask
					while dist >= 0:
						im, mask, bbox = seam_removing.deleteLines(im, 3, 'vertical', mask, bbox, 'right')
						dist -= 1
					# while abs(middle_pt[1] - line) > 5:
					# 	im, mask, bbox = seam_removing.deleteLines(im, 3, 'vertical', mask, bbox, 'right')
					# 	line -= 1

			# Horizontal Seams

			im_temp = None
			count = 0
			mask = orig_mask
			# if not isHMiddle:
			# 	# object is above middle
			# 	if h_line is h_thirds[0] or h_line is h_thirds[3]:
			# 		print("Object is above middle")
			# 		while abs(middle_pt[0] - h_line) > 50:
			# 			im, mask, bbox = seam_removing.deleteLines(im, 3, 'horizontal', mask, bbox, 'bottom')
			# 			h_line -= 1
			# 			count += 1
			# 	#object is below middle
			# 	elif h_line is h_thirds[1] or h_line is h_thirds[2]:
			# 		print("Object is below middle")
			# 		while abs(middle_pt[0] - h_line) > 20:
			# 			print(count)
			# 			im, mask, bbox = seam_removing.deleteLines(im, 3, 'horizontal', mask, bbox, 'up')
			# 			h_line -= 2
			# 			middle_pt[0] -= 3
			# 			count += 1
					
			print ("count", count)
			axarr[0].set_title('Original')
			axarr[1].imshow(im)
			axarr[1].set_title('Modified')
			axarr[1].axvline(x=im.shape[1]/3)
			axarr[1].axvline(x=im.shape[1]*2/3)
			axarr[1].axhline(y=im.shape[0]/3)
			axarr[1].axhline(y=im.shape[0]*2/3)

			print(i)
			cur_im = Image.fromarray(im)
			cur_im.save("teddybear" + str(i) + str(count) + ".jpg")

			plt.show()
