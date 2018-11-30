import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def addLines(img_input, direction, mask, bbox, side, img_removed=None, orig_mask=None, num_iter=0):
    if img_removed is None:
        removed_seams = np.copy(img_input)
    else:
        removed_seams = img_removed

    if orig_mask is None:
        orig_mask = np.copy(mask)

    img_output = np.copy(img_input)
    energy = compute_energy_matrix(removed_seams)
    energy = manipulate_energy_image(energy, mask, bbox, side, num_iter)
    # Add 3 seams at a time
    for i in range(3):
        if direction == 'vertical':
            seam = find_vertical_seam(removed_seams, energy)
            # plt.imshow(energy)
            # plt.show()
            # displaySeam(energy, seam, 'VERTICAL')
            # removed_seams keeps track of what seams were already selected
            #    to be re-added
            removed_seams = remove_vertical_seam(removed_seams, seam)
            img_output = add_vertical_seam(img_output, seam, i + num_iter)
        # Recompute energy matrix
        energy = compute_energy_matrix(removed_seams)
        # Calculate New Mask
        if side == 'up':
            temp = np.zeros(mask.shape[1], dtype=bool)
            orig_mask = np.vstack((temp, orig_mask))
            mask = mask[1:,:]
            bbox[0][0] -= 1
            bbox[1][0] -= 1
        elif side == 'bottom':
            temp = np.zeros(mask.shape[1], dtype=bool)
            orig_mask = np.vstack((orig_mask, temp))
            mask = mask[:-1,:]
        elif side == 'left':
            temp = np.zeros((mask.shape[0], 1), dtype=bool)
            orig_mask = np.hstack((temp, orig_mask))
            mask = mask[:,1:]
            bbox[0][1] += 1
            bbox[1][1] += 1
        else:
            temp = np.zeros((mask.shape[0], 1), dtype=bool)
            orig_mask = np.hstack((orig_mask, temp))            
            mask = mask[:,:-1]

        energy = manipulate_energy_image(energy, mask, bbox, side, num_iter + i + 1)

    return img_output, removed_seams, mask, bbox, orig_mask

# Add a vertical seam to the image
def add_vertical_seam(img, seam, num_iter):
    seam = seam + num_iter
    rows, cols = img.shape[:2]
    zero_col_mat = np.zeros((rows,1,3), dtype=np.uint8)
    img_extended = np.hstack((img, zero_col_mat))

    for row in range(rows):
        for col in range(cols, int(seam[row]), -1):
            img_extended[row, col] = img[row, col-1]

        # To insert a value between two columns, take the average
        # value of the neighbors. It looks smooth this way and we
        # can avoid unwanted artifacts.
        for i in range(3):
            v1 = img_extended[row, int(seam[row])-1, i]
            v2 = img_extended[row, int(seam[row])+1, i]
            img_extended[row, int(seam[row]), i] = (int(v1)+int(v2))/2

    return img_extended

def displaySeam(im, seam, type):
    fig = plt.figure(figsize=(8,6))
    if type is 'HORIZONTAL':
        plt.title('Horizontal Seam')
        for i in range(seam.shape[0]):
            plt.plot(i, seam[i], 'ro')
            # im[seam[i], i] = [0, 255, 0]
    elif type is 'VERTICAL':
        plt.title('Vertical Seam')
        for i in range(seam.shape[0]):
            plt.plot(seam[i], i, 'go')
            # im[i, seam[i]] = [255, 0, 0]

    plt.imshow(im)

    # ax.set_title('Warped Input Image')
    plt.show()

def add_horizontal_seam(img, seam, num_iter):
    seam = seam + num_iter
    rows, cols = img.shape[:2]
    zero_row_mat = np.zeros((1, cols, 3), dtype=np.uint8)
    img_extended = np.vstack((img, zero_row_mat))

    for col in range(cols):
        for row in range(rows, int(seam[col]), -1):
            img_extended[row, col] = img[row-1, col]

        for i in range(3):
            v1 = img_extended[int(seam[col])-1, col, i]
            v2 = img_extended[int(seam[col])+1, col, i]
            img_extended[int(seam[col]), col, i] = (int(v1)+int(v2))/2

    return img_extended

# Remove the input vertical seam from the image
def remove_vertical_seam(img, seam):
    rows, cols = img.shape[:2]

    # To delete a point, move every point after it one step towards the left
    for row in range(rows):
        for col in range(int(seam[row]), cols-1):
            img[row, col] = img[row, col+1]

    # Discard the last column to create the final output image
    img = img[:, 0:cols-1]
    return img

def remove_horizontal_seam(img, seam):
    new_im = np.zeros((img.shape[0]-1, img.shape[1], 3))
    for i in range(len(seam)):
        new_im[:, i, 0] = np.delete(img[:, i, 0], seam[i])
        new_im[:, i, 1] = np.delete(img[:, i, 1], seam[i])
        new_im[:, i, 2] = np.delete(img[:, i, 2], seam[i])
    new_im = new_im.astype(np.uint8)
    return new_im

# Compute the energy matrix from the input image
def compute_energy_matrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute X derivative of the image
    sobel_x = cv2.Sobel(gray,cv2.CV_64F, 1, 0, ksize=3)

    # Compute Y derivative of the image
    sobel_y = cv2.Sobel(gray,cv2.CV_64F, 0, 1, ksize=3)

    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # Return weighted summation of the two images i.e. 0.5*X + 0.5*Y
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# Find horizontal seam in the input image
def find_optimal_horizontal_seam(img, energy):
    rows, cols = img.shape[:2]
    seam = np.zeros(img.shape[1])

    dist_to = np.zeros(img.shape[:2]) + float('inf')
    dist_to[:,0] = np.zeros(img.shape[0])
    edge_to = np.zeros(img.shape[:2])

    for col in range(cols-1):
        for row in range(rows):
            if row != 0 and dist_to[row-1, col+1] > dist_to[row, col] + energy[row-1, col+1]:
                dist_to[row-1, col+1] = dist_to[row, col] + energy[row-1, col+1]
                edge_to[row-1, col+1] = 1

            if dist_to[row, col+1] > dist_to[row, col] + energy[row, col+1]:
                dist_to[row, col+1] = dist_to[row, col] + energy[row, col+1]
                edge_to[row, col+1] = 0

            if row != rows-1 and dist_to[row+1, col+1] > dist_to[row, col] + energy[row+1, col+1]:
                dist_to[row+1, col+1] = dist_to[row, col] + energy[row+1, col+1]
                edge_to[row+1, col+1] = -1

    seam[cols-1] = np.argmin(dist_to[:, col-1])
    for i in (x for x in reversed(range(cols)) if x > 0):
        seam[i-1] = seam[i] + edge_to[int(seam[i]), i]
    return seam

# Find vertical seam in the input image
def find_vertical_seam(img, energy):
    rows, cols = img.shape[:2]

    # Initialize the seam vector with 0 for each element
    seam = np.zeros(img.shape[0])

    # Initialize distance and edge matrices
    dist_to = np.zeros(img.shape[:2]) + float('inf')
    dist_to[0,:] = np.zeros(img.shape[1])
    edge_to = np.zeros(img.shape[:2])

    # Dynamic programming; iterate using double loop and compute the paths efficiently
    for row in range(rows-1):
        for col in range(cols):
            if col != 0 and dist_to[row+1, col-1] > dist_to[row, col] + energy[row+1, col-1]:
                dist_to[row+1, col-1] = dist_to[row, col] + energy[row+1, col-1]
                edge_to[row+1, col-1] = 1

            if dist_to[row+1, col] > dist_to[row, col] + energy[row+1, col]:
                dist_to[row+1, col] = dist_to[row, col] + energy[row+1, col]
                edge_to[row+1, col] = 0

            if col != cols-1 and dist_to[row+1, col+1] > dist_to[row, col] + energy[row+1, col+1]:
                    dist_to[row+1, col+1] = dist_to[row, col] + energy[row+1, col+1]
                    edge_to[row+1, col+1] = -1

    # Retracing the path
    # Returns the indices of the minimum values along X axis.
    seam[rows-1] = np.argmin(dist_to[rows-1, :])
    for i in (x for x in reversed(range(rows)) if x > 0):
        seam[i-1] = seam[i] + edge_to[i, int(seam[i])]

    return seam


def manipulate_energy_image(energy, mask, bbox, side, num_iter):
    # Maniplate Energy Regions
    if side == 'up':
        energy[bbox[1][0]:, bbox[0][1]:bbox[1][1] + 1] = 255
    elif side == 'bottom':
        energy[:bbox[0][0], bbox[0][1]:bbox[1][1] + 1] = 255
    elif side == 'left':
        # print("bbox", bbox[1][1]-num_iter, num_iter)
        energy[bbox[0][0]:bbox[1][0] + 1, bbox[1][1]-20 - num_iter*2:] = 255
    else:
        energy[bbox[0][0]:bbox[1][0] + 1, :bbox[0][1]] = 255

    # Mask Energy Image
    energy[mask] = 255
    # plt.imshow(mag)
    # plt.show()
    
    return energy