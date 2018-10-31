import sys 
import cv2 
import numpy as np 

def addLines(img_input, img_removed=None):
    if img_removed is None:
        img = np.copy(img_input) 
    else:
        img = img_removed
    img_output = np.copy(img_input) 
    energy = compute_energy_matrix(img) # Same than previous code sample
 
    for i in range(3): 
        seam = find_vertical_seam(img, energy) # Same than previous code sample
        img = remove_vertical_seam(img, seam) # Same than previous code sample
        img_output = add_vertical_seam(img_output, seam, i) 
        energy = compute_energy_matrix(img) 
        print('Number of seams added =', i+1)

    return img_output, img

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