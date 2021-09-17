
import numpy as np
from math import exp


def filter_2d(im, kernel):
    '''
    Filter an image by taking the dot product of each 
    image neighborhood with the kernel matrix.
    Args:
    im = (H x W) grayscale floating point image
    kernel = (M x N) matrix, smaller than im
    Returns: 
    (H-M+1 x W-N+1) filtered image.
    '''

    M = kernel.shape[0] 
    N = kernel.shape[1]
    H = im.shape[0]
    W = im.shape[1]
    
    filtered_image = np.zeros((H-M+1, W-N+1), dtype = 'float64')
    
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            image_patch = im[i:i+M, j:j+N]
            filtered_image[i, j] = np.sum(np.multiply(image_patch, kernel))
            
    return filtered_image

def convert_to_grayscale(im):
    '''
    Convert color image to grayscale.
    Args: im = (nxmx3) floating point color image scaled between 0 and 1
    Returns: (nxm) floating point grayscale image scaled between 0 and 1
    '''
    return np.mean(im, axis = 2)


def gkernal2(size, sigma):
    '''
    Create a gaussian kernel of size x size. 
    Args: 
    size = must be an odd positive number
    sigma = standard deviation of gaussian in pixels
    Returns: A floating point (size x size) guassian kernel 
    ''' 
    #Make kernel of zeros:
    kernel = np.zeros((size, size))
    
    #Handle sigma = 0 case (will result in dividing by zero below if unchecked)
    if sigma == 0:
        return kernel 
    
    index = int((size-1)/2)
    
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1/(2*np.pi*sigma**2))*exp(-((i-index)**2 + (j-index)**2)/(2*sigma**2))
            
    return kernel

gkernel = gkernal2(11,2)

def classify(im):
    '''
    Example submission for coding challenge. 
    Args: im (nxmx3) unsigned 8-bit color image 
    Returns: One of three strings: 'brick', 'ball', or 'cylinder'  
    '''
    im = im[10:245,10:245]

    gray = convert_to_grayscale(im/255)
    
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    
    
    Gx = filter_2d(gray, Kx)
    Gy = filter_2d(gray, Ky)
    
    #Compute Gradient Magnitude and Direction:
    G_magnitude = np.sqrt(Gx**2+Gy**2)
    G_direction = np.arctan2(Gy, Gx)
    G_max = np.max(G_magnitude)
    edges = G_magnitude > (G_max*0.55)
    total_value = np.sum(G_magnitude[np.where(G_magnitude > (G_max*0.55))])
    y_c, x_c = np.where(edges)


    objects = ['brick', 'ball', 'cylinder']

    #Let's guess randomly! Maybe we'll get lucky.
    #random_integer = np.random.randint(low = 0, high = 3)
    #return objects [random_integer]

    if  (total_value/len(x_c)) > 1.2 and (total_value/len(x_c)) <= 1.6:
        return objects[2]
    elif (total_value/len(x_c)) > 0 and (total_value/len(x_c)) <= 1.2:
        return objects[1]
    else:
        return objects[0]