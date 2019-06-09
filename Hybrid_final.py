import numpy as np
import cv2
import math
import matplotlib.image as mpimg
import matplotlib
from scipy import misc
from matplotlib import pyplot as plt

# Creation of Gaussians
def Make_Gaussian(Img, Low_pass_filter, S):
    # Take the size of the input image
    Rows =  Img.shape[0]
    Col = Img.shape[1]
    # Find the centre of the Image, if it's a decimal increment by one
    I_center = int(Rows/2) + 1 if Rows % 2 == 1 else int(Rows/2)
    J_center = int(Col/2) + 1 if Col % 2 == 1 else int(Col/2)
    # Create an empty 0 image to store the gaussian
    K = np.zeros((Rows, Col))

    """ Iterate through every col and row, at each stage when calculating the Gaussian
     value at I and j take away the value of the centre. This will allow the forming of 
     the highest concentration when i = I_center and j = J_center being both 0 the exp 
     part will equal to one. This is the highest point for the low-frequency template"""
    for i in range(Rows):
        for j in range(Col):
            c = math.exp(-(((i - I_center)**2+(j - J_center)**2)/(2*S**2)))
            if Low_pass_filter:
                K[i][j] = c
            else:
                # if we are working on the high frequency part we can invert the Gaussian
                K[i][j] = (1-c)
    return K


# Return the Component using FFT convolution
def Convolution_FFT(Img, Gaussian):
    # Create an empty image that will later be used to store the result
    Filtered = np.zeros(Img.shape)

    # Take the image, apply the fast Fourier transform,  
    def FFT(Component, Gaussian):
        """applies the FFT to one of the 3 RGB channels, placing us in the 
        frequency domain and shifts the 0 frequency component back in the middle"""
        C_fft = np.fft.fftshift(np.fft.fft2(Component))
        # Matrix multiplication

        fig, ax = plt.subplots(figsize=(14,7),nrows=1, ncols=2)
        magnitude_spectrum = 20*np.log(np.abs(C_fft))

        ax[0].imshow((magnitude_spectrum))

        Result = C_fft*Gaussian

        
        ax[1].imshow(np.real((Result)))

        # inversing everithing allows to go back to the spatial.
        ifft = np.fft.ifft2(np.fft.ifftshift(Result))
        return ifft
    # check to see if the shape has more than one channel, if not its a grayscale.
    if len(Img.shape) == 2:
        Filtered[:,:] = FFT(Img[:,:], Gaussian)
    else:
        """Do the FFT function and convolution for every RGB channel and fill each 
        channel in Filtered"""
        for i in range(Img.shape[-1]):
            Filtered[:,:,i] = FFT(Img[:,:,i], Gaussian)
    return Filtered

# Add the low and high frequency component together producing the final hybrid image
def Merge(low, high):
    low = low + high
    return low




# Load the two images
Low = mpimg.imread('data/fish.bmp', 1)
High = mpimg.imread('data/submarine.bmp', 1)

Gaussian_low = Make_Gaussian(Low, True, 12)
Gaussian_high = Make_Gaussian(High, False, 4)
misc.imsave("Gauss_low.png", np.real(Gaussian_low))
misc.imsave("Gauss_high.png", np.real(Gaussian_high))
Low_frequency_component = Convolution_FFT(Low, Gaussian_low)
High_frequency_component = Convolution_FFT(High, Gaussian_high)
misc.imsave("low-passed.png", np.real(Low_frequency_component))
misc.imsave("high-passed.png", np.real(High_frequency_component))
M = Merge(Low_frequency_component, High_frequency_component)
misc.imsave("Merged.png", np.real(M))

from PIL import Image

size = [ M.shape[1],M.shape[0]]

# Plotting of the image in different sizes
Images = []
for i in range(4):
    im = Image.open("Merged.png")
    im.thumbnail([size[0]/(i+1), size[1]/(i+1)])
    A = np.array(im)
    Images.append(A)

fig, axes = plt.subplots(figsize=(14,7),nrows=1, ncols=len(Images))

for i in range(len(Images)):
    axes[i].imshow(Images[i])
    axes[i].set_xlim([0,size[0] ])
    axes[i].set_ylim([size[1],0])
    axes[i].axis('off')


plt.show()