import cv2 as cv  # to implement all processes onto the images
import numpy as np  # to convert images into arrays and matrices
import matplotlib.pyplot as plt  # to plot the images
import easygui  # to store path of image
import tkinter as tk  # to provide dialogbox and menu for selecting the required image
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import sys  # to manipulate python runtime environment
import os  # standard python library
# openCV,matplotlib,easygui,tkinter are all machine learning tools

# defining upload function


def upload():
    ImagePath = easygui.fileopenbox()
    cartoonify(ImagePath)

# defining cartoonification function


def cartoonify(ImagePath):
    # reading the image
    img = cv.imread(ImagePath)

    # exit if image is not selected
    if img is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()

    # resizing the image
    resize = cv.resize(img, (480, 360), interpolation=cv.INTER_AREA)

    # converting to bgr image to grayscale
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)

    # blurring the gray image by Gaussian blur
    blurred = cv.GaussianBlur(gray, (9, 3), 0)

    # detecteing the edges using thresholding
    edge = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 7)

    # blurring the BGR resized image by median blur
    reblur = cv.medianBlur(resize, 5)

    # defining a function to quantise the image
    def quan(img, k):
        data = np.float32(img).reshape(-1, 5)
        criteria = (cv.TermCriteria_EPS + cv.TERM_CRITERIA_MAX_ITER, 15, 0.01)
        ret, label, center = cv.kmeans(
            data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        return result

    # calling quantisation  function
    q = quan(reblur, 70)

    # applying bilateral filter technique on quantised image
    noise = cv.bilateralFilter(q, 15, 190, 190)

    # masking the filterd image with the edges that were detected
    ci = cv.bitwise_and(noise, noise, mask=edge)

    # converting BGR to RGB for plotting the images
    resize = cv.cvtColor(resize, cv.COLOR_BGR2RGB)
    ci = cv.cvtColor(ci, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2RGB)
    noise = cv.cvtColor(noise, cv.COLOR_BGR2RGB)
    reblur = cv.cvtColor(reblur, cv.COLOR_BGR2RGB)
    edge = cv.cvtColor(edge, cv.COLOR_BGR2RGB)
    q = cv.cvtColor(q, cv.COLOR_BGR2RGB)
    blurred = cv.cvtColor(blurred, cv.COLOR_BGR2RGB)

    # resizing the images using matplot library
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(6, 4), dpi=250, facecolor='w', edgecolor='k')

    # defining subplotting images for original and cartoonified image
    plt.subplot(1, 2, 1)
    plt.imshow(resize)
    plt.title('Original image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(ci)
    plt.title('Cartoonified')
    plt.xticks([])
    plt.yticks([])

    plt.show()

    from matplotlib.pyplot import figure
    figure(num=None, figsize=(5, 4), dpi=210, facecolor='w', edgecolor='k')

    plt.subplot(2, 4, 1)
    plt.imshow(resize)
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 2)
    plt.imshow(gray)
    plt.title('Gray image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 3)
    plt.imshow(blurred)
    plt.title('Blurred gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 4)
    plt.imshow(edge)
    plt.title('Detected edges')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 5)
    plt.imshow(reblur)
    plt.title('Blurred BGR')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 6)
    plt.imshow(q)
    plt.title('Quantised')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 7)
    plt.imshow(noise)
    plt.title('Filtered')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 8)
    plt.imshow(ci)
    plt.title('Masked')
    plt.xticks([])
    plt.yticks([])

    plt.show()


# defining the pop-up menu and dialog box using tkinter ML library
top = tk.Tk()
top.geometry('600x400')
top.title('Image Cartoonifier')
top.configure(background='white')
label = Label(top, background='#15d6b9', font=('calibri', 18, 'bold'))

upload = Button(top, text="Upload Image",
                command=upload, padx=10, pady=5)
upload.configure(background='#2dc8e3', foreground='white',
                 font=('Cooper Std Black', 18, 'bold'))
upload.pack(side=TOP, pady=160)

top.mainloop()
