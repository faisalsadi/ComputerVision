# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import cv2
from functions import disparity


# Example usage
def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except IOError:
        print(f"Error: Could not read the file '{file_path}'.")





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # Read the array from file
    # array = np.loadtxt("example/disp_right.txt", delimiter=',')
    #
    # # Convert the array to an image
    # imager = (array / np.max(array)) * 255
    # imager = imager.astype(np.uint8)
    #
    # # Read the array from file
    # array = np.loadtxt('example/disp_left.txt', delimiter=',')
    #
    # # Convert the array to an image
    # imagel = (array / np.max(array)) * 255
    # imagel = imagel.astype(np.uint8)
    # # Read the array from file
    # array = np.loadtxt('example/depth_right.txt', delimiter=',')
    #
    # # Convert the array to an image
    # imagedr = (array / np.max(array)) * 255
    # imagedr = imagedr.astype(np.uint8)
    # # Read the array from file
    # array = np.loadtxt('example/depth_left.txt', delimiter=',')
    #
    # # Convert the array to an image
    # imagedl = (array / np.max(array)) * 255
    # imagedl = imagedl.astype(np.uint8)
    #
    # # Save the image to disk
    #
    # cv2.imwrite("set_1/dis_left.jpg", imagel)
    # cv2.imwrite("set_1/dis_right.jpg", imager)
    # cv2.imwrite("set_1/depth_left.jpg", imagedl )
    # cv2.imwrite("set_1/depth_right.jpg", imagedr *255)

    # # example
    # imageR = cv2.imread("Data/example/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    # imageL = cv2.imread("Data/example/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #
    # #example k=5 kernel=25
    # #set 1 k=5\7 kernel=41
    # disparity(imageL,imageR,k=5, file_content = read_file('Data/example/max_disp.txt'),kernel_size=25,path="Data/example/",f=576)

    # set1
    imageR = cv2.imread("Data/set_1/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("Data/set_1/im_left.jpg", cv2.IMREAD_GRAYSCALE)

    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('Data/set_1/max_disp.txt'),kernel_size=41,path="Data/set_1/",f=688.000061035156)
    print(1)
    # set2
    imageR = cv2.imread("Data/set_2/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("Data/set_2/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('Data/set_2/max_disp.txt'),kernel_size=42,path="Data/set_2/",f=1120)
    print(2)

    #set3
    imageR = cv2.imread("Data/set_3/im_right.jpg" ,cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("Data/set_3/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('Data/set_3/max_disp.txt'),kernel_size=25,path="Data/set_3/",f=800)
    print(3)


    # set4

    imageR = cv2.imread("Data/set_4/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("Data/set_4/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('Data/set_4/max_disp.txt'),kernel_size=45,path="Data/set_4/",f=896)
    print(4)


    # set5
    imageR = cv2.imread("Data/set_5/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("Data/set_5/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('Data/set_5/max_disp.txt'),kernel_size=30,path="Data/set_5/",f=689.029357910156)
    print(5)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


def calculate_depth_disparity():
    # set1
    imageR = cv2.imread("Data/set_1/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("Data/set_1/im_left.jpg", cv2.IMREAD_GRAYSCALE)

    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('Data/set_1/max_disp.txt'),kernel_size=41,path="Data/set_1/",f=688.000061035156)
    print(1)
    # set2
    imageR = cv2.imread("Data/set_2/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("Data/set_2/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('Data/set_2/max_disp.txt'),kernel_size=42,path="Data/set_2/",f=1120)
    print(2)

    #set3
    imageR = cv2.imread("Data/set_3/im_right.jpg" ,cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("Data/set_3/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('Data/set_3/max_disp.txt'),kernel_size=25,path="Data/set_3/",f=800)
    print(3)


    # set4

    imageR = cv2.imread("Data/set_4/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("Data/set_4/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('Data/set_4/max_disp.txt'),kernel_size=45,path="Data/set_4/",f=896)
    print(4)


    # set5
    imageR = cv2.imread("Data/set_5/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("Data/set_5/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('Data/set_5/max_disp.txt'),kernel_size=30,path="Data/set_5/",f=689.029357910156)
    print(5)