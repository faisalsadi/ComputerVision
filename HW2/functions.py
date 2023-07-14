import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


def xor(array1, array2):
    result = np.bitwise_xor(array1, array2)
    count = 0
    size = result.size
    for i in range(size):
        if result[i] == 1:
            count += 1
    return count


def censusTransform(matrix, k):
    n = matrix.shape[0]
    m = matrix.shape[1]
    k_values = np.zeros((n, m), dtype=object)
    for i in range(n):
        for j in range(m):
            if i + k // 2 >= n - 1:
                end_row = n - 1
                start_row = i - (k - (n - i))

            else:
                start_row = max(i - k // 2, 0)
                end_row = i + (k - (i - start_row)) - 1
            if j + k // 2 >= m - 1:
                end_col = m - 1
                start_col = j - (k - (m - j))

            else:
                start_col = max(j - k // 2, 0)
                end_col = j + (k - (j - start_col)) - 1
            submatrix = np.zeros((k, k))
            submatrix = matrix[start_row:end_row + 1, start_col:end_col + 1].copy()
            for x in range(k):
                for y in range(k):

                    if submatrix[x][y] >= matrix[i][j]:
                        submatrix[x][y] = 1
                    else:
                        submatrix[x][y] = 0
            victor = submatrix.reshape(k * k)
            k_values[i, j] = victor

    return k_values


# from lift to right image(lift is the maain picture)
def costVolumeLR(victorL, victorR, file_content):
    n = victorL.shape[0]
    m = victorL.shape[1]
    file_content = int(file_content)
    victor = np.zeros((n, m, file_content))
    # victor[:][:][:]=-1
    for i in range(n):
        for j in range(m):
            vL = victorL[i][j].astype(int)
            # start_row=max(0,j-file_content)
            x = 0
            for k in range(file_content):
                if j - k >= 0:
                    vR = victorR[i][j - k].astype(int)
                    count = xor(vL, vR)
                    victor[i][j][k] = count
                    # x+=1
                else:
                    victor[i][j][k] = file_content
            # for z in range(x, file_content):
            #     victor[i][j][z] = float('inf')
    return victor


# from right to lift image(right is the maain picture)

def costVolumeRL(victorR, victorL, file_content):
    n = victorR.shape[0]
    m = victorR.shape[1]
    file_content = int(file_content)
    victor = np.zeros((n, m, file_content))
    for i in range(n):
        for j in range(m):
            vR = victorR[i][j].astype(int)
            # start_row=max(0,j-file_content)
            x = 0
            for k in range(file_content):
                if j + k < m:
                    vL = victorL[i][j + k].astype(int)
                    count = xor(vL, vR)
                    victor[i][j][k] = count
                    # x+=1
                else:
                    victor[i][j][k] = file_content
            # for z in range(x, file_content):
            #     victor[i][j][z] = float('inf')
    return victor


def filterAv(matrix, maxDis, kernel_size):
    maxDis = int(maxDis)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    for d in range(maxDis):
        matrix[:, :, d] = cv2.filter2D(matrix[:, :, d], -1, kernel)
        # print(matrix[:][:][d])

    return matrix


def minMat(matrix, maxdis):
    maxdis = int(maxdis)
    n, m = matrix.shape[0], matrix.shape[1]
    array = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            min = matrix[i][j][0]
            min_index = 0
            for d in range(maxdis):
                if matrix[i][j][d] < min:
                    min = matrix[i][j][d]
                    min_index = d
            array[i][j] = min_index

    return array


def consistency_testLR(disparity_left, disparity_right, max_difference=1):
    height, width = disparity_left.shape
    disparity = np.zeros_like(disparity_left)

    for y in range(height):
        for x in range(width):
            disparity_l = int(disparity_left[y, x])
            if x - disparity_l >= 0:
                disparity_r = disparity_right[y, x - disparity_l]
                diff = np.abs(disparity_l - disparity_r)
                if diff < max_difference:
                    disparity[y, x] = disparity_left[y, x]
    return disparity


def consistency_testRL(disparity_right, disparity_left, max_difference=1):
    height = disparity_left.shape[0]
    width = disparity_left.shape[1]
    disparity = np.zeros_like(disparity_left)

    for y in range(height):
        for x in range(width):
            disparity_r = int(disparity_right[y, x])
            if x + disparity_r < width:
                disparity_l = disparity_left[y, x + disparity_r]
                diff = np.abs(disparity_r - disparity_l)
                if diff < max_difference:
                    disparity[y, x] = disparity_right[y, x]
    return disparity


def depth(disparity,f, baseline=0.1):
    height, width = disparity.shape
    arr = np.zeros_like(disparity)
    for i in range(height):
        for j in range(width):
            if disparity[i][j] != 0:
                arr[i][j] = f*baseline / disparity[i][j]
    return arr


def disparity(imageL, imageR, k, file_content, kernel_size, path,f):
    victorL = censusTransform(imageL, k)
    victorR = censusTransform(imageR, k)
    # from right image to left
    victorRL = costVolumeRL(victorR, victorL, file_content)
    filterR = filterAv(victorRL, file_content, kernel_size)
    minArrayR = minMat(filterR, file_content)

    # #from left image to right
    victorLR = costVolumeLR(victorL, victorR, file_content)
    filterL = filterAv(victorLR, file_content, kernel_size)
    minArrayL = minMat(filterL, file_content)
    disp_left = consistency_testLR(minArrayL, minArrayR, 1)
    disp_right = consistency_testRL(minArrayR, minArrayL, 1)

    # depth_right = depth(disp_right / np.max(disp_right), 0.1)
    # depth_lift = depth(disp_left / np.max(disp_left), 0.1)
    depth_right = depth(disp_right ,f, 0.1)
    depth_lift = depth(disp_left , f,0.1)

    # # cv2.imshow('dis', minArrayR/np.max(minArrayR))
    # cv2.imshow('dis_left',disp_left/np.max(disp_left))
    #
    # cv2.imshow('dis_right',disp_right/np.max(disp_right))
    # cv2.imshow('depth_right',depth_right)
    # cv2.imshow('depth_left',depth_lift)
    #
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # disp_left_image = (disp_left / np.max(disp_left)) * 255
    # disp_left_image = disp_left_image.astype(np.uint8)
    #
    # disp_right_image = (disp_right / np.max(disp_right)) * 255
    # disp_right_image = disp_right_image.astype(np.uint8)
    #
    # depth_left_image = (depth_lift / np.max(depth_lift)) * 255
    # depth_left_image = depth_left_image.astype(np.uint8)
    #
    # depth_right_image = (depth_right / np.max(depth_right)) * 255
    # depth_right_image = depth_right_image.astype(np.uint8)
    #
    # cv2.imwrite(path + "disp_left.jpg", disp_left_image)
    # cv2.imwrite(path + "disp_right.jpg", disp_right_image)
    #
    # cv2.imwrite(path + "depth_left.jpg", depth_left_image)
    # cv2.imwrite(path + "depth_right.jpg", depth_right_image)



    disp_left_image = (disp_left / np.max(disp_left))

    disp_right_image = (disp_right / np.max(disp_right))

    depth_left_image = (depth_lift / np.max(depth_lift))

    depth_right_image = (depth_right / np.max(depth_right))

    plt.imsave(path + "disp_left.jpg", disp_left_image,cmap='gray')
    plt.imsave(path + "disp_right.jpg", disp_right_image,cmap='gray')

    plt.imsave(path + "depth_left.jpg", depth_left_image,cmap='gray')
    plt.imsave(path + "depth_right.jpg", depth_right_image,cmap='gray')
    np.savetxt(path+'depth_left.txt',depth_lift,delimiter=',')
    np.savetxt(path+'depth_right.txt',depth_right,delimiter=',')
    np.savetxt(path+'disp_left.txt',disp_left,delimiter=',')
    np.savetxt(path+'disp_right.txt',disp_right,delimiter=',')




