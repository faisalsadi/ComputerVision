import numpy as np
import cv2
import matplotlib.pyplot as plt
from mapsCalculation import calculate_depth_disparity
def reproject_to_3d(image, K, depth_map):
    # Get image dimensions
    height, width = image.shape[:2]

    # Create array to store 3D points
    points_3d = np.zeros((height, width, 3), dtype=np.float32)

    # Calculate inverse camera matrix
    inv_camera_matrix = np.linalg.inv(K)

    # Reproject each pixel to 3D
    for y in range(height):
        for x in range(width):
            # Get depth value for the pixel
            depth = depth_map[y, x]

            # Calculate 3D coordinates using the camera matrix and depth value
            pixel = np.array([[x, y, 1]], dtype=np.float32)
            pixel_3d = depth * np.dot(inv_camera_matrix, pixel.T)
            points_3d[y, x] = pixel_3d[:, 0]

    return points_3d
def project_to_camera_plane(points_3d, camera_matrix):
    # Get image dimensions
    height, width = points_3d.shape[:2]

    # Create array to store projected 2D points
    points_2d = np.zeros((height, width, 2), dtype=np.float32)

    # Project 3D points to camera plane
    for y in range(height):
        for x in range(width):
            # Get 3D coordinates of the point
            point_3d = points_3d[y, x]

            # Convert 3D point to homogeneous coordinates
            point_3d_homogeneous = np.append(point_3d, 1)

            # Project 3D point to 2D using camera matrix
            point_2d_homogeneous = np.dot(camera_matrix, point_3d_homogeneous)
            point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
            # Store the projected 2D point
            points_2d[y, x] = point_2d

    return points_2d





## Main
############################################################################################
if __name__ == '__main__':
    # calculate disparity's and depth's maps and images and store them in set{i} directory
    calculate_depth_disparity()
    focals = [0,688.000061035156,1120,800,896,689.029357910156]
    # this loop generates the 11 required synth images
    for j in range(1,6):
        image = cv2.imread(f"Data/set_{j}/im_left.jpg")
        depth_map = np.loadtxt(f"Data/set_{j}/depth_left.txt",delimiter=',')
        K = np.array([[focals[j], 0, 511.5], [0, focals[j], 217.5], [0, 0, 1]], dtype=np.float32)
        # Call the function to reproject image coordinates to 3D space
        points_3d = reproject_to_3d(image, K, depth_map)
        for i in range (11):
            ext=np.array([[1, 0, 0,-0.01*i], [0, 1, 0,0], [0, 0, 1,0]], dtype=np.float32)
            camera_matrix =np.dot(K ,ext )

            # Call the function to project 3D points to the camera plane
            points_2d = project_to_camera_plane(points_3d, camera_matrix)

            # Create a blank image with the same size as the original image
            reprojected_image = np.zeros_like(image)

            # Copy pixel values from the original image to the reprojected image
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if not(np.isnan(points_2d[y, x]).any()):
                        # Get the projected 2D coordinates of the point
                        point_2d = points_2d[y, x]

                        # Round the 2D coordinates to the nearest pixel
                        point_2d_rounded = np.round(point_2d).astype(int)
                        if point_2d_rounded[1]< image.shape[0] and point_2d_rounded[1] >=0 and point_2d_rounded[0]< image.shape[1] and point_2d_rounded[0] >= 0:
                            # Copy the RGB pixel value from the original image to the reprojected image
                            reprojected_image[point_2d_rounded[1], point_2d_rounded[0]] = image[y, x]

            # Display the reprojected image
            plt.imshow(reprojected_image)
            plt.show()
            cv2.imwrite(f"Data/set_{j}/synth_{i+1}.jpg", reprojected_image)
            print(i+1,"/",11)

