####################
from matplotlib import colors


def ransac_loop(matches, keypoints1, keypoints2, num_iterations=100, tolerance=10):
    best_residual_error = np.inf
    best_inliers = []
    best_transformation = None

    for i in range(num_iterations):
        # Choose 4 random matches
        indices = np.random.choice(matches.shape[0], 4, replace=False)
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches[indices]])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches[indices]])

        # Compute the transformation matrix
        transformation, _ = cv2.getAffineTransform(src_pts, dst_pts, cv2.RANSAC, tolerance)

        # Calculate residual error for all matches
        errors = []
        inliers = []
        for j, match in enumerate(matches):
            src_pt = np.array([keypoints1[match.queryIdx].pt])
            dst_pt = np.array([keypoints2[match.trainIdx].pt])
            predicted_dst_pt = cv2.perspectiveTransform(src_pt.reshape(-1, 1, 2), transformation).reshape(2)
            error = np.linalg.norm(dst_pt - predicted_dst_pt)
            errors.append(error)
            if error < tolerance:
                inliers.append(j)

        residual_error = np.mean(errors)

        # Update best transformation if the current transformation is better
        if residual_error < best_residual_error:
            best_residual_error = residual_error
            best_inliers = inliers
            best_transformation = transformation

    return best_transformation, best_inliers, best_residual_error


def ransac_affine(src_points, dst_points, max_iterations=1000, distance_threshold=10):
    """
    Performs RANSAC algorithm to calculate the best affine transformation matrix that maps src_points to dst_points.

    :param src_points: array of source points (Nx2)
    :param dst_points: array of destination points (Nx2)
    :param max_iterations: maximum number of iterations for RANSAC algorithm
    :param distance_threshold: maximum allowable distance between a point and its transformed point
    :return: best_affine_mat: the best affine transformation matrix (2x3)
    """
    best_affine_mat = None
    best_inlier_count = 0

    for i in range(max_iterations):
        # Randomly select 3 pairs of points
        indices = np.random.choice(len(src_points), size=3, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]

        # Calculate affine transformation matrix
        src_x = src_sample[:, 0]
        src_y = src_sample[:, 1]
        A = np.vstack([src_x, src_y, np.ones(len(src_x))]).T
        B = dst_sample.flatten()
        affine_mat, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        affine_mat = np.append(affine_mat, [0, 0, 1]).reshape(3, 3)

        # Transform all source points using the calculated matrix
        transformed_points = np.zeros_like(src_points)
        for j, point in enumerate(src_points):
            transformed_points[j] = affine_mat[:2, :2] @ point + affine_mat[:2, 2]

        # Calculate the residual errors for each point
        errors = np.sqrt(np.sum((dst_points - transformed_points) ** 2, axis=1))

        # Count inliers that are within distance_threshold
        inlier_count = np.sum(errors < distance_threshold)

        # If current inlier count is higher than the best so far, update the best matrix and inlier count
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_affine_mat = affine_mat

    return best_affine_mat


def transform_points(points, affine_mat):
    """
    Transforms an array of points using the given affine transformation matrix.

    :param points: array of points to be transformed (Nx2)
    :param affine_mat: 2D affine transformation matrix (2x3)
    :return: transformed_points: array of transformed points (Nx2)
    """
    transformed_points = np.zeros_like(points)
    for i, point in enumerate(points):
        transformed_points[i] = affine_mat[:2, :2] @ point + affine_mat[:2, 2]
    return transformed_points



def transform_points_homography(points, homography_mat):
    """
    Transforms an array of points using the given homography transformation matrix.

    :param points: array of points to be transformed (Nx2)
    :param homography_mat: 2D homography transformation matrix (3x3)
    :return: transformed_points: array of transformed points (Nx2)
    """
    # Add homogeneous coordinate to each point
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    #xyw = np.matmul(homography_mat, points_homogeneous.T)
    # Perform homography transformation
    # print(homography_mat)
    transformed_homogeneous = homography_mat @ points_homogeneous.T

    # Convert back to Euclidean coordinate
    transformed_points = (transformed_homogeneous[:2, :] / transformed_homogeneous[2, :]).T

    return transformed_points
def ransac_affine1(src_pts, dst_pts, max_iter=1000, inlier_threshold=2):
    best_affine = None
    best_inliers = []
    for i in range(max_iter):
        # Randomly select 3 points from both sets of points
        idx = np.random.choice(src_pts.shape[0], 3, replace=False)
        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]

        # Calculate affine transformation matrix
        affine = cv2.getAffineTransform(src_sample, dst_sample)
        if affine is None:
            continue

        # Calculate transformed points using the affine matrix
        # transformed_pts = np.dot(affine, np.vstack((src_pts.T, np.ones((1, src_pts.shape[0])))))
        # transformed_pts = transformed_pts[:2].T / transformed_pts[2].T
        transformed_pts=transform_points(src_pts,affine)

        # Calculate residuals and inliers
        residuals = np.sqrt(np.sum((transformed_pts - dst_pts) ** 2, axis=1))
        inliers = np.where(residuals < inlier_threshold)[0]

        # Update best affine transformation and inliers
        if len(inliers) > len(best_inliers) and len(inliers)>=3:
            best_affine = affine
            best_inliers = inliers

    # Calculate final affine transformation matrix using all inliers
    #final_affine = cv2.getAffineTransform(src_pts[best_inliers], dst_pts[best_inliers])

    return best_affine,len(best_inliers)


def ransac_homography(src_pts, dst_pts, max_iter=1000, inlier_threshold=2):
    best_homograph = None
    best_inliers = []
    for i in range(max_iter):
        # Randomly select 4 points from both sets of points
        idx = np.random.choice(src_pts.shape[0], 4, replace=False)
        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]

        # Calculate affine transformation matrix
        homograph = cv2.findHomography(src_sample, dst_sample)
        if homograph[0] is None:
            continue
        # Calculate transformed points using the affine matrix
        # transformed_pts = np.dot(affine, np.vstack((src_pts.T, np.ones((1, src_pts.shape[0])))))
        # transformed_pts = transformed_pts[:2].T / transformed_pts[2].T
        # print(src_sample)
        # print(dst_sample)
        # print(homograph)
        transformed_pts=transform_points_homography(src_pts,homograph[0])

        # Calculate residuals and inliers
        residuals = np.sqrt(np.sum((transformed_pts - dst_pts) ** 2, axis=1))
        inliers = np.where(residuals < inlier_threshold)[0]

        # Update best affine transformation and inliers
        if len(inliers) > len(best_inliers) and len(inliers)>=4:
            best_homograph = homograph
            best_inliers = inliers

    # Calculate final affine transformation matrix using all inliers
    #final_affine = cv2.getAffineTransform(src_pts[best_inliers], dst_pts[best_inliers])

    return best_homograph,len(best_inliers)
################################################################################################################
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# # Initialize SIFT detector
def homograph(pieces_path,transform_path,dimension,output_dir,ratio_thresh=0.7,ransac_iterations=1000,inlier_threshold=2,min_inlier=4):
    sift = cv2.SIFT_create()
    # Load puzzle pieces
    #pieces_dir = "puzzles/puzzle_homography_6/pieces"
    pieces_dir = pieces_path
    pieces = []
    for filename in os.listdir(pieces_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            piece = cv2.imread(os.path.join(pieces_dir, filename))
            pieces.append(piece)
    # Load transformation
    transform_file = transform_path
    with open(transform_file, "r") as f:
        data = f.readlines()
        warp_mat = np.array([list(map(float, line.strip().split())) for line in data])
    #warp_mat=np.delete(warp_mat,-1,axis=0)
    l = len(pieces)
    onesimage =np.ones((pieces[0].shape[0], pieces[0].shape[1],3), dtype=np.uint8)
    pieces[0] = cv2.warpPerspective(pieces[0],warp_mat,dimension,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_TRANSPARENT)
    onesimage = cv2.warpPerspective(onesimage,warp_mat,dimension,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_TRANSPARENT)
    result = pieces[0].copy()
    wrpd = np.zeros(len(pieces), dtype=bool)
    wrpd[0]=True
    # Ratio test parameters
    kps=[]
    dess=[]
    for i in range(len(pieces)):
        # pieces[i]=cv2.resize(pieces[i], dimension)
        k,d=sift.detectAndCompute(cv2.cvtColor(pieces[i], cv2.COLOR_BGR2GRAY), None)
        kps.append(k)
        dess.append(d)
    ratio_threshold = ratio_thresh
    coverage_count = np.zeros((result.shape[0], result.shape[1],3), dtype=np.uint8)
    coverage_count[onesimage[:, :, 0] > 0] += 1
    done=1
    while not (np.all(wrpd == True)):
        # distance matrix calculation

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        best_inl=min_inlier
        best_index=-1
        best_transformation=[]
        print("solved :",done)
        ## loop over all pieces
        for j in range(len(pieces)):
            if wrpd[j] == False:
                #gray= cv2.cvtColor(pieces[j],cv2.COLOR_BGR2GRAY)
                kp_target, des_target = kps[j],dess[j]
                # Calculate distance matrix

                dist_matrix = np.linalg.norm(des[:, np.newaxis] - des_target, axis=2)
                good_matches = []
                # Loop through each descriptor in des0 and compare to closest descriptors in des1
                for i, descriptor in enumerate(des):
                    # Calculate distance to closest and second closest descriptors
                    distances = dist_matrix[i]
                    sorted_distances_idx = np.argsort(distances)
                    closest_distance = distances[sorted_distances_idx[0]]
                    second_closest_distance = distances[sorted_distances_idx[1]]

                    # Check if the match passes the ratio test
                    if closest_distance / second_closest_distance < ratio_threshold:
                        # Save the index of the matching descriptor in des1
                        match_idx = sorted_distances_idx[0]
                        good_matches.append((i, match_idx))

        # # Draw matching lines on image
        # img_matches = cv2.drawMatches(pieces[0], kps[0], pieces[1], kps[1], [cv2.DMatch(_[0], _[1], 0) for _ in good_matches], None)
        #
        # # Show image with matches
        # cv2.imshow("Matches", img_matches)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


            # Compute the affine transformation using OpenCV's getAffineTransform() function
                src_pts = np.float32([kp[m[0]].pt for m in good_matches])#.reshape(-1,  2)
                dst_pts = np.float32([kp_target[m[1]].pt for m in good_matches])#.reshape(-1,  2)
                if len(src_pts)<4 or len(dst_pts)<4:
                    continue
                M,inl= ransac_homography(dst_pts,src_pts,max_iter=ransac_iterations,inlier_threshold=inlier_threshold)
                if inl>=best_inl and not( (M is None) or  (M[0]is None)):
                    best_inl = inl
                    best_index = j
                    best_transformation = M[0]

        # Print the resulting transformation matrix
        # print("Affine transformation matrix:")
        # print(M)
        if best_index == -1:
            print("detect failed")
            break
        # ones_image= np.ones((pieces[best_index].shape[0], pieces[best_index].shape[1],3), dtype=np.uint8)
        mask = np.ones_like(pieces[best_index])
        pieces[best_index] = cv2.warpPerspective(pieces[best_index],best_transformation,(result.shape[1],result.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_TRANSPARENT)
        mask=cv2.warpPerspective(mask,best_transformation,(result.shape[1],result.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_TRANSPARENT)
        # Stitch warped puzzle pieces together

        # coverage_count[mask [:, :, 0] > 0] += 30
        coverage_count[mask== 1] += 1
        # result[pieces[best_index] != 0] = pieces[best_index][pieces[best_index] != 0]
        #result[mask == 1] = pieces[best_index][mask == 1]
        gray_img_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray_img_piece = cv2.cvtColor(pieces[best_index], cv2.COLOR_BGR2GRAY)
        for i1 in range (result.shape[0]):
            for i2 in range (result.shape[1]):
                 if gray_img_res[i1][i2] == 0 and gray_img_piece[i1][i2]!=0:
                    result[i1][i2] = pieces[best_index][i1][i2]
                 # if gray_img_res[i1][i2] != 0 and gray_img_piece[i1][i2] != 0:
                 #    result[i1][i2] =(result[i1][i2]+ pieces[best_index][i1][i2])/2


        plt.imshow(result)
        plt.show()
        wrpd[best_index]=True
        done += 1
    # Display results
    plt.imshow(result)
    plt.show()
    plt.imshow(cv2.cvtColor(coverage_count, cv2.COLOR_BGR2GRAY), vmin=coverage_count.min(), vmax=coverage_count.max())
    plt.colorbar()
    plt.savefig(output_dir+"coverage.jpeg")
    plt.show()
    output_dir_res = output_dir+"solution_"+str(done)+"_"+str(l)+".jpeg"
    cv2.imwrite(output_dir_res, result)
    for i in range(len(wrpd)):
        if wrpd[i]==True:
            output_dir_res = output_dir+"piece_"+str(i+1)+"_relative.jpeg"
            cv2.imwrite(output_dir_res, pieces[i])
# cv2.imshow("coverage", coverage_count)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# new_image = Image.new(result,(699, 549))

# Draw on the image (optional)
# ...

# Save the image to a file
# cv2.imwrite('puzzles/puzzle_homography_1/image.png', result)

def affine(pieces_path,transform_path,dimension,output_dir,ratio_thresh=0.7,ransac_iterations=1000,inlier_threshold=2,min_inlier=10):
    # # Initialize SIFT detector
    sift = cv2.SIFT_create()
    # Load puzzle pieces
    pieces_dir = pieces_path
    pieces = []
    for filename in os.listdir(pieces_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            piece = cv2.imread(os.path.join(pieces_dir, filename))
            pieces.append(piece)
    # Load transformation
    transform_file = transform_path
    with open(transform_file, "r") as f:
        data = f.readlines()
        warp_mat = np.array([list(map(float, line.strip().split())) for line in data])
    warp_mat = np.delete(warp_mat, -1, axis=0)
    onesimage = np.ones((pieces[0].shape[0], pieces[0].shape[1], 3), dtype=np.uint8)
    pieces[0] = cv2.warpAffine(pieces[0], warp_mat, dimension, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_TRANSPARENT)
    onesimage = cv2.warpAffine(onesimage, warp_mat, dimension, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_TRANSPARENT)
    print("solved :", 1)
    l=len(pieces)
    result = pieces[0].copy()
    wrpd = np.zeros(len(pieces), dtype=bool)
    wrpd[0]=True
    plt.imshow(result)
    plt.show()
    # Ratio test parameters
    kps = []
    dess = []
    for i in range(len(pieces)):
        #pieces[i]=cv2.resize(pieces[i], dimension)
        k, d = sift.detectAndCompute(cv2.cvtColor(pieces[i], cv2.COLOR_BGR2GRAY), None)
        kps.append(k)
        dess.append(d)
    ratio_threshold = ratio_thresh
    coverage_count = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    #coverage_count [:,:,2]+=100
    coverage_count[onesimage[:, :, 0] > 0] += 1
    done = 1
    while not (np.all(wrpd == True)):
        # distance matrix calculation

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        best_inl = min_inlier
        best_index = -1
        best_transformation = []

        ## loop over all pieces
        for j in range(len(pieces)):
            if wrpd[j]==False:
                # gray= cv2.cvtColor(pieces[j],cv2.COLOR_BGR2GRAY)
                kp_target, des_target = kps[j], dess[j]
                # Calculate distance matrix
                dist_matrix = np.linalg.norm(des[:, np.newaxis] - des_target, axis=2)
                # dist_matrix = []
                # for i1 in range(des.shape[0]):
                #     row = []
                #     for j1 in range(des_target.shape[0]):
                #         dist = 0
                #         for k1 in range(des.shape[1]):
                #             dist += (des[i1][k1] - des_target[j1][k1]) ** 2
                #         row.append(math.sqrt(dist))
                #     dist_matrix.append(row)
                good_matches = []
                # Loop through each descriptor in des0 and compare to closest descriptors in des1
                for i, descriptor in enumerate(des):
                    # Calculate distance to closest and second closest descriptors
                    distances = dist_matrix[i]
                    sorted_distances_idx = np.argsort(distances)
                    closest_distance = distances[sorted_distances_idx[0]]
                    second_closest_distance = distances[sorted_distances_idx[1]]

                    # Check if the match passes the ratio test
                    if closest_distance / second_closest_distance < ratio_threshold:
                        # Save the index of the matching descriptor in des1
                        match_idx = sorted_distances_idx[0]
                        good_matches.append((i, match_idx))

                # # Draw matching lines on image
                # img_matches = cv2.drawMatches(pieces[0], kps[0], pieces[1], kps[1], [cv2.DMatch(_[0], _[1], 0) for _ in good_matches], None)
                #
                # # Show image with matches
                # cv2.imshow("Matches", img_matches)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Compute the affine transformation using OpenCV's getAffineTransform() function
                src_pts = np.float32([kp[m[0]].pt for m in good_matches])  # .reshape(-1,  2)
                dst_pts = np.float32([kp_target[m[1]].pt for m in good_matches])  # .reshape(-1,  2)
                if len(src_pts) < 3 or len(dst_pts) < 3:
                    continue
                M, inl = ransac_affine1(dst_pts, src_pts,max_iter=ransac_iterations,inlier_threshold=inlier_threshold)
                if inl >= best_inl and not (M is None):
                    best_inl = inl
                    best_index = j
                    best_transformation = M
        # Print the resulting transformation matrix
        # print("Affine transformation matrix:")
        # print(M)
        if best_index == -1:
            print("detect failed")
            break
        ones_image = np.ones((pieces[best_index].shape[0], pieces[best_index].shape[1], 3), dtype=np.uint8)
        pieces[best_index] = cv2.warpAffine(pieces[best_index], best_transformation, (result.shape[1], result.shape[0]),
                                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

        ones_image = cv2.warpAffine(ones_image, best_transformation, (result.shape[1], result.shape[0]),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        # Stitch warped puzzle pieces together
        # plt.imshow(pieces[best_index])
        # plt.show()
        coverage_count[ones_image[:, :, 0] > 0] += 1
        gray_img_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray_img_piece = cv2.cvtColor(pieces[best_index], cv2.COLOR_BGR2GRAY)
        for i1 in range(result.shape[0]):
            for i2 in range(result.shape[1]):
                if gray_img_res[i1][i2] == 0 and gray_img_piece[i1][i2] != 0:
                    result[i1][i2] = pieces[best_index][i1][i2]
        # Stack the images vertically
        # result = cv2.addWeighted(result, 0.5, pieces[best_index], 0.5, 0)
        plt.imshow(result)
        plt.show()
        plt.imshow(pieces[best_index])
        plt.show()
        plt.imshow(cv2.cvtColor(coverage_count, cv2.COLOR_BGR2GRAY), vmin=coverage_count.min(),
                   vmax=coverage_count.max())
        plt.colorbar()
        plt.show()
        # del pieces[best_index]
        # del kps[best_index]
        # del dess[best_index]
        wrpd[best_index]=True
        done += 1
        print("solved :", done)
    # Display results
    # Apply a color map to the coverage count image
    plt.imshow(result)
    plt.show()





    plt.imshow(cv2.cvtColor(coverage_count, cv2.COLOR_BGR2GRAY), vmin=coverage_count.min(), vmax=coverage_count.max())
    plt.colorbar()
    plt.savefig(output_dir+"coverage.jpeg")
    plt.show()
    # # Save the plot

    output_dir_res = output_dir+"solution_"+str(done)+"_"+str(l)+".jpeg"
    cv2.imwrite(output_dir_res, result)
    for i in range(len(wrpd)):
        if wrpd[i]==True:
            output_dir_res = output_dir+"piece_"+str(i+1)+"_relative.jpeg"
            cv2.imwrite(output_dir_res, pieces[i])

