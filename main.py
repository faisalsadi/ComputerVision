# import cv2
#
# # Load puzzle piece image
# img = cv2.imread(r"puzzles/puzzle_affine_1/pieces/piece_2.jpg")
#
# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Initialize SIFT detector
# sift = cv2.SIFT_create()
#
# # transform_file = "puzzles/puzzle_affine_1/warp_mat_1__H_521__W_760_.txt"
# # with open(transform_file, "r") as f:
# #     data = f.readlines()
# #     warp_mat = np.array([list(map(float, line.strip().split())) for line in data])
#
# # Detect keypoints and compute descriptors
# kp, des = sift.detectAndCompute(gray, None)
#
# # Draw keypoints on image
# img_kp = cv2.drawKeypoints(img, kp, None)
#
# # Show image with keypoints
# cv2.imshow('Keypoints', img_kp)
# cv2.waitKey(0)
# cv2.destroyAllWinwdows()

############################################################









####################
import cv2
import numpy as np
import os
# # Initialize SIFT detector
sift = cv2.SIFT_create()
# Load puzzle pieces
pieces_dir = "puzzles/puzzle_affine_1/pieces"
pieces = []
for filename in os.listdir(pieces_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        piece = cv2.imread(os.path.join(pieces_dir, filename))
        pieces.append(piece)

# Load transformation
transform_file = "puzzles/puzzle_affine_1/warp_mat_1__H_521__W_760_.txt"
with open(transform_file, "r") as f:
    data = f.readlines()
    warp_mat = np.array([list(map(float, line.strip().split())) for line in data])
img0 = cv2.imread(r"puzzles/puzzle_affine_1/pieces/piece_1.jpg")

# Convert to grayscale
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)


img1 = cv2.imread(r"puzzles/puzzle_affine_1/pieces/piece_2.jpg")

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)



# Detect keypoints and compute descriptors
kp0, des0 = sift.detectAndCompute(gray0, None)

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)

# Apply transformation to first puzzle piece
first_piece = pieces[0]
transformed_piece = cv2.warpPerspective(first_piece, warp_mat, (760, 521))
first_piece1 = pieces[1]
transformed_piece1 = cv2.warpPerspective(first_piece1, warp_mat, (760, 521))

# numpy_horizontal_concat = np.concatenate((transformed_piece, transformed_piece1), axis=1)
# # Display transformed puzzle piece
# cv2.imshow("Transformed Piece", numpy_horizontal_concat)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# distance matrix calculation
# Calculate distance matrix
dist_matrix = np.linalg.norm(des0[:, np.newaxis] - des1, axis=2)
# Set ratio test threshold
# ratio_thresh = 0.75
#
# # Find two closest matches for each keypoint in first image
# matches = np.argpartition(dist_matrix, 2, axis=1)[:, :2]
#
# # Initialize arrays to store matches that pass ratio test
# good_matches = []
# idx1 = []
# idx2 = []
#
# # Loop through all keypoints in first image
# for i in range(matches.shape[0]):
#     # Get distances to closest and second closest matches
#     dist1 = dist_matrix[i, matches[i, 0]]
#     dist2 = dist_matrix[i, matches[i, 1]]
#
#     # Check if ratio of distances is less than threshold
#     if dist1 / dist2 < ratio_thresh:
#         # Store match if it passes ratio test
#         good_matches.append(cv2.DMatch(i, matches[i, 0], dist1))
#         idx1.append(matches[i, 0])
#         idx2.append(i)
#
# # Convert matches to array
# good_matches = np.array(good_matches)

# Ratio test parameters
ratio_threshold = 0.4
good_matches = []

# Loop through each descriptor in des0 and compare to closest descriptors in des1
for i, descriptor in enumerate(des0):
    # Calculate distance to closest and second closest descriptors
    distances = np.linalg.norm(descriptor - des1, axis=1)
    sorted_distances_idx = np.argsort(distances)
    closest_distance = distances[sorted_distances_idx[0]]
    second_closest_distance = distances[sorted_distances_idx[1]]

    # Check if the match passes the ratio test
    if closest_distance / second_closest_distance < ratio_threshold:
        # Save the index of the matching descriptor in des1
        match_idx = sorted_distances_idx[0]
        good_matches.append((i, match_idx))

# Draw matching lines on image
img_matches = cv2.drawMatches(img0, kp0, img1, kp1, [cv2.DMatch(_[0], _[1], 0) for _ in good_matches], None)

# Show image with matches
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Define the coordinates of 3 matches in the source and destination images
img1_pt1=good_matches[1][0]
img2_pt1=good_matches[1][1]

img1_pt2=good_matches[2][0]
img2_pt2=good_matches[2][1]

img1_pt3=good_matches[3][0]
img2_pt3=good_matches[3][1]

q=kp1[img1_pt1].pt

src_pts = np.float32([kp0[img1_pt1].pt, kp0[img1_pt2].pt, kp0[img1_pt3].pt])
dst_pts = np.float32([kp1[img2_pt1].pt, kp1[img2_pt2].pt, kp1[img2_pt3].pt])

# Compute the affine transformation using OpenCV's getAffineTransform() function
M = cv2.getAffineTransform(src_pts, dst_pts)

# Print the resulting transformation matrix
print("Affine transformation matrix:")
print(M)
x=5


#
#

