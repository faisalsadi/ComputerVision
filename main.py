import cv2

# Load puzzle piece image
img = cv2.imread(r"puzzles/puzzle_affine_1/pieces/piece_2.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp, des = sift.detectAndCompute(gray, None)

# Draw keypoints on image
img_kp = cv2.drawKeypoints(img, kp, None)

# Show image with keypoints
cv2.imshow('Keypoints', img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()