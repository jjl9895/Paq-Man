from scipy.spatial.distance import euclidean
import numpy as np
import cv2
import imutils
from imutils import perspective
from imutils import contours

# Function to display an array of images
def display_images(image_list):
    for index, image in enumerate(image_list):
        cv2.imshow("image_" + str(index), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "images/PaqmanTest.JPG"

# Read and preprocess the image
input_image = cv2.imread(image_path)

grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(grayscale_image, (9, 9), 0)

edge_detected_image = cv2.Canny(blurred_image, 50, 100)
edge_detected_image = cv2.dilate(edge_detected_image, None, iterations=1)
edge_detected_image = cv2.erode(edge_detected_image, None, iterations=1)

# Find image contours
found_contours = cv2.findContours(edge_detected_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
found_contours = imutils.grab_contours(found_contours)

# Sort contours from left to right; leftmost contour is the reference object
(sorted_contours, _) = contours.sort_contours(found_contours)

# Filter out small contours based on area
filtered_contours = [contour for contour in sorted_contours if cv2.contourArea(contour) > 100]

# Reference object dimensions; a 3cm x 3cm box is used for reference
reference_contour = filtered_contours[0]
min_area_rect = cv2.minAreaRect(reference_contour)
rect_points = cv2.boxPoints(min_area_rect)
rect_points = np.array(rect_points, dtype="int")
ordered_rect_points = perspective.order_points(rect_points)
(top_left, top_right, bottom_right, bottom_left) = ordered_rect_points


dist_in_pixel = euclidean(top_left, top_right)
dist_in_cm = 3
pixel_per_cm = dist_in_pixel/dist_in_cm

# Draw remaining contours
for cnt in filtered_contours:

	# Make a box
	box = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)

	(tl, tr, br, bl) = box

	cv2.drawContours(input_image, [box.astype("int")], -1, (0, 0, 255), 2)

	# Horizontal and vertical midpoints
	horizontal_mp = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
	vertical_mp = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))

	w = euclidean(tl, tr)/pixel_per_cm
	h = euclidean(tr, br)/pixel_per_cm

	cv2.putText(input_image, "{:.1f}cm".format(w), (int(horizontal_mp[0] - 15), int(horizontal_mp[1] - 10)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
	cv2.putText(input_image, "{:.1f}cm".format(h), (int(vertical_mp[0] + 10), int(vertical_mp[1])),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

display_images([input_image])