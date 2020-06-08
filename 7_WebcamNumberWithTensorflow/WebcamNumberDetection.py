import cv2
import numpy as np
import math
import sys
import os

###################################################
###################################################
##                                               ##
##              BACKGROUND REMOVAL               ##
##                                               ##
###################################################
###################################################

# This function use a Background Substractor to keep
#   the Foreground in a image
def getForeground(img, bgSubstractor, learningRate = 0):
    fgmask = bgSubstractor.apply(img, learningRate = learningRate)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    
    return cv2.bitwise_and(img, img, mask=fgmask)

###################################################
###################################################
##                                               ##
##                 SKIN DETECTION                ##
##                                               ##
###################################################
###################################################

face_cascade = cv2.CascadeClassifier("res\\haarcascade_frontalface_alt.xml")
(low_H, low_S, low_V) = (0, 0, 0)
(high_H, high_S, high_V) = (0, 0, 0)

def calibrate(img, r1, r2):
    global low_H, low_S, low_V, high_H, high_S, high_V
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    (x1, y1, w1, h1) = r1
    (x2, y2, w2, h2) = r2
    
    offsetLowThreshold = 80
    offsetHighThreshold = 30
    
    sample1 = cv2.mean(hsv[x1:x1+w1, y1:y1+h1])
    sample2 = cv2.mean(hsv[x2:x2+w2, y2:y2+h2])
    
    low_H = min(sample1[0], sample2[0]) - offsetLowThreshold
    high_H = max(sample1[0], sample2[0]) + offsetHighThreshold
    
    low_S = min(sample1[1], sample2[1]) - offsetLowThreshold
    high_S = max(sample1[1], sample2[1]) + offsetHighThreshold
    
    low_V = min(sample1[2], sample2[2]) - offsetLowThreshold
    high_V = max(sample1[2], sample2[2]) + offsetHighThreshold

def regionOfSample(img, width, height):
    size = 20
    
    r1 = (int(width / 5), int(height / 2), size, size)
    r2 = (int(width / 5), int(height / 3), size, size)

    return r1, r2

def removeFace(img, out):
    global face_cascade
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.equalizeHist(gray)

    scaleFactor = 1.1
    minNeighbors = 2
    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors, 0 | cv2.CASCADE_SCALE_IMAGE)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(out, (x, y), (x+w, y+h), 0, -1)

def skinMask(img):
    global low_H, low_S, low_V, high_H, high_S, high_V
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
    structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3));
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, structuringElement);
    
    mask = cv2.dilate(mask, (-1, -1), 3)
    
    return mask;
    

###################################################
###################################################
##                                               ##
##               FINGERS DETECTION               ##
##                                               ##
###################################################
###################################################

LIMIT_ANGLE_SUP = 60
LIMIT_ANGLE_INF = 5
BOUNDING_RECT_FINGER_SIZE_SCALING = 0.3
BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING = 0.05

def distancePoints(a, b):
	return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def distancePointsX(a, b):
	return math.fabs(a[0] - a[1])

def computeAngle(a, b, c):
	ab = distancePoints(a, b);
	bc = distancePoints(b, c);
	ac = distancePoints(a, c);
    
	return math.acos((ab ** 2 + bc ** 2 - ac ** 2) / (2 * ab * bc))

def compactOnNeighborhoodMedian(points, max_neighbor_distance):
    res = []
    
    if type(points) == type(None) or len(points) == 0:
        return res
    
    reference = points[0]
    median = points[0]
    
    for i in range(len(points)):
        if (distancePoints(reference, points[i]) > max_neighbor_distance):
            res = res + [median]
			
            reference = points[i]
            median = points[i]
		
        else:
            tmp = points[i] + median
            median = (tmp[0] / 2, tmp[1])
            
    res = res + [median]
    
    return res

def findClosestOnX(points, pivot):

    distance_x_1 = float('inf')
    distance_1 = float('inf')
    distance_x_2 = float('inf')
    distance_2 = float('inf')
    index_1 = 0
    index_2 = 0
    
    for i in range(len(points)):
        distance_x = distancePointsX(pivot, points[i])
        distance = distancePoints(pivot, points[i])

        if (distance_x < distance_x_1 and distance_x != 0 and distance <= distance_1):
            distance_x_1 = distance_x
            distance_1 = distance
            index_1 = i

    for i in range(len(points)):
        distance_x = distancePointsX(pivot, points[i]);
        distance = distancePoints(pivot, points[i]);
        
        if (distance_x < distance_x_2 and distance_x != 0 and distance <= distance_2 and distance_x != distance_x_1):
            distance_x_2 = distance_x
            distance_2 = distance
            index_2 = i
    
    return (points[index_1], points[index_2])
    
def findBigestContour(contours):
    maxArea = -1
    for i in range(len(contours)):
        temp = contours[i]
        area = cv2.contourArea(temp)
        if area > maxArea:
            maxArea = area
            cIndex = i
        
    return contours[cIndex], cIndex

def isFinger(a, b, c, limit_angle_inf, limit_angle_sup, center_hand, min_distance_from_center_hand):
    angle = computeAngle(a, b, c)
	
    if (angle > limit_angle_sup or angle < limit_angle_inf):
        return False

    # the finger point should not be under the two far points
    delta_y_1 = b[1] - a[1]
    delta_y_2 = b[1] - c[1]
    if (delta_y_1 > 0 and delta_y_2 > 0):
        return False

	# the two far points should not be both under the center of the hand
    delta_y_3 = center_hand[1] - a[1]
    delta_y_4 = center_hand[1] - c[1]
    if (delta_y_3 < 0 and delta_y_4 < 0):
        return False
    
    distance_from_center_hand = distancePoints(b, center_hand);
    if (distance_from_center_hand < min_distance_from_center_hand):
        return False
	
	# When no fingers are up
    distance_from_center_hand_far_1 = distancePoints(a, center_hand)
    distance_from_center_hand_far_2 = distancePoints(c, center_hand)
    if (distance_from_center_hand_far_1 < min_distance_from_center_hand / 4 or distance_from_center_hand_far_2 < min_distance_from_center_hand / 4):
        return False
    
    return True

def fingersCount(mask, frame):    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if type(contours) == type(None) or (len(contours) <= 0):
        return None

    contour, indexContour = findBigestContour(contours)
    
    #hull_points = cv2.convexHull(contour, returnPoints = True)
    hull_ints = cv2.convexHull(contour, returnPoints = False)
    
    if type(hull_ints) != type(None) and len(hull_ints) > 3:
        defects = cv2.convexityDefects(contour, hull_ints)
    else:
        return None
    
    box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
    
    center_hand = ( int((box_x + box_w) / 2), int((box_y + box_h) / 2))

    starts = []
    fars = []
    
    for i in range(len(defects)):
        s, e, f, d = defects[i][0]
        starts = starts + [tuple(contour[s][0])]

        if (distancePoints(contour[f][0], center_hand) < box_h * BOUNDING_RECT_FINGER_SIZE_SCALING):
            fars = fars + [tuple(contour[f][0])]
            
    filtered_starts = compactOnNeighborhoodMedian(starts, box_h * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING)
    filtered_fars = compactOnNeighborhoodMedian(fars, box_h * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING)

    if (len(filtered_fars) > 1):
		
        fingers = []
        
        for i in range(len(filtered_starts)):
            closest_point = findClosestOnX(filtered_fars, filtered_starts[i]);
			
            if (isFinger(closest_point[0], filtered_starts[i], closest_point[1], LIMIT_ANGLE_INF, LIMIT_ANGLE_SUP, center_hand, box_h * BOUNDING_RECT_FINGER_SIZE_SCALING)):
                 fingers + [filtered_starts[i]]

        if len(fingers) > 0:

            filtered_fingers = []
            for i in range(len(fingers) - 1):
                if (distancePointsX(fingers[i], fingers[i + 1]) > box_h * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING * 1.5):
                    filtered_fingers = filtered_fingers + [fingers[i]]
            
            if len(fingers) > 2:
                if (distancePointsX(fingers[0], fingers[len(fingers) - 1]) > box_h * BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING * 1.5):
                    filtered_fingers = filtered_fingers + [fingers[len(fingers) - 1]]
            else:
                filtered_fingers = filtered_fingers + [fingers[len(fingers) - 1]]
                
    
            cv2.drawContours(frame, contours, indexContour, (0,0,255), 2, 8, hierarchy)

            return len(filtered_fingers)
        
    return -1

##################################################
##################################################
##                                              ##
##                     MAIN                     ##
##                                              ##
##################################################
##################################################

def main():
    
    # Background substraction threshold
    bgSubThreshold = 30
    isBgSubstractorRdy = False
    
    calibrated = False
    
    bg = None
    
    video = cv2.VideoCapture(0)
    
    try:
        while (True):
            # Read the next frame
            ret, frame = video.read()            
            
            # Skin detection
            r1, r2 = regionOfSample(frame, video.get(3), video.get(4))
            cv2.rectangle(frame, (r1[0], r1[1]), (r1[2], r1[3]), (255, 0, 255), 2)
            cv2.rectangle(frame, (r2[0], r2[1]), (r2[2], r2[3]), (255, 0, 255), 2)
            
            # Get foreGround
            if (isBgSubstractorRdy):
                foreground = getForeground(frame, bg)
                
                if (calibrated):
                    # Add black squares to the face
                    removeFace(frame, foreground)
                
                    handMask = skinMask(foreground)
                    cv2.imshow("handMask", handMask)
                    
                    fingerCount = fingersCount(handMask, frame)
                
                #cv2.imshow("foreground", foreground)
    
            cv2.imshow("output", frame)
            
            k = cv2.waitKey(1)
            if not ret or k == 27:  # press ESC to exit
                video.release()
                cv2.destroyAllWindows()
                print("Released Video Resource")
                break
            elif k == ord('b'):  # press 'b' to capture the background
                bg = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold, detectShadows = False)
                isBgSubstractorRdy = True
                print("Background Subtractor ready!")
            elif k == ord('s'):  # press 's' to calibrate skinDetector
                calibrate(frame, r1, r2)
                calibrated = True
                print("Skin detector ready!")
    except:
        video.release()
        cv2.destroyAllWindows()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_obj, exc_tb.tb_lineno)
        print("Released Video Resource")

if __name__ == '__main__':
    main()