# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 12:36:57 2019

@author: Abdul Ahad
"""
def pixel_distance(A, B):
    (col_A, row_A) = A
    (col_B, row_B) = B
    return math.sqrt(math.pow(col_B - col_A, 2) + math.pow(row_B - row_A, 2))
def get_angle(A, B, C):
    (col_A, row_A) = A
    (col_B, row_B) = B
    (col_C, row_C) = C
    a = pixel_distance(C, B)
    b = pixel_distance(A, C)
    c = pixel_distance(A, B)

    try:
        cos_angle = (math.pow(a, 2) + math.pow(b, 2) - math.pow(c, 2)) / (2 * a * b)
    except ZeroDivisionError as e:
        log.warning("get_angle: A %s, B %s, C %s, a %.3f, b %.3f, c %.3f" % (A, B, C, a, b, c))
        raise e

    # If CA and CB are very long and the angle at C very narrow we can get an
    # invalid cos_angle which will cause math.acos() to raise a ValueError exception
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1

    angle_ACB = math.acos(cos_angle)
    # log.info("get_angle: A %s, B %s, C %s, a %.3f, b %.3f, c %.3f, cos_angle %s, angle_ACB %s" %
    #          (A, B, C, a, b, c, pformat(cos_angle), int(math.degrees(angle_ACB))))
    return angle_ACB


def sort_corners(corner1, corner2, corner3, corner4):
    """
    Sort the corners such that
    - A is top left
    - B is top right
    - C is bottom left
    - D is bottom right

    Return an (A, B, C, D) tuple
    """
    results = []
    corners = (corner1, corner2, corner3, corner4)

    min_x = None
    max_x = None
    min_y = None
    max_y = None

    for (x, y) in corners:
        if min_x is None or x < min_x:
            min_x = x

        if max_x is None or x > max_x:
            max_x = x

        if min_y is None or y < min_y:
            min_y = y

        if max_y is None or y > max_y:
            max_y = y

    # top left
    top_left = None
    top_left_distance = None
    for (x, y) in corners:
        distance = pixel_distance((min_x, min_y), (x, y))
        if top_left_distance is None or distance < top_left_distance:
            top_left = (x, y)
            top_left_distance = distance

    results.append(top_left)

    # top right
    top_right = None
    top_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, min_y), (x, y))
        if top_right_distance is None or distance < top_right_distance:
            top_right = (x, y)
            top_right_distance = distance
    results.append(top_right)

    # bottom left
    bottom_left = None
    bottom_left_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((min_x, max_y), (x, y))

        if bottom_left_distance is None or distance < bottom_left_distance:
            bottom_left = (x, y)
            bottom_left_distance = distance
    results.append(bottom_left)

    # bottom right
    bottom_right = None
    bottom_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, max_y), (x, y))

        if bottom_right_distance is None or distance < bottom_right_distance:
            bottom_right = (x, y)
            bottom_right_distance = distance
    results.append(bottom_right)

    return results

def approx_is_square(approx, SIDE_VS_SIDE_THRESHOLD=0.60, ANGLE_THRESHOLD=20, ROTATE_THRESHOLD=30):
    """
    Rules
    - there must be four corners
    - all four lines must be roughly the same length
    - all four corners must be roughly 90 degrees
    - AB and CD must be horizontal lines
    - AC and BC must be vertical lines

    SIDE_VS_SIDE_THRESHOLD
        If this is 1 then all 4 sides must be the exact same length.  If it is
        less than one that all sides must be within the percentage length of
        the longest side.

    ANGLE_THRESHOLD
        If this is 0 then all 4 corners must be exactly 90 degrees.  If it
        is 10 then all four corners must be between 80 and 100 degrees.

    ROTATE_THRESHOLD
        Controls how many degrees the entire square can be rotated

    The corners are labeled

        A ---- B
        |      |
        |      |
        C ---- D
    """

    assert SIDE_VS_SIDE_THRESHOLD >= 0 and SIDE_VS_SIDE_THRESHOLD <= 1, "SIDE_VS_SIDE_THRESHOLD must be between 0 and 1"
    assert ANGLE_THRESHOLD >= 0 and ANGLE_THRESHOLD <= 90, "ANGLE_THRESHOLD must be between 0 and 90"

    # There must be four corners
    if len(approx) != 4:
        return False

    # Find the four corners
    (A, B, C, D) = sort_corners(tuple(approx[0][0]),
                                tuple(approx[1][0]),
                                tuple(approx[2][0]),
                                tuple(approx[3][0]))

    # Find the lengths of all four sides
    AB = pixel_distance(A, B)
    AC = pixel_distance(A, C)
    DB = pixel_distance(D, B)
    DC = pixel_distance(D, C)
    distances = (AB, AC, DB, DC)
    max_distance = max(distances)
    cutoff = int(max_distance * SIDE_VS_SIDE_THRESHOLD)

    #log.info("approx_is_square A %s, B, %s, C %s, D %s, distance AB %d, AC %d, DB %d, DC %d, max %d, cutoff %d" %
    #         (A, B, C, D, AB, AC, DB, DC, max_distance, cutoff))

    # If any side is much smaller than the longest side, return False
    for distance in distances:
        if distance < cutoff:
            return False

    # all four corners must be roughly 90 degrees
    min_angle = 90 - ANGLE_THRESHOLD
    max_angle = 90 + ANGLE_THRESHOLD

    # Angle at A
    angle_A = int(math.degrees(get_angle(C, B, A)))
    if angle_A < min_angle or angle_A > max_angle:
        return False

    # Angle at B
    angle_B = int(math.degrees(get_angle(A, D, B)))
    if angle_B < min_angle or angle_B > max_angle:
        return False

    # Angle at C
    angle_C = int(math.degrees(get_angle(A, D, C)))
    if angle_C < min_angle or angle_C > max_angle:
        return False

    # Angle at D
    angle_D = int(math.degrees(get_angle(C, B, D)))
    if angle_D < min_angle or angle_D > max_angle:
        return False

    far_left  = min(A[0], B[0], C[0], D[0])
    far_right = max(A[0], B[0], C[0], D[0])
    far_up    = min(A[1], B[1], C[1], D[1])
    far_down  = max(A[1], B[1], C[1], D[1])
    top_left = (far_left, far_up)
    top_right = (far_right, far_up)
    bottom_left = (far_left, far_down)
    bottom_right = (far_right, far_down)
    debug = False


    '''
    if A[0] >= 93 and A[0] <= 96 and A[1] >= 70 and A[1] <= 80:
        debug = True
        log.info("approx_is_square A %s, B, %s, C %s, D %s, distance AB %d, AC %d, DB %d, DC %d, max %d, cutoff %d, angle_A %s, angle_B %s, angle_C %s, angle_D %s, top_left %s, top_right %s, bottom_left %s, bottom_right %s" %
                 (A, B, C, D, AB, AC, DB, DC, max_distance, cutoff,
                  angle_A, angle_B, angle_C, angle_D,
                  pformat(top_left), pformat(top_right), pformat(bottom_left), pformat(bottom_right)))
    '''

    # Is AB horizontal?
    if B[1] < A[1]:
        # Angle at B relative to the AB line
        angle_B = int(math.degrees(get_angle(A, top_left, B)))

        if debug:
            log.info("AB is horizontal, angle_B %s, ROTATE_THRESHOLD %s" % (angle_B, ROTATE_THRESHOLD))

        if angle_B > ROTATE_THRESHOLD:
            if debug:
                log.info("AB horizontal rotation %s is above ROTATE_THRESHOLD %s" % (angle_B, ROTATE_THRESHOLD))
            return False
    else:
        # Angle at A relative to the AB line
        angle_A = int(math.degrees(get_angle(B, top_right, A)))

        if debug:
            log.info("AB is vertical, angle_A %s, ROTATE_THRESHOLD %s" % (angle_A, ROTATE_THRESHOLD))

        if angle_A > ROTATE_THRESHOLD:
            if debug:
                log.info("AB vertical rotation %s is above ROTATE_THRESHOLD %s" % (angle_A, ROTATE_THRESHOLD))
            return False

    # TODO - if the area of the approx is way more than the
    # area of the contour then this is not a square
    return True


def square_width_height(approx):
    """
    This assumes that approx is a square. Return the width and height of the square.
    """
    width = 0
    height = 0

    # Find the four corners
    (A, B, C, D) = sort_corners(tuple(approx[0][0]),
                                tuple(approx[1][0]),
                                tuple(approx[2][0]),
                                tuple(approx[3][0]))

    # Find the lengths of all four sides
    AB = pixel_distance(A, B)
    AC = pixel_distance(A, C)
    DB = pixel_distance(D, B)
    DC = pixel_distance(D, C)

    width = max(AB, DC)
    height = max(AC, DB)
    if (width>140 and height >140):
        print(A,B,C,D)
        cv2.rectangle(img,(A[0],A[1]),(D[0],D[1]),(0,0,255),3)
        crop = img[A[1]:D[1],A[0]:D[0]]
        cv2.imshow("cropped",crop)
        
        





import numpy as np
import math
import cv2



img =cv2.imread("rubiks-6.png")
imgGRAY=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(imgGRAY, 100, 255)
img2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
for cnt in contours:
    area=cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
    y=approx_is_square(approx, SIDE_VS_SIDE_THRESHOLD=0.60, ANGLE_THRESHOLD=20, ROTATE_THRESHOLD=30)
    if(y==True):
        square_width_height(approx)
        cv2.imshow("image",img)
        
cv2.waitKey(0)
cv2.destroyAllWindows()
    
        #if(width>140 and height>140):
            #print(x,"sggws")
    

        
