#!/usr/bin/env python
# -*- coding: utf-8 -*-

from picamera.array import PiRGBArray
from picamera import PiCamera
from optparse import OptionParser
import time
import cv2
import cv
import numpy as np
import rospy
from obstacle_detector.msg import *

def switch_to_octant_zero(theta, p):
    step = 6.28/8.0

    if theta < step and theta >= 0.0:
        return p
    if theta < step*2.0 and theta >= step:
        return (p[1],p[0])
    if theta < step*3.0 and theta >= step*2.0:
        return (p[1],-p[0])
    if theta < step*4.0 and theta >= step*3.0:
        return (-p[0],p[1])
    if theta < step*5.0 and theta >= step*4.0:
        return (-p[0],-p[1])
    if theta < step*6.0 and theta >= step*5.0:
        return (-p[1],-p[0])
    if theta < step*7.0 and theta >= step*6.0:
        return (-p[1],p[0])
    if theta < step*8.0 and theta >= step*7.0:
        return (p[0],-p[1])

def switch_from_octant_zero(theta, p):
    step = 6.28/8.0

    if theta < step and theta >= 0.0:
        return p
    if theta < step*2.0 and theta >= step:
        return (p[1],p[0])
    if theta < step*3.0 and theta >= step*2.0:
        return (-p[1],p[0])
    if theta < step*4.0 and theta >= step*3.0:
        return (-p[0],p[1])
    if theta < step*5.0 and theta >= step*4.0:
        return (-p[0],-p[1])
    if theta < step*6.0 and theta >= step*5.0:
        return (-p[1],-p[0])
    if theta < step*7.0 and theta >= step*6.0:
        return (p[1],-p[0])
    if theta < step*8.0 and theta >= step*7.0:
        return (p[0],-p[1])


def detect_collision_in_ray(image, theta, p1, p2):
    p1c = switch_to_octant_zero(theta, p1)
    p2c = switch_to_octant_zero(theta, p2)
    dx = (p2c[0] - p1c[0])
    dy = (p2c[1] - p1c[1])
    D = dy - dx

    line_pos = []
    line_col = []
    y = p1c[1]
    for x in range(p1c[0], p2c[0]-1):
        line_pos.append(switch_from_octant_zero(theta, (x,y)))
        line_col.append(image[line_pos[-1][1]][line_pos[-1][0]])

        if D >= 0:
            y += 1
            D -= dx
        D += dy

    filter = [-1,-1,-1,-1,0,1,1,1,1]
    line_grad =  np.convolve(line_col, filter, 'same')
    for idx, val in enumerate(line_grad):
        if theta > 3.49:
            if val > 100 and idx > 60 and idx < line_grad.size-10:
                #cv2.circle(image, line_pos[idx], 6, (255,0,0), 1)
                return line_pos[idx]
        else:
            if val > 200 and idx > 5 and idx < line_grad.size-10:
                #cv2.circle(image, line_pos[idx], 6, (255,0,0), 1)
                return line_pos[idx]

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--debug", dest="debug_on", help="enable debug output", default=False, action='store_true')
    (options, args) = parser.parse_args()

    rospy.init_node('obstacle_detector', anonymous=False)
    pub = rospy.Publisher('obstacle', ObstacleLocation, queue_size=10)

    with PiCamera() as camera:
        camera.resolution = (1024,768)
        camera.ISO = 100
        camera.sa = 100
        camera.awb = "flash"
        camera.co = 100


        raw_capture = PiRGBArray(camera)

        time.sleep(0.1)

        camera.capture(raw_capture, format="bgr")
        image = raw_capture.array
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(image_gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 200, param1=50, param2=40, minRadius=10, maxRadius=230)
        raw_capture.truncate(0)

        if circles is not None:
            x, y, r = circles[0][0]

            p1x = int(x + r*1.2)
            p1y = int(y)
            p2x = int(x + r*2.25)
            p2y = int(y)

            lines = []
            for theta in np.linspace(0.0, 6.28, 50, False):
                p1xr = int( np.cos(theta) * (p1x - x) - np.sin(theta) * (p1y - y) + x )
                p1yr = int( np.sin(theta) * (p1x - x) + np.cos(theta) * (p1y - y) + y )
                p2xr = int( np.cos(theta) * (p2x - x) - np.sin(theta) * (p2y - y) + x )
                p2yr = int( np.sin(theta) * (p2x - x) + np.cos(theta) * (p2y - y) + y )

                lines.append( (theta, p1xr, p1yr, p2xr, p2yr) )

            for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
                image = frame.array
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                if options.debug_on:
                    cv2.circle(image, (x,y), int(r*1.2), (0,255,0), 2)

                obstacles = []
                for line in lines:
                    collision_pos = detect_collision_in_ray(image_gray, line[0], (line[1],line[2]), (line[3],line[4]))

                    if collision_pos is not None:
                        obstacle = ObstacleLocation()
                        obstacle.centre = (x,y)
                        obstacle.theta = line[0]
                        obstacle.radius = np.sqrt( (collision_pos[0]-x)**2 + (collision_pos[1]-y)**2 )
                        obstacles.append(obstacle)

                        if options.debug_on:
                            cv2.circle(image, collision_pos, 6, (0,0,255), 1)

                msg = ObstacleArray()
                msg.obstacles = obstacles
                pub.publish(msg)

                if options.debug_on:
                    cv2.imshow("DEBUG", image)
                    key = cv2.waitKey(1) & 0xFF

                raw_capture.truncate(0)

                if key == ord("q"):
                    break
