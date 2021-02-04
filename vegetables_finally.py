import math
import logging
import rospy
from clover import srv
from std_srvs.srv import Trigger
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from enum import Enum
from clover.srv import SetLEDEffect
from mavros_msgs.srv import SetMode
from datetime import datetime


set_mode = rospy.ServiceProxy('mavros/set_mode', SetMode)

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)    


setup_logger('log1', "logs.txt")
logger_1 = logging.getLogger('log1')
image_n = 0
def take_photo(cv_image, name = 'photo', add_counter = 1):
    global image_n
    image_n +=add_counter
    try:
        cv2.imwrite(str(image_n)+'_'+name+'.jpg', cv_image)
    except:
        return

rospy.init_node('task_2')
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
land = rospy.ServiceProxy('land', Trigger)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)

bridge = CvBridge()

def navigate_wait(x=0, y=0, z=0, yaw=float('nan'), yaw_rate=0, speed=0.4, frame_id='body', tolerance=0.15, auto_arm=False):
    if(frame_id == 'aruco_map'):
        yaw = 3.14/2.0
    res = navigate(x=x, y=y, z=z, yaw=yaw, yaw_rate=yaw_rate, speed=speed, frame_id=frame_id, auto_arm=auto_arm)
    if not res.success:
        return res
    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            rospy.sleep(0.5)
            return res
        rospy.sleep(0.2)
	

class Color(Enum):
	YELLOW = (0,'yellow','products')
	GREEN = (1,'green','clothes')
	BLUE = (2,'blue','fragile packaging')
	RED = (3,'red','correspondence')
	

def detectColor():
	# Reading the video from the 
	# webcam in image frames 240*320
	imageFrame = bridge.imgmsg_to_cv2(rospy.wait_for_message('main_camera/image_raw', Image), 'bgr8')
	take_photo(imageFrame,'full',1)
	imageFrame = imageFrame[95:145, 135:185]
	take_photo(imageFrame,'crop',0)
	minarea = 150
	# Convert the imageFrame in 
	# BGR(RGB color space) to 
	# HSV(hue-saturation-value) 
	# color space 
	hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 

	# Set range for red color and 
	# define mask 
	red_lower_h = np.array([170, 87, 111], np.uint8) 
	red_upper_h = np.array([180, 255, 255], np.uint8) 
	red_mask_h = cv2.inRange(hsvFrame, red_lower_h, red_upper_h)

	# Set range for red color and 
	# define mask 
	red_lower_l = np.array([0, 87, 111], np.uint8) 
	red_upper_l = np.array([10, 255, 255], np.uint8) 
	red_mask_l = cv2.inRange(hsvFrame, red_lower_l, red_upper_l)

	# Set range for green color and 
	# define mask 
	green_lower = np.array([45, 60, 103], np.uint8) 
	green_upper = np.array([75, 255, 200], np.uint8) 
	green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

	# Set range for blue color and 
	# define mask 
	blue_lower = np.array([94, 80, 72], np.uint8) 
	blue_upper = np.array([130, 255, 255], np.uint8) 
	blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
	
	# Set range for yellow color and 
	# define mask 
	yellow_lower = np.array([26, 87, 111], np.uint8) 
	yellow_upper = np.array([34, 255, 255], np.uint8) 
	yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

	# Morphological Transform, Dilation 
	# for each color and bitwise_and operator 
	# between imageFrame and mask determines 
	# to detect only that particular color 
	kernal = np.ones((5, 5), "uint8") 
	
	# For red color_l 
	red_mask_l = cv2.dilate(red_mask_l, kernal) 
	res_red_l = cv2.bitwise_and(imageFrame, imageFrame, 
							mask = red_mask_l) 
	
	# For red color_h
	red_mask_h = cv2.dilate(red_mask_h, kernal) 
	res_red_h = cv2.bitwise_and(imageFrame, imageFrame, 
							mask = red_mask_h)

	# For green color 
	green_mask = cv2.dilate(green_mask, kernal) 
	res_green = cv2.bitwise_and(imageFrame, imageFrame, 
								mask = green_mask) 
	
	# For blue color 
	blue_mask = cv2.dilate(blue_mask, kernal) 
	res_blue = cv2.bitwise_and(imageFrame, imageFrame, 
							mask = blue_mask) 

	# For yellow color 
	yellow_mask = cv2.dilate(yellow_mask, kernal) 
	res_yellow = cv2.bitwise_and(imageFrame, imageFrame, 
							mask = yellow_mask) 

	# Creating contour to track red color 
	image, contours, hierarchy = cv2.findContours(red_mask_l, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
	red_sum = 0

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour) 
		if(area > minarea):
			red_sum+=area	 

	# Creating contour to track red color 
	image, contours, hierarchy = cv2.findContours(red_mask_h, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour) 
		if(area > minarea):
			red_sum+=area	 	

	# Creating contour to track green color 
	image, contours, hierarchy = cv2.findContours(green_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
	green_sum = 0

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour) 
		if(area > minarea):
			green_sum+=area	 

	# Creating contour to track blue color 
	image, contours, hierarchy = cv2.findContours(blue_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
	blue_sum = 0

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour) 
		if(area > minarea):
			blue_sum+=area	 

	# Creating contour to track yellow color 
	image, contours, hierarchy = cv2.findContours(yellow_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
	yellow_sum = 0

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour) 
		if(area > minarea):
			yellow_sum+=area

	dic = {Color.RED: red_sum,
		 Color.GREEN: green_sum,
		Color.YELLOW: yellow_sum,
		  Color.BLUE: blue_sum
	}

	key_max = max(dic, key=dic.get)
	if(dic[key_max] > 0):
		return key_max
	return 



def blink_color( r=0, g=0, b=0,time=5,effect='fill'):
	set_effect(effect=effect, r=r, g=g, b=b)
	rospy.sleep(time)
	set_effect(effect=effect, r=0, g=0, b=0)

img_0_x = cv2.imread("0_s.jpg")
img_1_x = cv2.imread("1_s.jpg")
img_2_x = cv2.imread("2_s.jpg")
img_3_x = cv2.imread("3_s.jpg")

def compare_xor_image(img):
    try:
        if (img is None):
            return None
        dim = (60, 90)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite("temp_file.jpg",img)
        img_in = cv2.imread("temp_file.jpg")
        mask = cv2.imread("mask.jpg")
        images = (img_0_x,img_1_x,img_2_x,img_3_x)
        min_mask = 99999
        num = 0
        for i in range(len(images)):
            res = cv2.bitwise_xor(images[i],img_in,mask)
            m_count =  np.sum(res == 255)
            if(m_count<min_mask):
                min_mask = m_count
                take_photo(res,"w_b_mask")
                num =i
                print(i,m_count)
        return num
    except: 
        return None


def compare_image(img1, img2):
    try:
        if (img2 is None):
            return None
        #logger_1.info('start comp')
        # Initiate SIFT detector
        FLANN_INDEX_KDTREE = 0
        kaze = cv2.KAZE_create(extended = True, threshold = 0.0005, nOctaves = 7, nOctaveLayers = 7)


        # find the keypoints and descriptors with SIFT
        kp1, des1 = kaze.detectAndCompute(img1, None)
        kp2, des2 = kaze.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        #logger_1.info('end comp')
        return len(good)
    except: 
        return None

def crop_photo(img2):
     # first_hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    lower_green = np.array([45, 98, 80], np.uint8) 
    upper_green = np.array([79, 255, 172], np.uint8)

    # first_mask = cv2.inRange(first_hsv, lower_green, upper_green)
    # first_res = cv2.bitwise_and(first, first, mask = first_mask)

    img2_mask = cv2.inRange(img2_hsv, lower_green, upper_green) #white_mask
    img2_res = cv2.bitwise_and(img2, img2, mask = img2_mask) #colored

    green_sum = 0
    kernal = np.ones((5, 5), "uint8") 

    green_mask = cv2.dilate(img2_mask, kernal)
    res_green = cv2.bitwise_and(img2, img2, mask = green_mask)

    contours = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    contour_id = 0
    mask = np.zeros_like(img2_mask)
    out = np.zeros_like(img2_mask)
    out[mask == 255] = img2_mask[mask == 255]
    x, y, w, h = 0,0,0,0
    max_area = 0
    min_area = 400

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)   
        if(area > max_area and area > min_area):
            x, y, w, h = cv2.boundingRect(contour) 
            # out = cv2.rectangle(img2, (x, y),  (x + w, y + h), (0, 255, 0), 2) 
            max_area = area
    if (max_area<min_area):
        return None,None
    
    img_mask = green_mask[y:y+h, x:x+w]#white
    print("max_area",max_area)
    try:
        offset_h = int(h*1.1)-h
        offset_w = int(w*1.1)-w
        y_d = y-offset_h
        y_h = y+offset_h+h
        x_d = x-offset_w
        x_h = x+w+offset_w
        if(y_d>0 and y_h<240 and x_d>0 and x_h<280):
            img2 = img2[y_d:y_h, x_d:x_h] #greenmult = 2
        mult = 2
        dim = (w*mult,h*mult)
        img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
        img2 = cv2.fastNlMeansDenoisingColored(img2, h = 5, hColor = 5, templateWindowSize = 7, searchWindowSize = 21 )

        # sharpen_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        # sharpen_kernel = np.array([[1.0, 4.0, 6.0, 4.0, 1.0], [4.0, 16.0, 24.0, 16.0, 4.0], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])/(256)
        sharpen_kernel = np.array([[1.0, 1, 1], [1.0, 1, 1], [1.0, 1, 1]])/9
        img2 = cv2.filter2D(img2, -1, sharpen_kernel)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img2 = cv2.filter2D(img2, -1, sharpen_kernel)

    except:
        return None, img_mask
    return img2,img_mask    

query_img0 = cv2.imread('00.jpg')
query_img1 = cv2.imread('11.jpg')
query_img2 = cv2.imread('22.jpg')
query_img3 = cv2.imread('33.jpg')

def detect_number(take_first_num = False ):
    train_img = bridge.imgmsg_to_cv2(rospy.wait_for_message('main_camera/image_raw', Image), 'bgr8')
    take_photo(train_img,'full_point',1)
    train_img = train_img[0:240, 20:300]
    take_photo(train_img,'crop_point',0)
    query_img = [query_img0, query_img1, query_img2, query_img3]
    train_img, black_img = crop_photo(train_img)
    if (train_img is None and black_img is not None):
        if(take_first_num):
            number_one = compare_xor_image(black_img)
            logger_1.info("So, error again. I use first number")
            return number_one
        else:
            return 9 ## 'need to check' code
    number_one = compare_xor_image(black_img)
    logger_1.info("Hmmmm.... I mean it's number {}".format(number_one))
    max = 0
    ind = -1
    if (train_img is None):
        return None
    for i in range(4):
        a = compare_image(query_img[i], train_img)
        print(a)
        if(max<a):
            max = a
            ind = i
    if (ind ==-1):
        return None
    return ind

def fly_storage(min_x = 0, min_y = 0, max_x=0,max_y=0,h = 1.1):
    color_counter = [0,0,0,0]
    plus_flag = True
    for y in range (int(round(min_y*2)),int(round(max_y*2-1))):
        for x in range (int(round(min_x*2)),int(round(max_x*2-1))):
            if (plus_flag):
                navigate_wait(x=x*0.45, y=y*0.45, speed=0.25, z=h, frame_id='aruco_map')
            else:
                navigate_wait(x=(max_x*2-2-x)*0.45, y=y*0.45, speed=0.25, z=h, frame_id='aruco_map')
            rospy.sleep(0.3)
            colour = detectColor()
            if(colour):
                logger_1.info(colour)
                color_counter[colour.value[0]] +=1
        plus_flag = not plus_flag
    return color_counter

def fly_check_points(min_x = 0, min_y = 0, max_x=0,max_y=0,h = 1.5):
    n=0
    points_counter = [None,None,None,None,None,None,None,None,None,None,None,None]
    plus_flag = True
    for y in range (int(round(min_y)),int(round(max_y))):
        for x in range (int(round(min_x)),int(round(max_x))):
            x_o,y_o = 0,0
            if (plus_flag):
                x_o=x*0.9
                y_o=y*0.9
            else:
                x_o=(max_x-1-x)*0.9
                y_o=y*0.9
            navigate_wait(x=x_o, y=y_o, z=h, frame_id='aruco_map')
            rospy.sleep(0.2)
            number = detect_number()
            if (number == 9): #need to repeat
                logger_1.info("Detection error. Need to try again...")
                number = detect_number(take_first_num=True)
            logger_1.info("So, it's number {}".format(number))
            if(number != None):
                points_counter[n] = (number,(x_o,y_o))
                n+=1
        plus_flag = not plus_flag
    return points_counter

def print_items_info(items_array = [0,0,0,0]):
    sum_p = 0
    for number in items_array:
        sum_p +=number
    logger_1.info("Balance {} cargo".format(sum_p))
    for n in range(0,4):
        logger_1.info("Type {}: {} cargo".format(n,items_array[n]))

def go_points(points):
    n=0
    while(points[n] is not None):
        x = points[n][1][0]
        y = points[n][1][1]
        color = points[n][0]
        navigate_wait(x=x, y=y, z=2.05, frame_id='aruco_map',tolerance=0.1,speed=0.25)
        rospy.sleep(6)
        land()
        rospy.sleep(2)
        if(color == 0):
            logger_1.info('D{}_delivered products'.format(color))
            blink_color(g = 255,r=255,time = 5)
        if(color == 1):
            logger_1.info('D{}_delivered clothes'.format(color))
            blink_color(g = 255,time = 5)
        if(color == 2):
            logger_1.info('D{}_delivered fragile packaging'.format(color))
            blink_color(b=255,time = 5)
        if(color == 3):
            logger_1.info('D{}_delivered correspondence'.format(color))
            blink_color(r=255,time = 5)
        navigate_wait(z=1, frame_id='body', auto_arm=True)
        n+=1

def print_delivered(points,items_count_array):
    n=0
    sum_p=0
    for number in items_count_array:
        sum_p +=number
    sum_n = 0
    while(points[n] is not None):
        color = points[n][0]
        count = items_count_array[color]
        sum_n +=count
        items_count_array[color] = 0
        logger_1.info('D{}_delivered to {} cargo'.format(color,count))
        n+=1
    balance = sum_p-sum_n
    logger_1.info('Balance: {} cargo.'.format(balance))

def main():
    navigate_wait(z=2, frame_id='body', auto_arm=True)
    navigate_wait(x = 0 , y = 4.5*0.9, z=2, frame_id='aruco_map',speed=0.4)
    items_count_array = fly_storage(0, 4.5, 5, 7)
    # items_count_array = [0, 1, 7, 3] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print_items_info(items_count_array)
    navigate_wait(z=1, frame_id='body')
    navigate_wait(x = 0 , y = 0, z=2, frame_id='aruco_map',speed=0.4)
    rospy.sleep(2)
    points = fly_check_points(0, 0, 5, 5, 2.1)
    # while(True):
        # points = fly_check_points(0, 0, 5, 5, 1.7)#   to remove
    # points = [(0,(1,2)),(2,(2,4)),None]#   to remove
    print(points)
    go_points(points)
    navigate_wait(x=0, y=0, z=2, frame_id='aruco_map')
    navigate_wait(x=0, y=0, z=1, frame_id='aruco_map')
    rospy.sleep(6)
    land()
    print_delivered(points, items_count_array )

if __name__ == "__main__":
    try:
        logger_1.info("Ah dronpoint... Here we go again")
        main()
    except Exception, e :
        logger_1.error('Error: landing...\n'+str(e))
        land()
    finally:
        set_effect(r=255, g=255, b=255, effect='fill')
        print("btw I use arch linux")
