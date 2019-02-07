'''
Yolo Keras Reference: https://github.com/qqwweee/keras-yolo3
grabscreen by Frannecklp

Traffic Light Detection Author: Will-J-Gale
'''

from TLD_Model import TLDModel
from keras.models import load_model
from Yolo import yolo_out, get_classes, draw, preprocess
import os, cv2
import numpy as np
from random import shuffle
from grabscreen import grab_screen

def extractLaneTrafficLight(image, boxes, classes, scores):
    
    trafficLightIndexes = np.where(classes == TRAFFIC_LIGHT)[0] #Extract indexes of traffic light class
    laneTrafficLight = None
    center = image.shape[1] // 2
    closest = 1000000

    #Get the traffic light closest to center
    for index in trafficLightIndexes:
        xPos = boxes[index][0]
        distanceToCenter = abs(xPos - center)
        if(distanceToCenter < closest):
            closest = distanceToCenter
            laneTrafficLight = boxes[index]

    x, y, w, h = laneTrafficLight

    #Stops values going below 0 (occasionaly values are -1 which causes errors)
    x = max(int(x), 0)
    y = max(int(y), 0)
    w = max(int(w), 0)
    h = max(int(h), 0)

    #Resize for TLDModel
    trafficLightImage = cv2.resize(image[y:y+h, x:x+w, :], (TLD_MODEL_SIZE[1], TLD_MODEL_SIZE[0]))

    return trafficLightImage, (x, y, w, h)

def processYolo(image):
    global yoloModel
    yoloInput = preprocess(screen)
    prediction = yoloModel.predict(yoloInput)
    boxes, classes, scores = yolo_out(prediction, screen.shape)

    return boxes, classes, scores

def addBorderAndResize(image):
    global PADDING
    image = cv2.copyMakeBorder(image,0,0,0,PADDING,cv2.BORDER_CONSTANT,value=[0,0,0])
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)

    return image
    
def drawTrafficLights(image, lightClass):
    global RED, YELLOW, GREEN, NONE, PADDING
    drawnImage = None

    #Select which traffic light image to use
    if(lightClass == 0):
        drawnImage = RED
    elif(lightClass == 1):
        drawnImage = YELLOW
    elif(lightClass == 2):
        drawnImage = GREEN
    else:
        drawnImage = NONE

    #Place image in center of padding section
    xStart = (image.shape[1] - (PADDING // 2)) + (drawnImage.shape[1] // 3)
    xEnd = xStart + drawnImage.shape[1]

    yStart = (image.shape[0] // 2) - (drawnImage.shape[0] // 2)
    yEnd = yStart + drawnImage.shape[0]

    #Overlay the traffic light image on the screen
    image[yStart:yEnd, xStart:xEnd, :] = drawnImage

def highlightSeenTrafficLight(image, position):
    #Put rectangle around the traffic light the algorithm is focusing on
    x, y, w, h = position
    cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 3)

#Traffic Light Images
RED = cv2.imread("Images/RED.png")
YELLOW = cv2.imread("Images/YELLOW.png")
GREEN = cv2.imread("Images/GREEN.png")
NONE = cv2.imread("Images/NONE.png")

#Screen capture parameters
width = 1920 - 1
height = 1080 
gameWidth = 1274 
gameHeight = 714

xOffset = 2
yOffset = 17
gameStartX = ((width//2) - (gameWidth//2)) + xOffset
gameStartY = ((height//2) - (gameHeight//2)) + yOffset
gameEndX = ((width//2) + (gameWidth//2)) + xOffset
gameEndY = ((height//2) + (gameHeight//2)) + yOffset

SCREEN_REGION = (gameStartX, gameStartY, gameEndX, gameEndY)
SCREEN_NAME = "Traffic Light Detection"

PADDING = 200

#Yolo Model Parameters
classesFileName = 'data/coco_classes.txt'
all_classes = get_classes(classesFileName)
yoloModel = load_model('data/yolo.h5')

#TLD Models Parameters
TLD_MODEL_SIZE = (120, 60, 3)
MODEL_NAME = "TLD.model"
TRAFFIC_LIGHT = 9
lightClasses = ["Red", "Yellow", "Green"]

#Model parameters
tldModel = TLDModel()
tldModel.load_weights(MODEL_NAME)

lightClass = None
index = 2000

while(True):
    #Capture screen and convert to RGB
    screen = grab_screen(SCREEN_REGION)
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    #Detect objects using YOLO
    boxes, classes, scores = processYolo(screen)

    #Classify traffic light colour (if there are any)
    if(classes is not None and TRAFFIC_LIGHT in classes):
        tlImage, tlPosition = extractLaneTrafficLight(screen, boxes, classes, scores)
        prediction = tldModel.predict(np.expand_dims(tlImage, axis=0))[0]
        lightClass = np.argmax(prediction)
        highlightSeenTrafficLight(screen, tlPosition)   
    else:
        lightClass = None

      #Uncomment to show ALL boxes on screen 
##    if(boxes is not None):
##        draw(screen, boxes, scores, classes, all_classes)

    #Add black border, resize the image and then draw the traffic light image    
    screen = addBorderAndResize(screen)
    drawTrafficLights(screen, lightClass)
    cv2.imwrite(f"GifImages/{index:05d}.png", screen)
    index += 1
    #Show the result on the screen
    cv2.imshow(SCREEN_NAME, screen)    
    cv2.waitKey(1)


















