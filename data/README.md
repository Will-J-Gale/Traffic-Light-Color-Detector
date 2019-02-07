# Traffic-Light_Color_Detector
![alt text](https://github.com/Will-J-Gale/GTA-Fully-Convolutional-Lane-Finding/blob/master/Images/3.%20LaneFinding.gif)
![alt text](https://github.com/Will-J-Gale/GTA-Fully-Convolutional-Lane-Finding/blob/master/Images/3.%20LaneFinding.gif)  
![alt text](https://github.com/Will-J-Gale/GTA-Fully-Convolutional-Lane-Finding/blob/master/Images/3.%20LaneFinding.gif)  

## Description
This project utilizes the Yolo Keres from https://github.com/qqwweee/keras-yolo

   1. Objects are first detected using the Yolo Model
   2. If any traffic lights are detected, a crop of the traffic light in our lane is taken and resized to 60x120 (HxW)
   3. This cropped image is fed into another convolutional network which classifies the color of the traffic light

## Prerequisites 
1. GTA 5 + Mods (Not all mods are 100% necessary)
   * Script Hook V
   * Native trainer
   * Enhanced native trainer
   * GTA V FoV v1.35
   * Extended Camera Settings
   * Hood Camera 
2. Python 3.6
3. OpenCV
4. Numpy
5. Tensorflow GPU
6. Keras

## Usage
It is recommended to use dual monitors
1. Run GTA5 in windowed mode 1280x720
2. Find a car and enable Hood Camera
3. Run Traffic Light Detector.py

## Potential improvments
Currently this project uses 2 separate convolutional networks.  
However, creating a fully end-to-end network that can detect the traffic light color would improve the speed 