import picamera     # Importing the library for camera module
import time   # Importing sleep from time library to add delay in program
import datetime
now = datetime.datetime.now()


camera = picamera.PiCamera()    # Setting up the camera

camera.resolution = (640, 480)
camera.framerate = 10




camera.start_recording('/home/pi/Desktop/10min_{}:{}_.h264'.format(now.hour,now.minute)) # Video will be saved at desktop
time.sleep(600)
camera.stop_recording()
camera.stop_preview()


print("MP4Box -fps 10 -add vid.h264 vid.mp4")