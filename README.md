# crowd_stats

This software is intended to do crowd detection and track number of people that are in fornt of camera and return teir age, gender, emotional state and if they are paying attention to the advertising or not.

At this stage the system detects people.

The main hardware is  RaspPi3 and camera + Movidius Neural Stick and it shoud work in real time.

# In order to run, install ncsdk_v2 api on your machine and run test_api_v2.py with python3

(!) throw some videos to videos folder in order to run another tests.


TO-DO list:
SOFTWARE:
1. add more layres of detection, i.e. more graphs to predict gender, age...
2. track people, give them id's
3. start writing the state of the art
4. Write a document
 

HARDWARE:
1. set up 3A power converter (DC12V3A to DCV3A)
2. make a box
