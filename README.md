# crowd_recog

This software is intended to do crowd recognition and track number of people that are paying attention to some specific point, in this case it would be an ad board on the street.
At this stage I have only some test scripts to chek the performance on pc and raspberry, also have test_cam to wathc some real time predictions.

The main hardware is  RaspPi3 and camera + Movidius Neural Stick and it shoud work in real time.


# in order to run this code on a new pc please run install_dep.sh and install NCSdk api



TO-DO list:
SOFTWARE:
1.1 make a parser for test_cam_benchmark.py script to pass a "mode" through command line - DONE
1.2 test loading and unloading the model and time required for the process
2. Train model
	2.1 may be smartphones or any other object
	2.2 Learn how to train a good model
	2.3 Learn how to convert model to a compatible graph for NCS
3. Start to collect apropriate data for training 
4. start writing the state of the art
5. Write a document
 

HARDWARE:
1. set up 3A power converter (DC12V3A to DCV3A)
2. make a box
