from mvnc import mvncapi as mvnc
import time


# grab a list of all NCS devices plugged in to USB
print("[INFO] finding NCS devices...")
devices = mvnc.EnumerateDevices()

# if no devices found, exit the script
if len(devices) == 0:
	print("[INFO] No devices found. Please plug in a NCS")
	quit()

# use the first device since this is a simple test script
# (you'll want to modify this is using multiple NCS devices)
print("[INFO] found {} devices. device0 will be used. "
	"opening device0...".format(len(devices)))
device = mvnc.Device(devices[0])
device.OpenDevice()

# open the CNN graph file
print("[INFO] loading the graph file into RPi memory...")
with open("graphs/mobilenetgraph", mode="rb") as f:
	graph_in_memory = f.read()

# load the graph into the NCS
while True:
	start = time.clock()
	graph = device.AllocateGraph(graph_in_memory)
	time_consumed = (time.clock() - start)
	print("Allocating Graph: {:0.3f}".format(time_consumed))

	#start = time.clock()
	#graph.DeallocateGraph()
	#time_consumed = (time.clock() - start)
	#print("Deallocating Graph: {:0.3f}".format(time_consumed))
	