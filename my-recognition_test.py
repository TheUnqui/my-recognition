import jetson.inference #sehr schnelle performance) -> image recognition/objctdetection,segmentation
import jetson.utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
args = parser.parse_args()

#load image
img = jetson.utils.loadImage(args.filename) #läd image direkt in GPU memory

#load network
net = jetson.inference.imageNet(args.network)

#classify the image 
# -> gibt classindex der vom Netzwerk geglaubt wird zu haben
# -> gibt einen confidence Index zwischen 0 und 100 aus 
class_idx, confidence = net.Classify(img)

#get class name
class_name = net.GetClassDesc(class_idx)

#print out the results
print('classified image as {:s} (class ID {:d}, confidence {:f})'.format(class_name,class_idx, confidence))
 
