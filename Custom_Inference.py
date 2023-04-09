import jetson.inference #sehr schnelle performance) -> image recognition/objctdetection,segmentation
import jetson.utils

import argparse



parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--network", type=str)
parser.add_argument("networkType", type=str)
args = parser.parse_args()

#load image
img = jetson.utils.loadImage(args.filename) #läd image direkt in GPU memory

type = args.networkType
#load network
if (type=="imageNet") : 
        net = jetson.inference.imageNet(args.network)

        #classify the image 
        # -> gibt classindex der vom Netzwerk geglaubt wird zu haben
        # -> gibt einen confidence Index zwischen 0 und 100 aus 
        class_idx, confidence = net.Classify(img)

        #get class name (methoden)
        class_name = net.GetClassDesc(class_idx)    

        #print out the results
        print('classified image as {:s} (class ID {:d}, confidence {:f})'.format(class_name,class_idx, confidence))



if (type=="detectNet") : 
        net = jetson.inference.detectNet(args.network)
if (type=="segNet") : 
        net = jetson.inference.segNet(args.network)
if (type=="poseNet") : 
        net = jetson.inference.poseNet(args.network)
if (type=="depthNet") : 
        net = jetson.inference.depthNet(args.network)
if (type=="backgroundNet") : 
        net = jetson.inference.backgroundNet(args.network)
if (type=="actionNet") : 
        net = jetson.inference.actionNet(args.network)





