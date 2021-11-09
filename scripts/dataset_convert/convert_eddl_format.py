# $ python convert_eddl_format.py target_folder test_train_or_ratio data_number
#
# target_folder: must give minimal folder path to convert data
# test_train_or_ratio: must define 'test' or 'train' about this data,
#                      if you want seperate total data to test and train automatically,
#                      you can input one integer for test ratio,
#                      e.q. if you input 2, it mean 2% data will become test data
# data_number: if you input 0 or nothing, it convert total images under each label folder,
#        e.q.
#          a. python convert_eddl_format.py notMNIST_small test 0
#          b. python convert_eddl_format.py notMNIST_small test
#          c. python convert_eddl_format.py notMNIST_small train 0
#          d. python convert_eddl_format.py notMNIST_small train
#

import numpy
import imageio
import glob
import sys
import os
import random
from PIL import Image

channels = 0
height = 0
width = 0
classes = 0

dstPath = "convert_EDDL"
testLabelPath = dstPath+"/test-labels.bin"
testImagePath = dstPath+"/test-images.bin"
trainLabelPath = dstPath+"/train-labels.bin"
trainImagePath = dstPath+"/train-images.bin"


def get_subdir(folder):
    listDir = None
    for root, dirs, files in os.walk(folder):
        if not dirs == []:
            listDir = dirs
            break
    listDir.sort()
    return listDir


def get_labels_and_files(folder, number=0):
    global classes
    # Make a list of lists of files for each label
    filelists = []
    subdir = get_subdir(folder)
    for label in range(0, len(subdir)):
        filelist = []
        filelists.append(filelist)
        dirname = os.path.join(folder, subdir[label])
        for file in os.listdir(dirname):
            if (file.endswith('.png')):
                fullname = os.path.join(dirname, file)
                if (os.path.getsize(fullname) > 0):
                    filelist.append(fullname)
                else:
                    print('file ' + fullname + ' is empty')
            if (file.endswith('.jpeg')):
                fullname = os.path.join(dirname, file)
                if (os.path.getsize(fullname) > 0):
                    filelist.append(fullname)
                else:
                    print('file ' + fullname + ' is empty')
        # sort each list of files so they start off in the same order
        # regardless of how the order the OS returns them in
        filelist.sort()

    # Take the specified number of items for each label and
    # build them into an array of (label, filename) pairs
    # Since we seeded the RNG, we should get the same sample each run
    labelsAndFiles = []
    classes = len(subdir)
    for label in range(0, len(subdir)):
        count = number if number > 0 else len(filelists[label])
        filelist = random.sample(filelists[label], count)  
        for filename in filelist:
            labelsAndFiles.append((label, filename))
	

    return labelsAndFiles


def make_arrays(labelsAndFiles, ratio):
    global height, width, channels
    
    images = []
    labels = []
    imShape = imageio.imread(labelsAndFiles[0][1]).shape
    if len(imShape) > 2:
        height, width, channels = imShape
    else:
        height, width = imShape
        channels = 1
    print("Shape: ", imShape) 
    print() 
    for i in range(0, len(labelsAndFiles)):
        # display progress, since this can take a while
        if (i % 100 == 0):
            sys.stdout.write("\r%d%% complete" %
                             ((i * 100) / len(labelsAndFiles)))
            sys.stdout.flush()

        filename = labelsAndFiles[i][1]
        try:
            image = imageio.imread(filename)
            newimShape = image.shape
            # Resize to the shape of 1st image, if required
            if (newimShape != imShape):
                print("New Shape: ", newimShape, "--> Resizing to ", imShape)   
                image = Image.fromarray(image).resize((width,height))
            images.append(image)
            labels.append(labelsAndFiles[i][0])
        except Exception as e:
            # If this happens we won't have the requested number
            print(e)
            print("Exception: " + filename)

    if ratio == 'train':
        ratio = 0
    elif ratio == 'test':
        ratio = 1
    else:
        ratio = float(ratio) / 100
    count = len(images)
    trainNum = int(count * (1 - ratio))
    testNum = count - trainNum
    if channels > 1:
        trainImagedata = numpy.zeros(
            (trainNum, height, width, channels), dtype=numpy.uint8)
        testImagedata = numpy.zeros(
            (testNum, height, width, channels), dtype=numpy.uint8)
    else:
        trainImagedata = numpy.zeros(
            (trainNum, height, width), dtype=numpy.uint8)
        testImagedata = numpy.zeros(
            (testNum, height, width), dtype=numpy.uint8)
    trainLabeldata = numpy.zeros(trainNum, dtype=numpy.uint8)
    testLabeldata = numpy.zeros(testNum, dtype=numpy.uint8)

    for i in range(trainNum):
        trainImagedata[i] = images[i]
        trainLabeldata[i] = labels[i]

    for i in range(0, testNum):
        testImagedata[i] = images[trainNum + i]
        testLabeldata[i] = labels[trainNum + i]
    print("\n")
    return trainImagedata, trainLabeldata, testImagedata, testLabeldata


def write_labeldata(labeldata, outputfile):
    global classes
#    header = numpy.array([0x0801, len(labeldata)], dtype='>i4')
    print("Labels (#,classes)",len(labeldata),classes)
    header = numpy.array([2,  len(labeldata), classes], dtype='<i4')
    #print(labeldata)
    one_hot = numpy.eye(classes)[labeldata]
    one_hot = one_hot.astype('float32');
    #print(one_hot)
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(one_hot.tobytes())

def write_imagedata(imagedata, outputfile):
    global height, width
    print("Images (#,channels,height,width,height*width)",len(imagedata),channels,height, width, height*width)
#    header = numpy.array([0x0803, len(imagedata), height, width], dtype='>i4')
#    header = numpy.array([2, len(imagedata), width*height],  dtype='<i4')
    header = numpy.array([4, len(imagedata), channels, height, width],  dtype='<i4')
    imagedata = imagedata.astype('float32')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(imagedata.tobytes())


def main(argv):
    global idxLabelPath, idxImagePath
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
    if len(argv) == 3:
        labelsAndFiles = get_labels_and_files(argv[1])
    elif len(argv) == 4:
        labelsAndFiles = get_labels_and_files(argv[1], int(argv[3]))
    random.shuffle(labelsAndFiles)

    trainImagedata, trainLabeldata, testImagedata, testLabeldata = make_arrays(
        labelsAndFiles, argv[2])

    if argv[2] == 'train':
        print("Writing training...")
        write_labeldata(trainLabeldata, trainLabelPath)
        write_imagedata(trainImagedata, trainImagePath)
    elif argv[2] == 'test':
        print("Writing test...")
        write_labeldata(testLabeldata, testLabelPath)
        write_imagedata(testImagedata, testImagePath)
    else:
        print("Writing training...")
        write_labeldata(trainLabeldata, trainLabelPath)
        write_imagedata(trainImagedata, trainImagePath)
        print("Writing test...")
        write_labeldata(testLabeldata, testLabelPath)
        write_imagedata(testImagedata, testImagePath)


if __name__ == '__main__':
    main(sys.argv)
