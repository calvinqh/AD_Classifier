from pyspark import SparkContext

from pyspark.mllib.regression import LabeledPoint

def cleanSample(sample):
    return LabeledPoint(getLabelID(sample[1]), sample[2:]) 

def rosmapFilter(sample, *args):
    for arg in args:
        if sample[1] == arg:
            return None 
    return sample 

def getLabelID(label):
    if(label == '1'):
        return 0
    return 1

def loadRosmapData(context, fileName):
    # Read datafile into spark context
    rawData = context.textFile(fileName)
    
    # Retrieve header of file
    header = rawData.first()

    # Remove the header from file
    headerlessRawData = rawData.filter(lambda line: line != header)
    
    # Parse the line to remove the commas
    allSamples = headerlessRawData.map(lambda line: line.split(','))

    # Apply filter to remove samples with diagnosis of NA, 2,3,6
    trainingSamples = allSamples.filter(lambda sample: rosmapFilter(sample, 'NA', '2', '3', '6'))

    # Transform training samples to trainable data format
    # Currently they are all strings
    trainingSamples = trainingSamples.map(cleanSample)

    return trainingSamples
