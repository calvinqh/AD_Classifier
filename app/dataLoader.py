from pyspark import SparkContext

from pyspark.mllib.regression import LabeledPoint

from app.featCluster import generateFeatureClusters

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
    headerLine = rawData.first()
    header = headerLine.split(',')
    header = [elem for elem in header if elem != 'PATIENT_ID']

    # Remove the header from file
    headerlessRawData = rawData.filter(lambda line: line != headerLine)
    
    # Parse the line to remove the commas
    allSamples = headerlessRawData.map(lambda line: line.split(','))

    # Apply filter to remove samples with diagnosis of NA, 2,3,6
    sampleSubset = allSamples.filter(lambda sample: rosmapFilter(sample, 'NA', '2', '3', '6'))

    # Transform training samples to trainable data format
    # trainingSamples will now have properly Labeled Points
    trainingSamples = sampleSubset.map(cleanSample)
    
    return trainingSamples

