import numpy as np

from pyspark import SparkContext

from pyspark.mllib.stat import Statistics

from pyspark.mllib.clustering import KMeans, KMeansModel


def removePatientIDandDiagnosis(sample):
    filteredSamples = []
    for item in sample[2:]:
        try:
            filteredSamples.append(float(item))
        except:
            filteredSamples.append(0)
    return filteredSamples

def loadRosmapClusterData(context, fileName):
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
    # Currently they are all strings
    trainingSamples = sampleSubset.map(cleanSample)
    
    geneExpressionsOnly = sampleSubset.map(removePatientIDandDiagnosis)

    # Create a map of <eid, [diag, value]>
    geneToDetail = sampleSubset.flatMap(lambda sample: geneToDetailMapper(sample,header))

    generateFeatureClusters(context,geneExpressionsOnly, trainingSamples, header, 10)

    return trainingSamples


def updateSample(sample, cF): 
    instance = [sample.label] #retrieve sample diagnosis
    for c,ids in cF.items():
        acc = 0
        # Loop through all expression values
        # Accumulate sum for the current cluster
        for i in range(len(sample.features)):
            if(i in ids):
                acc += sample.features[i] 
        # add cluster with their value
        instance.append(acc/len(ids))
    return instance

def generateFeatureClusters(context, geneExp, samples, headers, numClusters):

    # Ignore the first item (the diagnosis header)
    headers = headers[1:]
    # 1) Generate statistic data for each of the genes/entrez ids

    # Retrieve the mean, variance, max and min of each gene
    # The entrez id associate with each gene is the row index (matches to the headers index)
    cStats = Statistics.colStats(geneExp)
    print(len(cStats.mean()))
    data = np.array([cStats.mean(),cStats.variance(),cStats.max(), cStats.min()]).transpose()
    # Create a stats array with the index as first column
    # e_id for e_id in headers
    dataWithIndex = np.array([[e_id for e_id in headers],cStats.mean(),cStats.variance(),cStats.max(), cStats.min()]).transpose()
    print(dataWithIndex.shape) 
    # 2) Create dataframes that will be used to train KMeans

    # Create dataframe for the stats data (with no entrez ids)
    df = context.parallelize(data)
    # create dataframe for the stats data (with entrez ids)
    # Will be used to cluster features later
    dfWithIndex = context.parallelize(dataWithIndex)

    # 3) Train KMeans with statistic data 
    # use the stats data to discover clusters for the genes
    model = KMeans.train(df, numClusters, maxIterations=100, initializationMode="random")

    # 4) save model
    model.save(context, './models/clusters')
  
    # 5) Label each feature with their cluster
    # For each gene statistic, map it to (prediction, e_id)
    clusterLabeledFeatures = dfWithIndex.map(lambda point: (model.predict(point[1:]),point[0]))

    featuresToCluster = dfWithIndex.map(lambda point: point[0],(model.predict(point[1:])))

    # 6) Group together the features by their cluster label
    clusteredFeatures = clusterLabeledFeatures.groupByKey()
    #print(clusteredFeatures.count())
    #print(clusteredFeatures.take(2))

    cF = clusteredFeatures.collectAsMap()
    
    # 7) Transform the sample data to use the clusters
    samplesWithClusters = samples.map(lambda sample: updateSample(sample, cF)) 

    return samplesWithClusters 
