from pyspark import SparkContext

from pyspark.mllib.classification import SVMWithSGD, SVMModel

from .utils import loadRosmapData

if __name__ == "__main__":
    print("Executing main.")

    sc = SparkContext(appName="AlzheimersDiseaseClassifier")

    print("Loading Data from ROSMAP file..")
    # Load data into the spark context
    fileName = "../data/ROSMAP_RNASeq_entrez.csv"
    data = loadRosmapData(sc,fileName)
    print("Loading data complete!")
    
    # Build the model
    model = SVMWithSGD.train(data, iterations=100)

    # Evaluating the model on training data
    

    # Save and load the model
    outputLocation = "./models/"
    outputModel = "svmADClassifier.model"
    outputFile = outputLocation+outputModel
    model.save(sc, outputFile)

    sampleModel = SVMModel.load(sc, outputFile)
