import sys
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel

from app.utils import loadRosmapData

if __name__ == "__main__":

    if(len(sys.argv) < 2):
        print("Usage: python trainModel <output:modelDir>")
        sys.exit(0) 

    print("Training SVD Model.")

    sc = SparkContext(appName="AlzheimersDiseaseClassifier")

    # Load data into the spark context
    print("Loading Data from ROSMAP file..")
    fileName = "../data/ROSMAP_RNASeq_entrez.csv"
    data = loadRosmapData(sc,fileName)
    print("Loading data complete!")
    
    # Build the model
    print("Training started...")
    model = SVMWithSGD.train(data, iterations=100)
    print("Training completed!")
    
    # Save and load the model
    print("Saving model...")
    outputLocation = "./models/"
    outputModel = sys.argv[1] 
    outputFile = outputLocation+outputModel
    model.save(sc, outputFile)
    print("Model saved.")
