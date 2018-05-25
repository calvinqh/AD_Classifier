import sys
from pyspark import SparkContext
from pyspark.mllib.classification import SVMModel

from app.utils import loadRosmapData

if __name__ == "__main__":

    if(len(sys.argv) < 2):
        print("Usage: python evalModel <input:modelDir> ")
        sys.exit(0) 

    print("Evaluating", sys.argv[1])

    sc = SparkContext(appName="AlzheimersDiseaseClassifier")

    # Load data into the spark context
    print("Loading Data from ROSMAP file..")
    fileName = "../data/ROSMAP_RNASeq_entrez.csv"
    data = loadRosmapData(sc,fileName)
    print("Loading data complete!")

    # Load the model
    print("Loading model...")
    outputLocation = "./models/"
    outputModel = sys.argv[1] 
    outputFile = outputLocation+outputModel

    model = SVMModel.load(sc, outputFile)
    print("Model loading completed!")

    # Perform evals on model
    print("Performing evaluations...")
    labelsAndPredictions = data.map(lambda sample: (sample.label,model.predict(sample.features)))
    trainErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(data.count())
    print("Evaluations completed!")

    print("Accuracy:", 1-trainErr)
    print("Error:", trainErr)
