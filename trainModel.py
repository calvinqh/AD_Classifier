import sys
from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel

from app.dataLoader import loadRosmapData

from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics


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
    fold1, fold2, fold3 = data.randomSplit([.3333,.3333,1-.6666], seed=3)
    trainSet1 = fold2.union(fold3)
    trainSet2 = fold1.union(fold3)
    trainSet3 = fold1.union(fold2)
    print("Loading data complete!")
    
    # Train models for each fold
    # Also train model for full dataset
    print("Training started...")
    model = SVMWithSGD.train(data, iterations=100)
    model1 = SVMWithSGD.train(trainSet1, iterations=100)
    model2 = SVMWithSGD.train(trainSet2, iterations=100)
    model3 = SVMWithSGD.train(trainSet3, iterations=100)
    print("Training completed!")
    
    # Save and the model that trained using full dataset
    print("Saving fully trained model...")
    outputLocation = "./models/"
    outputModel = sys.argv[1]
    outputFile = outputLocation+outputModel
    model.save(sc, outputFile)
    print("Model saved.")

    # Evaluate model1-3 (because they used cross eval)
    print("Performing evaluations for test 1...")
    labelsAndPredictions1 = fold1.map(lambda sample: (float(model.predict(sample.features)),sample.label))
    metrics1 = BinaryClassificationMetrics(labelsAndPredictions1)
    m1 = MulticlassMetrics(labelsAndPredictions1)
    print("Evaluations completed!")

    print("Performing evaluations for test 2...")
    labelsAndPredictions2 = fold2.map(lambda sample: (float(model.predict(sample.features)),sample.label))
    metrics2 = BinaryClassificationMetrics(labelsAndPredictions2)
    m2 = MulticlassMetrics(labelsAndPredictions2)
    print("Evaluations completed!")

    print("Performing evaluations for test 3...")
    labelsAndPredictions3 = fold3.map(lambda sample: (float(model.predict(sample.features)),sample.label))
    metrics3 = BinaryClassificationMetrics(labelsAndPredictions3)
    m3 = MulticlassMetrics(labelsAndPredictions3)
    print("Evaluations completed!")

    print("============= RESULTS =============")
    print("Test 1")
    print("AUPRC:", metrics1.areaUnderPR)
    print("AUROC:", metrics1.areaUnderROC)
    print("Accuracy:", m1.accuracy)
    print("-----------------------------------")
    print("Test 2")
    print("AUPRC:", metrics2.areaUnderPR)
    print("AUROC:", metrics2.areaUnderROC)
    print("Accuracy:", m2.accuracy)
    print("-----------------------------------")
    print("Test 3")
    print("AUPRC:", metrics3.areaUnderPR)
    print("AUROC:", metrics3.areaUnderROC)
    print("Accuracy:", m3.accuracy)
    print("-----------------------------------")
    print("Average Results")
    avg_auprc = metrics1.areaUnderPR+metrics2.areaUnderPR+metrics3.areaUnderPR
    avg_auprc/=3
    avg_auroc = metrics1.areaUnderROC+metrics2.areaUnderROC+metrics3.areaUnderROC
    avg_auroc/=3
    avg_acc = m1.accuracy + m2.accuracy + m3.accuracy
    avg_acc/=3
    print("AUPRC:",avg_auprc)
    print("AUROC:",avg_auroc)
    print("Accuracy", avg_acc)
