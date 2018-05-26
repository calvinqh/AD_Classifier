# Alzheimer Disease Classifier
The goal of this project is to produce a model that will classify patients and determine if they have Alzheimer's disease . The follow datasets are provided:

 - ROSMAP_RNASeq_entrez.csv
 - ROSMAP_RNASeq_disease_label.csv
 - patients.csv

The two classes that this model will predict are:

 - AD (Alzheimer's Disease)
 - NCI (No cognitive impairment)

## About the datasets
### ROSMAP_RNASeq_entrez.csv
This file contains patient id, diagnosis, and a wide range of entrez ids that fall in the range 1 to 197322. Entrez ids are a identifiers for genes the NCBI  uses in their database [[source]](https://www.wikidata.org/wiki/Property:P351). The diagnosis is the class this training model will be targeting. This class takes in the form of numbers 1-6. Each number is associated with a label for this diagnosis (refer to disease label file).


### ROSMAP_RNASeq_disease_label.csv
This file contains the diagnosis id and the description. The diagnosis id was used in the ROSMAP_RNASeq_entrez file to represent the diease the patient was diagnosied with.

### patients.csv (optional)
This file contains addition information about the patient. The patient id, age, gender and education. These additional features can be used to add additional features to our training data. This additional file, may or may not be introduced in my project.

## Project Workflow
### Part 1: Feature Engineering (Incomplete)
In the ROSMAP_RNASeq_entrez file, the samples are described by large number of features and it also contains samples with MCI (mild cognitive impairment). First off having such a high dimension for our dataset can be troubling because it will take our learning algorithm longer to train. In order to address the problems of high dimensionality we will perform some feature engineering in order to reduce its affer. 

Second, since this model will only be classifying patients in the two classes NCI and AD, we will have to remove samples with MCI and other dementia.

The way I will approach the first problem is to use Map Reduce to first cluster features together. After that I will use Map Reduce to perform a Student T test to select the top-K clusters that will now be used to represent the samples. This new transformed dataset will be used to train my classifier.


### Part 2: Machine Learning
After I have performed feature engineering, in order to reduce the dimension of our samples, it is now time to train our model with our data. The classification algorithm I will be using is SVM.  The framework that I will use to accomplish this is with Hadoop Spark.  Following that I will be carrying out 3 fold cross validation to display the results. The metric that i will be using is accuracy, AUPRC, and AUROC to evaluate the performance of the classifier. 

## Dependencies

 - Data files
	 - ROSMAP_RNASeq_entrez.csv
	 - ROSMAP_RNASeq_disease_label.csv
	 - patients.csv
 - pyspark
 - Python 3
 - NumpPy
 - Pandas
 - Make (optional)
 - virtualenvwrapper

## To run
```
mkvirtualenv ad_class
workon ad_class
pip install -r requirements.txt
python trainModel <output: directoryNameForModel>
```




