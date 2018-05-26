train:
	python trainModel.py test
eval:
	python evalModel.py test
clean:
	(rm -rf models/test;)
	(rm -rf models/clusters;)
