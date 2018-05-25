train:
	python trainModel.py test
eval:
	python evalModel.py test
clean:
	(rm -rf models/test;)
	(rm -rf models/test1;)
	(rm -rf models/test2;)
	(rm -rf models/test3;)
