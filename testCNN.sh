#rm -r outputs

#python trainCNN.py

#for i in $(seq 0 9)
#do
#	cd outputs/fakeNews/${i}epoch/
#	cp config.json ../../../cache
#	cp pytorch_model.bin ../../../cache
#	cp cnnModel ../../../cache
#	cp fcModel ../../../cache
#	cd ../../../
#	python testCNN.py
#done
#
#mkdir results/experimentV14
#mv reports/fakeNews_evaluation_report/* results/experimentV14/

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cp cnnModel ../../../cache
	cp fcModel ../../../cache
	cd ../../../
	python testCNNTrain.py
done

mkdir results/experimentV14_train
mv reports/fakeNews_evaluation_report/* results/experimentV14_train/


