#rm -r outputs

#python trainCNN.py

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cd ../../../
	python testCNN.py
done

mkdir results/experimentV7
mv reports/fakeNews_evaluation_report/* results/experimentV7/


