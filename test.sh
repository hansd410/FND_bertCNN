rm -r outputs

python train.py 30

for i in $(seq 0 29)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cd ../../../cache
	tar -cvf fakeNews.tar config.json pytorch_model.bin
	gzip -f fakeNews.tar
	cd ../
	python test.py
done

mkdir results/experimentV13
mv reports/fakeNews_evaluation_report/* results/experimentV13/

cd results
python mergeResult.py experimentV13
cd ..


for i in $(seq 0 29)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cd ../../../cache
	tar -cvf fakeNews.tar config.json pytorch_model.bin
	gzip -f fakeNews.tar
	cd ../
	python testTrain.py
done

mkdir results/experimentV13_train
mv reports/fakeNews_evaluation_report/* results/experimentV13_train

cd results
python mergeResult.py experimentV13_train
cd ..


