#rm -r outputs

#python train.py 10

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cd ../../../
	python test.py
done

mkdir results/experimentV6
mv reports/fakeNews_evaluation_report/* results/experimentV6/


