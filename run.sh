python main.py -cnn False

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cp cnnModel ../../../cache
	cp fcModel ../../../cache
	cd ../../../
	python main.py -mode test -bert-data-dir data/test.tsv -cnn False
done

mkdir results/experimentV15
mv reports/fakeNews_evaluation_report/* results/experimentV15/
cd results
python mergeResult.py experimentV15
cd ../


python main.py -cnn False -bert-hidden-dropout 0.3

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cp cnnModel ../../../cache
	cp fcModel ../../../cache
	cd ../../../
	python main.py -mode test -bert-data-dir data/test.tsv -cnn False
done

mkdir results/experimentV16
mv reports/fakeNews_evaluation_report/* results/experimentV16/
cd results
python mergeResult.py experimentV16
cd ../


python main.py -cnn False -bert-hidden-dropout 0.5

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cp cnnModel ../../../cache
	cp fcModel ../../../cache
	cd ../../../
	python main.py -mode test -bert-data-dir data/test.tsv -cnn False
done

mkdir results/experimentV17
mv reports/fakeNews_evaluation_report/* results/experimentV17/
cd results
python mergeResult.py experimentV17
cd ../


python main.py -cnn False -bert-hidden-dropout 0.7

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cp cnnModel ../../../cache
	cp fcModel ../../../cache
	cd ../../../
	python main.py -mode test -bert-data-dir data/test.tsv -cnn False
done

mkdir results/experimentV18
mv reports/fakeNews_evaluation_report/* results/experimentV18/
cd results
python mergeResult.py experimentV18
cd ../


python main.py -cnn False -bert-hidden-dropout 0.9

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cp cnnModel ../../../cache
	cp fcModel ../../../cache
	cd ../../../
	python main.py -mode test -bert-data-dir data/test.tsv -cnn False
done

mkdir results/experimentV19
mv reports/fakeNews_evaluation_report/* results/experimentV19/
cd results
python mergeResult.py experimentV19
cd ../


python main.py -cnn False -bert-att-dropout 0.3

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cp cnnModel ../../../cache
	cp fcModel ../../../cache
	cd ../../../
	python main.py -mode test -bert-data-dir data/test.tsv -cnn False
done

mkdir results/experimentV20
mv reports/fakeNews_evaluation_report/* results/experimentV20/
cd results
python mergeResult.py experimentV20
cd ../


python main.py -cnn False -bert-att-dropout 0.5

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cp cnnModel ../../../cache
	cp fcModel ../../../cache
	cd ../../../
	python main.py -mode test -bert-data-dir data/test.tsv -cnn False
done

mkdir results/experimentV21
mv reports/fakeNews_evaluation_report/* results/experimentV21/
cd results
python mergeResult.py experimentV21
cd ../


python main.py -cnn False -bert-att-dropout 0.7

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cp cnnModel ../../../cache
	cp fcModel ../../../cache
	cd ../../../
	python main.py -mode test -bert-data-dir data/test.tsv -cnn False
done

mkdir results/experimentV22
mv reports/fakeNews_evaluation_report/* results/experimentV22/
cd results
python mergeResult.py experimentV22
cd ../


python main.py -cnn False -bert-att-dropout 0.9

for i in $(seq 0 9)
do
	cd outputs/fakeNews/${i}epoch/
	cp config.json ../../../cache
	cp pytorch_model.bin ../../../cache
	cp cnnModel ../../../cache
	cp fcModel ../../../cache
	cd ../../../
	python main.py -mode test -bert-data-dir data/test.tsv -cnn False
done

mkdir results/experimentV23
mv reports/fakeNews_evaluation_report/* results/experimentV23/
cd results
python mergeResult.py experimentV23
cd ../


#python main.py 
#
#for i in $(seq 0 9)
#do
#	cd outputs/fakeNews/${i}epoch/
#	cp config.json ../../../cache
#	cp pytorch_model.bin ../../../cache
#	cp cnnModel ../../../cache
#	cp fcModel ../../../cache
#	cd ../../../
#	python main.py -mode test -bert-data-dir data/test.tsv
#done
#
#mkdir results/experimentV24
#mv reports/fakeNews_evaluation_report/* results/experimentV24/
#cd results
#python mergeResult.py experimentV24
#cd ../
#
#
#python main.py -cnn-dropout 0.3
#
#for i in $(seq 0 9)
#do
#	cd outputs/fakeNews/${i}epoch/
#	cp config.json ../../../cache
#	cp pytorch_model.bin ../../../cache
#	cp cnnModel ../../../cache
#	cp fcModel ../../../cache
#	cd ../../../
#	python main.py -mode test -bert-data-dir data/test.tsv
#done
#
#mkdir results/experimentV25
#mv reports/fakeNews_evaluation_report/* results/experimentV25/
#cd results
#python mergeResult.py experimentV25
#cd ../
#
#
#python main.py -cnn-dropout 0.5
#
#for i in $(seq 0 9)
#do
#	cd outputs/fakeNews/${i}epoch/
#	cp config.json ../../../cache
#	cp pytorch_model.bin ../../../cache
#	cp cnnModel ../../../cache
#	cp fcModel ../../../cache
#	cd ../../../
#	python main.py -mode test -bert-data-dir data/test.tsv
#done
#
#mkdir results/experimentV26
#mv reports/fakeNews_evaluation_report/* results/experimentV26/
#cd results
#python mergeResult.py experimentV26
#cd ../
#
#
#python main.py -cnn-dropout 0.7
#
#for i in $(seq 0 9)
#do
#	cd outputs/fakeNews/${i}epoch/
#	cp config.json ../../../cache
#	cp pytorch_model.bin ../../../cache
#	cp cnnModel ../../../cache
#	cp fcModel ../../../cache
#	cd ../../../
#	python main.py -mode test -bert-data-dir data/test.tsv
#done
#
#mkdir results/experimentV27
#mv reports/fakeNews_evaluation_report/* results/experimentV27/
#cd results
#python mergeResult.py experimentV27
#cd ../
#
#
#python main.py -cnn-dropout 0.9
#
#for i in $(seq 0 9)
#do
#	cd outputs/fakeNews/${i}epoch/
#	cp config.json ../../../cache
#	cp pytorch_model.bin ../../../cache
#	cp cnnModel ../../../cache
#	cp fcModel ../../../cache
#	cd ../../../
#	python main.py -mode test -bert-data-dir data/test.tsv
#done
#
#mkdir results/experimentV28
#mv reports/fakeNews_evaluation_report/* results/experimentV28/
#cd results
#python mergeResult.py experimentV28
#cd ../

