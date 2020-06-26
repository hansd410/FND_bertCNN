#cp outputs/fakeNews/1epoch/vocab.txt outputs/fakeNews
#cp outputs/fakeNews/1epoch/config.json outputs/fakeNews/
#cp outputs/fakeNews/1epoch/pytorch_model.bin outputs/fakeNews
#cd outputs/fakeNews
cd outputs/fakeNews/0epoch/
tar -cvf fakeNews.tar config.json pytorch_model.bin
gzip fakeNews.tar
mv fakeNews.tar.gz ../../../cache
cd ../../../
python test.py
