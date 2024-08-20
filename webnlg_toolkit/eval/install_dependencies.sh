# INSTALL BLEURT-20
pip install --upgrade pip
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .

wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
rm BLEURT-20.zip
cd ../
mv bleurt metrics
# wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip
# unzip bleurt-base-128.zip
# rm bleurt-base-128.zip 
# cd ../
# mv bleurt metrics

# INSTALL METEOR
wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -xvf meteor-1.5.tar.gz
mv meteor-1.5 metrics
rm meteor-1.5.tar.gz

# INSTALL PARENT
# echo "Installing PARENT"
# wget https://github.com/KaijuML/parent/archive/refs/heads/master.zip
# unzip master.zip
# mv parent-master parent
# rm master.zip
# cd parent
# pip install .
# cd ../
# mv parent metrics

# INSTALL SESCORE2
# echo "Installing SESCORE2"
# wget https://github.com/xu1998hz/SEScore2/archive/refs/heads/main.zip
# unzip main.zip
# mv SEScore2-main SEScore2
# rm main.zip
# mv SEScore2 metrics

# INSTALL Data Quest-Eval
# echo "Installing Data Quest-Eval"
# wget https://github.com/ThomasScialom/QuestEval/archive/refs/heads/main.zip
# unzip main.zip
# mv QuestEval-main DQE
# rm main.zip
# mv DQE metrics

