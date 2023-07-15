wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh -b -p anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
git clone https://github.com/vuquocan1987/StereotypeWords.git
cd StereotypeWords
conda env create -f textdebias.yaml
conda activate textdebias
git checkout Revamp
pip install -r requirements.txt
pip install -e .
mkdir w2v
python3 generatew2v.py
pytest tests/test_smoke.py
