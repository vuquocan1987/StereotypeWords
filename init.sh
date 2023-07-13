wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh
git clone https://github.com/vuquocan1987/StereotypeWords.git
cd StereotypeWords
conda env create -f textdebias.yaml
conda activate textdebias
pytest tests/test_smoke.py
