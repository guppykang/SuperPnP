echo 'export PYTHONPATH="/jbk001-data1/git/SuperPnP:$PYTHONPATH"' >> ~/.bashrc
echo 'export JUPYTER_PATH="/jbk001-data1/git/SuperPnP:$JUPYTER_PATH"' >> ~/.bashrc
pip install -r ./requirements_torch.txt
pip install -r ./requirements.txt
git config --global user.email "guppykang@gmail.com"
git config --global user.name "Joshua Kang"
sudo apt update
sudo apt install tmux
