
# pip install
pip install -r ./requirements_torch.txt
pip install -r ./requirements.txt
pip install -r ./requirements_cv.txt
# install packages
git clone https://github.com/eric-yyjau/pytorch-superpoint.git
cd pytorch-superpoint
git checkout module_20200707
pip install --upgrade setuptools wheel
python setup.py bdist_wheel
pip install -e .

cd ..
export GIT_LFS_SKIP_SMUDGE=1
git clone https://github.com/eric-yyjau/pytorch-deepFEPE.git
cd pytorch-deepFEPE
git checkout module_20201003
pip install --upgrade setuptools wheel
python setup.py bdist_wheel
pip install -e .
cd ..
