cd /jbk001-data1/git/SuperPnP/
./setup_nautilus.sh
echo himom

/opt/conda/bin/python run_eval.py finetuned_depth -m superglueflow -d tum --run

