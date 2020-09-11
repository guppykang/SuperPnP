cd /jbk001-data1/git/SuperPnP/
./setup_nautilus.sh
echo himom

if [ "$1" == "superglueflow" ]; then
  python run_eval.py test -m superglueflow -d tum --run
elif [ "$1" == "siftflow" ]; then
  python run_eval.py test -m siftflow -d tum --run

fi
