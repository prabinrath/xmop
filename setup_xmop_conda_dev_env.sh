conda create -n xmop_dev python=3.10.13
conda activate xmop_dev
mkdir -p log/
mkdir -p checkpoints/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export PYTHONPATH=`pwd`:$PYTHONPATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh