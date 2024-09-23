mkdir -p log/
mkdir -p checkpoints/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export PYTHONPATH=`pwd`:$PYTHONPATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
sh setup_xmop_dependencies.sh