export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64"
python train.py --batch_size 8 --dataset ./data/voc2012_train.tfrecord --val_dataset ./data/voc2012_val.tfrecord --epochs 100 --mode eager_fit --transfer none
