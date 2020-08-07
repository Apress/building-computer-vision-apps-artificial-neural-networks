%%shell
%tensorflow_version 1.x
export PYTHONPATH=$PYTHONPATH:/content/facenet
export PYTHONPATH=$PYTHONPATH:/content/facenet/src
for N in {1..10}; do \
python facenet/src/align/align_dataset_mtcnn.py \
/content/train \
/content/train_aligned \
--image_size 182 \
--margin 44 \
--random_order \
--gpu_memory_fraction 0.10 \
& done