%tensorflow_version 1.x
!export PYTHONPATH=$PYTHONPATH:/content/facenet/src
!python facenet/src/train_tripletloss.py \
--logs_base_dir logs/facenet/ \
--models_base_dir /content/drive/'My Drive'/chapter8/facenet_model/ \
--data_dir /content/drive/'My Drive'/chapter8/train_aligned/ \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAGRAD \
--learning_rate 0.01 \
--weight_decay 1e-4 \
--max_nrof_epochs 10 \
--epoch_size 200