python facenet/src/classifier.py TRAIN \
~/presidents_aligned \
~/20180402-114759/20180402-114759.pb \
~/presidents_aligned/face_classifier.pkl \
--batch_size 1000 \
--min_nrof_images_per_class 40 \
--nrof_train_images_per_class 35 \
--use_split_dataset
