%%shell
%tensorflow_version 1.x

python /content/generic_xml_to_tf_record.py \
    --label_map_path=/content/steel_label_map.pbtxt \
    --data_dir=/content/NEU-DET \
    --output_path=/content/NEU-DET/out \
    --annotations_dir=ANNOTATIONS \
    --image_dir=IMAGES