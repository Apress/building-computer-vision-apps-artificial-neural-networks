%%shell
%tensorflow_version 1.x
export PYTHONPATH=$PYTHONPATH:/content/models/research
export PYTHONPATH=$PYTHONPATH:/content/models/research/slim
cd /content/models/research

python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /content/pre-trained-model/ssd_inception_v2_coco_2018_01_28/steel_defect_pipeline.config \
    --trained_checkpoint_prefix /content/neu-det-models/model.ckpt-10000 \
    --output_directory /content/NEU-DET/final_model