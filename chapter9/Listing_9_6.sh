%%shell
%tensorflow_version 1.x
export PYTHONPATH=$PYTHONPATH:/content/models/research:/content/models/research/slim
cd models/research/
PIPELINE_CONFIG_PATH=/content/pre-trained-model/ssd_inception_v2_coco_2018_01_28/steel_defect_pipeline.config
MODEL_DIR=/content/neu-det-models/
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr