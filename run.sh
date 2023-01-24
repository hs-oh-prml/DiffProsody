export PYTHONPATH=.
DEVICE=0;
DIR_NAME="/workspace/checkpoints/";
MODEL_NAME="DiffProsody";
HPARAMS=""

CONFIG="egs/datasets/audio/vctk/diffprosody.yaml";
CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py --config $CONFIG --exp_name $MODEL_NAME --reset --hparams=$HPARAMS
CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py --config $CONFIG --exp_name $MODEL_NAME --infer --hparams=$HPARAMS

CUDA_VISIBLE_DEVICES=$DEVICE python extract_lpv.py --config $CONFIG --exp_name $MODEL_NAME

MODEL_NAME2="DiffProsodyGenerator";
CONFIG2="egs/datasets/audio/vctk/prosody_generator.yaml";
HPARAMS2="tts_model=$DIR_NAME$MODEL_NAME"

CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py --config $CONFIG2 --exp_name $MODEL_NAME2 --reset --hparams=$HPARAMS2
CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py --config $CONFIG2 --exp_name $MODEL_NAME2 --infer --hparams=$HPARAMS2