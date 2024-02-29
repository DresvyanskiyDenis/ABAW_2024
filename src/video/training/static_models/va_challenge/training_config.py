from typing import Dict

# labels and data paths
VA_TRAIN_LABELS_PATH:str = "/home/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Train_Set/"
VA_DEV_LABELS_PATH:str = "/home/ddresvya/Data/6th ABAW Annotations/VA_Estimation_Challenge/Validation_Set/"
Exp_TRAIN_LABELS_PATH:str = "/home/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set/"
Exp_DEV_LABELS_PATH:str = "/home/ddresvya/Data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set/"

METAFILE_PATH:str = "/home/ddresvya/Data/preprocessed/faces/metadata.csv"

# model architecture params
NUM_CLASSES:int = None
MODEL_INPUT_SIZE:Dict[str, int] = {
	"EfficientNet-B1":224,
	"EfficientNet-B4":380,
	"ViT_b_16":224,
}

# training metaparams
NUM_EPOCHS:int = 100
OPTIMIZER:str = "AdamW"
AUGMENT_PROB:float = 0.05
EARLY_STOPPING_PATIENCE:int = 10
WEIGHT_DECAY:float = 0.0001

# scheduller
LR_SCHEDULLER:str = "Warmup_cyclic"
ANNEALING_PERIOD:int = 5
LR_MAX_CYCLIC:float = 0.005
LR_MIN_CYCLIC:float = 0.0001
LR_MIN_WARMUP:float = 0.00001
WARMUP_STEPS:int = 100
WARMUP_MODE:str = "linear"

# gradual unfreezing
UNFREEZING_LAYERS_PER_EPOCH:int = 1
LAYERS_TO_UNFREEZE_BEFORE_START:int = 7

# Discriminative learning
DISCRIMINATIVE_LEARNING_INITIAL_LR:float = 0.005
DISCRIMINATIVE_LEARNING_MINIMAL_LR:float = 0.00005
DISCRIMINATIVE_LEARNING_MULTIPLICATOR:float = 0.85
DISCRIMINATIVE_LEARNING_STEP:int = 1
DISCRIMINATIVE_LEARNING_START_LAYER:int = -6


# general params
BEST_MODEL_SAVE_PATH:str = "best_models/"
NUM_WORKERS:int = 8
splitting_seed:int = 101095