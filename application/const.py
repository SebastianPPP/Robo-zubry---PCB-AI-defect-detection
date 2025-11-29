from pathlib import Path


# Dirs
DATA_DIR = Path().resolve().parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_TRAIN_DIR = DATA_DIR / 'train'
DATA_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

DATA_TEST_DIR = DATA_DIR / 'test'
DATA_TEST_DIR.mkdir(parents=True, exist_ok=True)

DATA_VAL_DIR = DATA_DIR / 'val'
DATA_VAL_DIR.mkdir(parents=True, exist_ok=True)

# Train dirs
DATA_TRAIN_IMAGES_DIR = DATA_TRAIN_DIR / 'images'
DATA_TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

DATA_TRAIN_LABELS_DIR = DATA_TRAIN_DIR / 'labels'
DATA_TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Test dirs
DATA_TEST_IMAGES_DIR = DATA_TEST_DIR / 'images'
DATA_TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

DATA_TEST_LABELS_DIR = DATA_TEST_DIR / 'labels'
DATA_TEST_LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Validation dirs
DATA_VAL_IMAGES_DIR = DATA_VAL_DIR / 'images'
DATA_VAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

DATA_VAL_LABELS_DIR = DATA_VAL_DIR / 'labels'
DATA_VAL_LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Models
YOLO_V8N_160_E8 = 'YOLO_V8N_160_E8.pt'
YOLO_11N_320_E10 = 'v11n10e320.pt'
YOLO_V11_SMALL = 'v11_small.pt'
BINARY = "binary.pt"
GRAYSCALE = "grayscale.pt"
YOLO_V5NU = "YOLO_V5NU.pt"
