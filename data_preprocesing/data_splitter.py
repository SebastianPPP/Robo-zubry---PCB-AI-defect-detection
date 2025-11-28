from pathlib import Path
import shutil


TRAINING_PATH = Path('./data/train')
TEST_PATH = Path('./data/test')
VAL_PATH = Path('./data/val')

TRAINING_IMAGES_PATH = TRAINING_PATH / 'images'
TRAINING_LABELS_PATH = TRAINING_PATH / 'labels'

TESTING_IMAGES_PATH = TEST_PATH / 'images'
TESTING_LABELS_PATH = TEST_PATH / 'labels'

VAL_IMAGES_PATH = VAL_PATH / 'images'
VAL_LABELS_PATH = VAL_PATH / 'labels'


TRAIN_LIMIT = 2_000

print(TRAINING_IMAGES_PATH)
print(TRAINING_LABELS_PATH)
data_percent = TRAIN_LIMIT / len([p for p in TRAINING_IMAGES_PATH.iterdir()])
train_image_paths = [p for p in TRAINING_IMAGES_PATH.iterdir()][:2000]
train_label_paths = [p for p in TRAINING_LABELS_PATH.iterdir()][:2000]

TEST_NUM = int(len([p for p in TESTING_IMAGES_PATH.iterdir()]) * data_percent)
test_image_paths = [p for p in TESTING_IMAGES_PATH.iterdir()][:TEST_NUM]
test_label_paths = [p for p in TESTING_LABELS_PATH.iterdir()][:TEST_NUM]

VAL_NUM = int(len([p for p in VAL_IMAGES_PATH.iterdir()]) * data_percent)
val_image_paths = [p for p in VAL_IMAGES_PATH.iterdir()][:VAL_NUM]
val_label_paths = [p for p in VAL_LABELS_PATH.iterdir()][:VAL_NUM]

print(len(train_image_paths))
print(len(train_label_paths))

print(len(test_image_paths))
print(len(test_label_paths))

print(len(val_image_paths))
print(len(val_label_paths))

NEW_TRAIN_IMAGE_DATA = Path('./data_custom/train/images')
NEW_TRAIN_IMAGE_DATA.mkdir(parents=True, exist_ok=True)
NEW_TRAIN_LABELS_DATA = Path('./data_custom/train/labels')
NEW_TRAIN_LABELS_DATA.mkdir(parents=True, exist_ok=True)

NEW_TEST_IMAGE_DATA = Path('./data_custom/test/images')
NEW_TEST_IMAGE_DATA.mkdir(parents=True, exist_ok=True)
NEW_TEST_LABELS_DATA = Path('./data_custom/test/labels')
NEW_TEST_LABELS_DATA.mkdir(parents=True, exist_ok=True)

NEW_VAL_IMAGE_DATA = Path('./data_custom/val/images')
NEW_VAL_IMAGE_DATA.mkdir(parents=True, exist_ok=True)
NEW_VAL_LABELS_DATA = Path('./data_custom/val/labels')
NEW_VAL_LABELS_DATA.mkdir(parents=True, exist_ok=True)

for path in train_image_paths:
    shutil.copyfile(path, NEW_TRAIN_IMAGE_DATA / path.name)
for path in train_label_paths:

    shutil.copyfile(path, NEW_TRAIN_LABELS_DATA / path.name)

for path in test_image_paths:
    shutil.copyfile(path, NEW_TEST_IMAGE_DATA / path.name)
for path in test_label_paths:
    shutil.copyfile(path, NEW_TEST_LABELS_DATA / path.name)

for path in val_image_paths:
    shutil.copyfile(path, NEW_VAL_IMAGE_DATA / path.name)
for path in val_label_paths:
    shutil.copyfile(path, NEW_VAL_LABELS_DATA / path.name)