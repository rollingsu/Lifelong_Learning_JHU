export SAVE_ROOT="D:/SegVol-main/"
export DATASET_CODE="0000"
export IMAGE_DIR="C:\Users\24305\Documents\FLARE22Train\FLARE22Train\images"
export LABEL_DIR="C:\Users\24305\Documents\FLARE22Train\FLARE22Train\labels"
export TEST_RATIO=0.2 

# !!! remember to config the categories in data_process/train_data_process.py file !!!

python data_process/train_data_process.py/
-image_dir $IMAGE_DIR /
-label_dir $LABEL_DIR /
-dataset_code $DATASET_CODE /
-save_root $SAVE_ROOT /
-test_ratio $TEST_RATIO