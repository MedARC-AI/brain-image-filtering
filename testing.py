import h5py

# Load 73k NSD images
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/Users/stepheniechen/img2dataset/mscoco/{00000..00024}.tar"  \
    --train-num-samples 23000 \
    --dataset-type webdataset \
    --val-data="/Users/stepheniechen/img2dataset/mscoco/{00025..00031}.tar"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 1000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model RN50 
    