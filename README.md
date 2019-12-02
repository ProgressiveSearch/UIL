# UIL

## Get Started
The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

1. `cd` to folder where you want to download this repo
2. Run `git clone https://github.com/ProgressiveSearch/UIL.git`
3. Install dependencies:
    - [pytorch 1.0.0+](https://pytorch.org/)
    - torchvision
    - tensorboard
    - [yacs](https://github.com/rbgirshick/yacs)
4. Prepare dataset

    Create a directory to store reid datasets under this repo via
    ```bash
    cd reid_baseline
    mkdir datasets
    ```
    1. Download dataset to `datasets/` from [baidu pan](https://pan.baidu.com/s/1ntIi2Op) or [google driver](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
    2. Extract dataset. The dataset structure would like:
    ```bash
    datasets
        Market-1501-v15.09.15
            bounding_box_test/
            bounding_box_train/
    ```
5. Prepare pretrained model.
    If you use origin ResNet, you do not need to do anything. But if you want to use ResNet_ibn, you need to download pretrain model in [here](https://drive.google.com/open?id=1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S). And then you can put it in `~/.cache/torch/checkpoints` or anywhere you like.
    
    Then you should set this pretrain model path in `configs/softmax_triplet.yml`.

6. compile with cython to accelerate evalution
    ```bash
    cd csrc/eval_cylib; make
    ```

## Train
Most of the configuration files that we provide, you can run this command for training market1501
```bash
bash scripts/train_online.sh
```

Or you can just run code below to modify your cfg parameters 
```bash
CUDA_VISIBLE_DEVICES='0,1' python tools/train.py -cfg='configs/softmax_triplet.yml' DATASETS.NAMES '("dukemtmc","market1501",)' SOLVER.IMS_PER_BATCH '256'
```

## Test
You can test your model's performance directly by running this command
```bash
CUDA_VISIBLE_DEVICES='0' python tools/test.py -cfg='configs/softmax_triplet.yml' DATASET.TEST_NAMES 'dukemtmc' \
MODEL.BACKBONE 'resnet50' \
MODEL.WITH_IBN 'True' \
TEST.WEIGHT '/save/trained_model/path'
```
