:<<'END'
This is sample bash script for CUB-200-2011 dataset
support model:
vgg, resnet, se_resnet
vgg-16GAP(for localization task)
possible ADL positions:
choose among possible positions
vgg16
'11 12 13 1M 21 22 23 2M 31 32 33 3M 41 42 43 4M 51 52 53 6'
resnet50
'10 11 12 20 21 22 23 30 31 32 33 34 35 40 41 42 '
END

gpu=0,1
name=vgg
epoch=200
decay=60
model=vgg16_ADL
server=tcp://127.0.0.1:12345
batch=128
wd=1e-4
lr=0.001
ADL_pos="3M 4M 53"
data_root="../CUB_200_2011"

CUDA_VISIBLE_DEVICES=${gpu} python train.py -a ${model} --dist-url ${server} \
    --multiprocessing-distributed --world-size 1 --pretrained \
    --data ${data_root} --dataset CUB \
    --train-list datalist/CUB/train.txt \
    --test-list datalist/CUB/test.txt \
    --data-list datalist/CUB/ \
    --task wsol \
    --batch-size ${batch} --epochs ${epoch} --LR-decay ${decay} \
    --wd ${wd} --lr ${lr} --nest --name ${name} \
    --use-ADL --ADL-thr 0.8 --ADL-rate 0.75 --ADL-pos ${ADL_pos}
