# ViNet: Pushing the limits of Visual Modality for Audio-Visual Saliency Prediction

This project contains the Pytorch implementation of ViNet and AViNet

## Abstract

Here we implement ViNet Architecture for audio-visual saliency prediction. ViNet is a fully convolutional encoder-decoder architecture. The encoder uses visual features from a network trained for action recognition - S3D, and the decoder infers a saliency map via trilinear interpolation and 3D convolutions, combining features from multiple hierarchies. ViNet does not use audio as input, We also implement a novel architecture, AViNet, which uses both visual and audio features for saliency prediction. We use the a pretrained audio network - SoundNet, to extract audio features. 

## Examples
Below are some examples of our model. the first section is the ground truth saliency map, the third section is the saliency map predicted by ViNet.


![](./extras/orig.gif)
![](./extras/pred.gif)

## Dataset
* DHF1K can be downloaded from this [link](https://drive.google.com/drive/folders/1sW0tf9RQMO4RR7SyKhU8Kmbm4jwkFGpQ).
* The three audio-visual datasets used - DIEM, AVAD and ETMD can be downloaded from this [link](http://cvsp.cs.ntua.gr/research/stavis/data/).

## Training
The DHF1K dataset should be structured as follows:
```
└── Dataset  
    ├── Video-Number  
        ├── images  
        |── maps
        └── fixations
```

The audio-visual datasets should be structured as follows:
```
└── Dataset  
    ├── video_frames  
        ├── <dataset_name>
            ├── Video-Name
                ├── frames
    ├── video_audio  
        ├── <dataset_name>
            ├── Video-Name
                ├── audio
    ├── annotations
        ├── <dataset_name>
            ├── Video-Name
                ├── <frame_id>.mat (fixations)
                ├── maps
                    ├── <frame_id>.jpg (ground truth saliency map)
    ├── fold_lists
        ├── <dataset_file>.txt
```

To train ViNet on DHF1K, run the following command:
```bash
$ python3 vinet_train.py --train_data_path /path/to/train --validation_data_path /path/to/val --S3D_weights_file /path/to/S3D/weights
```

To train AViNet on the audio-visual datasets run, run the following command:
```bash
$ python3 avinet_train.py --data_path path/to/data --dataset <dataset_name> --vinet_weights_file /path/to/ViNet/weights
<dataset_name> = DIEM, AVAD, ETMD
```

## Testing
The DHF1K dataset should be structured as follows:
```
└── Dataset  
    ├── Video-Number  
        ├── images  
```

The audio-visual datasets should be structured as follows:
```
└── Dataset  
    ├── video_frames  
        ├── <dataset_name>
            ├── Video-Name
                ├── frames
    ├── video_audio  
        ├── <dataset_name>
            ├── Video-Name
                ├── audio
    ├── fold_lists
        ├── <dataset_file>.txt
```

To test ViNet on DHF1K, run the following command:
```bash
$ python3 vinet_test.py --test_data_path /path/to/test --weight_file /path/to/ViNet/weights --output_path /path/to/output
```

To test AViNet on the audio-visual datasets run, run the following command:
```bash
$ python3 avinet_test.py --data_path path/to/data --dataset <dataset_name> --weight_file /path/to/AViNet/weights --output_path /path/to/output
<dataset_name> = DIEM, AVAD, ETMD
```



