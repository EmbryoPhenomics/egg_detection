# ConvNets for Egg Detection in microscopy images

This repository includes training and inference code, as well as pre-trained weights, for the detection of Lymnaea stagnalis eggs in 2D microscopy images. Egg detection is achieved through the combination of image classification architectures (e.g. ResNet) with a bounding box regression module to detect single instances of eggs in a given image. Currently there are only weights available for the detection of eggs in Lymnaea stagnalis, though this could be extended to other species with suitable training data. Results of these models are shown below as well as the links to the pre-trained models: 

### Lymnaea stagnalis models

| name | resolution | accuracy | #params | model |
|:---:|:---:|:---:|:---:| :---:|
| Xception | 512x512 | 98.1 | 21M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/Xception_512_lymnaea.h5) 
| ResNet50 | 512x512 | 84.9 | 24M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/ResNet50_512_lymnaea.h5)
| ResNet101 | 512x512 | 83.6 | 43M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/ResNet101_512_lymnaea.h5)
| InceptionV3 | 512x512 | 87.4 | 22M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/InceptionV3_512_lymnaea.h5)
| InceptionResNetV2 | 512x512 | 85.0 | 54M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/InceptionResNetV2_512_lymnaea.h5)
| MobileNet | 512x512 | 96.6 | 3M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/MobileNet_512_lymnaea.h5)
| DenseNet121 | 512x512 | 97.9 | 7M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/DenseNet121_512_lymnaea.h5)
| NASNetMobile | 512x512 | 82.9 | 4M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/NASNetMobile_512_lymnaea.h5)
| EfficientNetB0 | 512x512 | 97.6 | 4M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/EfficientNetB0_512_lymnaea.h5)
| EfficientNetV2B0 | 512x512 | 97.4 | 6M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/EfficientNetV2B0_512_lymnaea.h5)
| EfficientNetV2S | 512x512 | 98.3 | 20M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/EfficientNetV2S_512_lymnaea.h5)
| EfficientNetV2M | 512x512 | 98.1 | 53M | [model](https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/EfficientNetV2M_512_lymnaea.h5)


### Lymnaea stagnalis image dataset

Annotations for training are included in this repository but the source images are available on request as the dataset size is too large for GitHub. Note that the images used for this dataset were captured with the [OpenVim](https://github.com/otills/openvim) phenotyping platform.

### References

For more information on the models used for this application, please refer to the following documentation: https://keras.io/api/applications/