from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

from keras.applications import *


models = {
    'Xception': (Xception, 'Xception', dict(include_preprocessing=True, method=xception.preprocess_input)),
    'ResNet50': (ResNet50, 'ResNet50', dict(include_preprocessing=False, method=None)),
    'ResNet101': (ResNet101, 'ResNet101', dict(include_preprocessing=False, method=None)),
    'InceptionV3': (InceptionV3, 'InceptionV3', dict(include_preprocessing=True, method=inception_v3.preprocess_input)),
    'InceptionResNetV2': (InceptionResNetV2, 'InceptionResNetV2', dict(include_preprocessing=True, method=inception_resnet_v2.preprocess_input)),
    'MobileNet': (MobileNet, 'MobileNet', dict(include_preprocessing=True, method=mobilenet.preprocess_input)),
    'DenseNet121': (DenseNet121, 'DenseNet121', dict(include_preprocessing=False, method=None)),
    'NASNetMobile': (NASNetMobile, 'NASNetMobile', dict(include_preprocessing=False, method=None)),
    'EfficientNetB0': (EfficientNetB0, 'EfficientNetB0', dict(include_preprocessing=True, method=efficientnet.preprocess_input)),
    'EfficientNetV2B0': (EfficientNetV2B0, 'EfficientNetV2B0', dict(include_preprocessing=False, flag=False, method=None)),
    'EfficientNetV2S': (EfficientNetV2S, 'EfficientNetV2S', dict(include_preprocessing=False, flag=False, method=None)),
    'EfficientNetV2M': (EfficientNetV2M, 'EfficientNetV2M', dict(include_preprocessing=False, flag=False, method=None))
}


weights = {
    'Xception': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/Xception_512_lymnaea.h5',
    'ResNet50': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/ResNet50_512_lymnaea.h5',
    'ResNet101': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/ResNet101_512_lymnaea.h5',
    'InceptionV3': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/InceptionV3_512_lymnaea.h5',
    'InceptionResNetV2': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/InceptionResNetV2_512_lymnaea.h5',
    'MobileNet': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/MobileNet_512_lymnaea.h5',
    'DenseNet121': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/DenseNet121_512_lymnaea.h5',
    'NASNetMobile': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/NASNetMobile_512_lymnaea.h5',
    'EfficientNetB0': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/EfficientNetB0_512_lymnaea.h5',
    'EfficientNetV2B0': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/EfficientNetV2B0_512_lymnaea.h5',
    'EfficientNetV2S': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/EfficientNetV2S_512_lymnaea.h5',
    'EfficientNetV2M': 'https://github.com/EmbryoPhenomics/egg_detection/releases/download/v0.1/EfficientNetV2M_512_lymnaea.h5',
}


def build_model(input_shape=(512, 512, 1), backbone='Xception', pretrained_weights=False):
    '''
    Model construction function

    This function creates a 2D convolutional neural network (CNN) for detecting 
    single instances of objects in images. Here it has been specifically applied to
    the detection of Lymnaea stagnalis eggs in microscopy images but it can be extended
    to other applications. 

    Parameters
    ----------
    input_shape : tuple
        Desired input shape of model. Note if you use the weights from training 
        on the Lymnaea stagnalis dataset you will need an input shape of (512, 512, 1).
    backbone : str
        Name of image classification architecture to use for the model.
    pretrained_weights : bool
        Whether to use pre-trained weights when constructing the model. Currently
        only model weights for the Lymnaea stagnalis dataset are available.

    Returns
    -------
    model : keras.Model
        Keras model instance.

    '''

    backbone, name, preprocessing_kwargs = models[backbone]

    inputs = keras.Input(shape=input_shape)

    if preprocessing_kwargs['include_preprocessing']:
        if preprocessing_kwargs['method'] is not None:
            input_tensor = preprocessing_kwargs['method'](inputs)
        else:
            input_tensor = inputs
    else:
        input_tensor = layers.Rescaling(1.0 / 255)(inputs)

    kwargs = dict(
        weights=None, 
        input_tensor=input_tensor, 
        include_top=False
    ) 

    if 'flag' in preprocessing_kwargs.keys():
        kwargs['include_preprocessing'] = False

    classifier_backbone = backbone(**kwargs)
    x = classifier_backbone.output
    
    # Bounding box regression head
    if 'avg_pool' not in preprocessing_kwargs.keys():
        x = layers.GlobalAveragePooling2D()(x)
        
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
        
    outputs = layers.Dense(4, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    if pretrained_weights:
        url = weights[name]
        pretrained_weights = keras.utils.get_file(origin=url)
        model.load_weights(pretrained_weights)

    return model

if __name__ == '__main__':
    model = build_model(input_shape=(512, 512, 1))
    model.summary()