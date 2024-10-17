# U-NET++ AND ATTENTION U-NET

### U-NET<br>
U-NET one of the type of computer vision architecture,which is widely used for image segmentation particularly in Bio Medical Feilds.</br><br>
Components of U-Net are:</br>
- Encoder: it performs convolutions followed by pooling operations(Downsampling) to capture high-level features from the input image.
The encoder gradually reduces the spatial dimensions while increasing the number of feature maps.
- BottleNeck: This is the deepest part of the network, where the feature maps are the smallest in spatial dimensions but richest in learned features.
- Decoder: consisting of upsampling operations (transposed convolutions) that gradually recover the spatial dimensions of the feature maps.
It incorporates skip connections by concatenating feature maps from the encoder at corresponding levels, which helps recover fine-grained details lost during downsampling.
- Skip Connection: Fine-grained features are passed along to the corresponding upsampling layers.

![U-NET ARCHITECTURE](https://media.geeksforgeeks.org/wp-content/uploads/20220614121231/Group14.jpg)

<img src="[image-url](https://media.geeksforgeeks.org/wp-content/uploads/20220614121231/Group14.jpg)" alt="Alt text" width="300"/>


### Varients of U-net
* NESTED U-NET(U-NET++)
* ATTENTION U-NET

## NESTED U-NET(U-NET++)



