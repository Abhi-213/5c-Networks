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

<div align="center">
<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220614121231/Group14.jpg" alt="UNET" width="700"/></div>

### Varients of U-net
* NESTED U-NET(U-NET++)
* ATTENTION U-NET

## NESTED U-NET(U-NET++)
Nested UNET is an improved version of the UNET which introduces nested and dense skip connections to improve the performance of the model.
Components of U-Net++ are:
* Dense Skip Connections: U-Net++ includes additional convolutional layers in the skip connections to refine the feature maps before concatenation with the decoder layers.
* Deep Supervision: the intermediate outputs of the decoder are used for prediction, which encourages better feature learning at each level of the network.
This helps to reduce the issue of vanishing gradients and allows for more effective training of deeper networks.

<div align="center">
<img src="https://media.springernature.com/lw1200/springer-static/image/art%3A10.1007%2Fs11042-022-13256-6/MediaObjects/11042_2022_13256_Fig1_HTML.png" alt="ATTENTION UNET" width="700"/></div>

## ATTENTION U-NET
ATTENTION UNET is an advanced varient of the UNET architecture,where it incorporates an attention mechanism to enhance the model's focus on relevant features within an image.
Components of ATTENTION U-NET are:
* Attention Gates (AGs):Attention Gates help the model focus on specific parts of the input feature maps, dynamically highlighting regions that are important for segmentation.
* Attention Mechanism:The attention mechanism works by computing a weight map that assigns higher values to the most relevant spatial locations.
* Skip Connections with Attention:skip connections in Attention U-Net carry feature maps from the encoder to the decoder. However, attention gates are applied to these skip connections, refining the features passed between the two paths.

<div align="center">
<img src="https://media.springernature.com/lw1200/springer-static/image/art%3A10.1007%2Fs11042-022-13256-6/MediaObjects/11042_2022_13256_Fig1_HTML.png" alt="ATTENTION UNET" width="700"/></div>

## Computer Vision: Brain MRI Metastasis Segmentation Assignment
Preprocessing of Brain MRI Images:
* CLAHE Preprocessing: Improves image contrast using CLAHE to enhance metastasis visibility.
* Custom Dataset Class: Loads images and corresponding masks, applies preprocessing, and converts them into PyTorch tensors.
* Data Augmentation: Uses Albumentations for transformations like flips, rotations, and brightness adjustments to augment the training data.
* Data Splitting: Divides the dataset into training (80%) and validation (20%) sets using train_test_split.
* DataLoader Setup: Creates PyTorch DataLoader objects for batching and shuffling the training and validation data, essential for efficient model training

Nested UNET Architecture:

```
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
     def forward(self, x):
        return self.conv(x);
 ```
ConvBlock Class:
Defines a convolutional block consisting of two convolutional layers, each followed by batch normalization and a ReLU activation function
