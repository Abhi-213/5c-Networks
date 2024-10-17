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
### Preprocessing of Brain MRI Images:
* CLAHE Preprocessing: Improves image contrast using CLAHE to enhance metastasis visibility.
* Custom Dataset Class: Loads images and corresponding masks, applies preprocessing, and converts them into PyTorch tensors.
* Data Augmentation: Uses Albumentations for transformations like flips, rotations, and brightness adjustments to augment the training data.
* Data Splitting: Divides the dataset into training (80%) and validation (20%) sets using train_test_split.
* DataLoader Setup: Creates PyTorch DataLoader objects for batching and shuffling the training and validation data, essential for efficient model training

### Nested UNET Architecture:

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

```
class NestedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, deep_supervision=False):
        super(NestedUNet, self).__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        # Downsampling path
        self.conv1_1 = ConvBlock(in_channels, nb_filter[0])
        self.conv2_1 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv3_1 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv4_1 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv5_1 = ConvBlock(nb_filter[3], nb_filter[4])

        # Upsampling path
        self.up1_2 = self.up_conv(nb_filter[1], nb_filter[0])
        self.up2_2 = self.up_conv(nb_filter[2], nb_filter[1])
        self.up3_2 = self.up_conv(nb_filter[3], nb_filter[2])
        self.up4_2 = self.up_conv(nb_filter[4], nb_filter[3])

        # Define the ConvBlocks for the upsampling path
        self.conv1_2 = ConvBlock(nb_filter[0] * 2, nb_filter[0])
        self.conv2_2 = ConvBlock(nb_filter[1] * 2, nb_filter[1])
        self.conv3_2 = ConvBlock(nb_filter[2] * 2, nb_filter[2])
        self.conv4_2 = ConvBlock(nb_filter[3] * 2, nb_filter[3])

        # Final convolution layers (deep supervision outputs)
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)  # Final output layer

    def forward(self, x):
        # Downsampling path
        x1_1 = self.conv1_1(x)
        x2_1 = self.conv2_1(F.max_pool2d(x1_1, 2))
        x3_1 = self.conv3_1(F.max_pool2d(x2_1, 2))
        x4_1 = self.conv4_1(F.max_pool2d(x3_1, 2))
        x5_1 = self.conv5_1(F.max_pool2d(x4_1, 2))

        # Upsampling path
        x4_2 = self.up4_2(x5_1)  # Upsample
        x4_2 = torch.cat([x4_2, x4_1], dim=1)  # Concatenate with downsampled x4
        x4_2 = self.conv4_2(x4_2)

        x3_2 = self.up3_2(x4_2)
        x3_2 = torch.cat([x3_2, x3_1], dim=1)
        x3_2 = self.conv3_2(x3_2)

        x2_2 = self.up2_2(x3_2)
        x2_2 = torch.cat([x2_2, x2_1], dim=1)
        x2_2 = self.conv2_2(x2_2)

        x1_2 = self.up1_2(x2_2)
        x1_2 = torch.cat([x1_2, x1_1], dim=1)
        x1_2 = self.conv1_2(x1_2)

        # Final output
        output = self.final(x1_2)
        return output

    def up_conv(self, in_channels, out_channels):
        """Upsample and return a ConvTranspose2d layer"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
```
NestedUNet Class:
This class defines the structure of the Nested U-Net model. It consists of a downsampling path and an upsampling path, allowing for skip connections that improve gradient flow and feature reuse.
* Downsampling:
   - The network uses a series of convolutional layers (ConvBlock), each followed by max pooling. The filter sizes double after each downsampling block.
   - self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1, self.conv5_1: These are convolutional blocks for feature extraction. Each ConvBlock processes the input features, progressively increasing the depth (i.e., number of filters) as the image resolution decreases through max pooling.
* Upsampling:
  - For upsampling, transposed convolution (ConvTranspose2d) is used to increase the resolution of the feature maps.At each stage, the upsampled output is concatenated with the corresponding downsampling output (skip connections).
  - self.up1_2, self.up2_2, self.up3_2, self.up4_2: These layers upsample the feature maps from the deeper layers.
  - self.conv1_2, self.conv2_2, self.conv3_2, self.conv4_2: Convolutional blocks that process the concatenated upsampled features and skip connection outputs.
* Final Output:
  - The final convolutional layer reduces the number of channels to match the output (segmentation map).
  - self.final: The last convolutional layer which produces the final prediction with the specified number of output channels
 
### ATTENTION U-NET ARCHITECTURE:


