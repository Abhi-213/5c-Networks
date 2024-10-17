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
<img src="https://media.geeksforgeeks.org/wp-content/uploads/20230628132335/UNET.webp" alt="NESTED UNET" width="700"/></div>

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
Defines a convolutional block consisting of two convolutional layers, each followed by batch normalization and a ReLU activation function.
Components:
* Convolutional Layer (nn.Conv2d): Applies a 2D convolution operation on the input.
* Batch Normalization (nn.BatchNorm2d): Normalizes the output to stabilize and accelerate training.
* ReLU Activation (nn.ReLU): Adds non-linearity, allowing the network to learn complex patterns.

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
        return self.conv(x)
```
ConvBlock Class:
This block consists of two convolutional layers followed by batch normalization and ReLU activation functions, stacked sequentially.
Components:
* Convolutional Layer (nn.Conv2d): Applies a 2D convolution operation on the input.
* Batch Normalization (nn.BatchNorm2d): Normalizes the output to stabilize and accelerate training.
* ReLU Activation (nn.ReLU): Adds non-linearity, allowing the network to learn complex patterns.

```
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
```
AttentionBlock Class:
The attention block enables the model to focus on relevant parts of the feature map by learning attention weights for each pixel, reducing irrelevant information from other regions.
Components:
* W_g and W_x: Two separate convolutional layers that transform the input from the encoder and the upsampled output from the decoder into a lower-dimensional space (F_int).
* psi: Combines the transformed encoder and decoder feature maps and applies a Sigmoid function to generate attention weights between 0 and 1.
* ReLU: Applied to the sum of the encoder and decoder feature maps before generating the attention weights.
Forward Pass:
* Inputs g (from the decoder) and x (from the encoder) are passed through separate transformation layers (W_g and W_x).
* Their outputs are summed, passed through a ReLU activation, and then through the psi layer to generate attention weights.
* The final output is the encoder feature map x multiplied by the computed attention weights.

```
def forward(self, x):
        # Downsampling path
        x1 = self.conv1(x)
        x2 = self.conv2(F.max_pool2d(x1, 2))
        x3 = self.conv3(F.max_pool2d(x2, 2))
        x4 = self.conv4(F.max_pool2d(x3, 2))
        x5 = self.conv5(F.max_pool2d(x4, 2))

        # Upsampling path
        d5 = self.up5(x5)
        x4 = self.att5(d5, x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.upconv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.upconv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.upconv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.upconv2(d2)

        output = self.final(d2)
        return output
```
AttentionUNet Class:
The main AttentionUNet class defines the complete architecture, which consists of an encoder (downsampling), bottleneck, and decoder (upsampling) paths. Attention blocks are applied in the decoder path to improve segmentation performance.
* Downsampling (Encoder):
  - The encoder consists of five convolutional blocks (conv1 to conv5). Each block is followed by a max pooling operation to reduce the spatial dimensions and capture high-level features. 
  - Conv1 to Conv5: Each block uses ConvBlock, applying convolution, batch normalization, and ReLU activation to extract meaningful features from the input.
* Bottleneck:
   - The deepest layer contains the highest-level feature representations with the lowest spatial resolution.
* Upsampling (Decoder):
  - The decoder restores the spatial dimensions of the image while integrating attention mechanisms to selectively focus on the most relevant encoder features:
  - Transpose Convolution (nn.ConvTranspose2d): Upsamples the feature maps to double their spatial resolution.
  - Attention Blocks (att5, att4, att3, att2): These blocks calculate attention weights to select the most important features from the encoder, helping the model focus on relevant areas of the image.
  - Concatenation: After upsampling and applying attention, the output is concatenated with the corresponding encoder feature maps to combine low-level and high-level features.
  - ConvBlock: After concatenation, the combined feature maps pass through another convolutional block to refine the upsampled output.
* Final Output:
  - The final output layer uses a 1x1 convolution to map the final feature map to the desired number of output channels.

### DICE METRICS
The Dice coefficient (also known as the Sørensen–Dice index) is a commonly used metric for evaluating the performance of segmentation models, particularly in the field of medical image analysis. It measures the overlap between two sets, such as the predicted segmentation mask and the ground truth mask. The Dice score ranges from 0 (no overlap) to 1 (perfect overlap).

The Dice score is calculated as follows:
<div align="center">
Dice coefficient=2⋅∣𝐴∩𝐵∣/∣𝐴∣+∣𝐵∣
Where:
* 𝐴 is the ground truth mask.
* B is the predicted segmentation mask.
* ∣A∣ and ∣B∣ are the cardinalities (number of pixels) in each mask.
</div>
