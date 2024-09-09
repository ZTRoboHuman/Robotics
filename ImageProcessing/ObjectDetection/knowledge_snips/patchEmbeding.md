## Patch embedding
Patch embedding is a concept primarily used in vision transformers (ViTs) and some other transformer-based models for computer vision. It is a technique to convert an image into a sequence of patch-based vectors (embeddings) that can be processed by a transformer. Since transformers are typically designed for sequential data like text, images need to be transformed into sequences of patches that represent the image content.

### How Patch Embedding Works
1. Splitting the Image into Patches:
- Instead of feeding the entire image pixel-by-pixel into a model like in convolutional neural networks (CNNs), the image is divided into smaller non-overlapping patches.
- For example, given an image of size 𝐻×𝑊×𝐶 (where 𝐻 is the height, 𝑊 is the width, and 𝐶 is the number of channels, such as 3 for an RGB image), the image can be split into patches of size 𝑃×𝑃×𝐶. The total number of patches would be $\frac{H}{P} \times \frac{W}{P}$
2. Flattening Each Patch:
Each patch is treated as a vector by flattening it. For instance, a patch of size 𝑃×𝑃×𝐶 is reshaped into a vector of size $P^2 \times C$.
This means the pixel values within each patch are concatenated into a single vector.
3. Linear Projection:
- After flattening, each patch vector is linearly transformed (via a fully connected layer) to a fixed-size embedding, typically denoted as 𝐷, which is the input size expected by the transformer.
- This transformation serves as a learnable embedding layer, mapping the patch’s pixel values to a high-dimensional space where the transformer can work with the sequence of patches.
4. Positional Embedding:
- Since transformers are permutation-invariant (i.e., they don’t inherently know the order of the input data), positional encodings are added to the patch embeddings to inject information about the position of each patch in the original image.
- These positional encodings allow the model to understand the spatial relationships between different patches, similar to how positional encodings in NLP transformers encode the order of words in a sentence.
5. Sequence of Embeddings:
After embedding and adding positional encodings, the image is represented as a sequence of patch embeddings, which can be fed into the transformer for further processing.

#### Example of Patch Embedding in Vision Transformer (ViT)
In a typical Vision Transformer (ViT):
- An input image of size 224×224×3 is split into patches of size 16×16, resulting in 14×14=196 patches.
- Each patch is flattened and projected into an embedding vector of size 𝐷, typically 768 for ViT-Base.
Notes: you may ask, where is the 3 channels? The answer is it is flatterned. Each patch contains pixel values in all 3 color channels (Red, Green, Blue). When the patch is flattened, the 16×16×3 values are transformed into a 1D vector by concatenating the pixel values from all three channels. The flattened patch will have a total of 16×16×3=768 elements (one for each pixel across all channels).

- Positional encodings are added to these 196 patch embeddings to maintain spatial structure. A positional encoding vector of the same dimension (768) is added to each patch embedding to maintain positional awareness in the transformer.

- The sequence of 196 patch embeddings is then processed by the transformer, treating it similarly to how it would process a sentence with 196 tokens in NLP.

#### Advantages of Patch Embedding
Efficient Handling of Images: Patch embeddings enable transformers, which were originally designed for sequential data (like text), to handle images by treating them as a sequence of patches.

- Flexibility: Patch embeddings make the model agnostic to the specific size of the input image as long as the image can be split into patches of a consistent size.
- Learning Representations: Instead of using predefined convolutional kernels like in CNNs, patch embeddings allow the model to learn from the raw pixel values in a data-driven manner.
- Comparison to CNNs
In contrast to CNNs, which process an image through convolutional layers using sliding windows over small regions (kernels), patch embedding in transformers splits the entire image into patches upfront and then processes them as a sequence. This allows transformers to model global relationships between different parts of the image more naturally.

####  Key Steps in Patch Embedding
  - Divide the image into fixed-size patches.
  - Flatten each patch into a vector.
  - Project the flattened vector into an embedding space via a linear layer.
  - Add positional embeddings to retain the spatial information.
  - Feed the sequence of patch embeddings into a transformer for further processing.

####  Summary
Patch embedding is a technique used in vision transformers to adapt images into a format that transformers can handle by splitting an image into patches, flattening them, and then mapping them into a sequence of vectors (embeddings) that represent the image in a way that the transformer can process. This allows transformers to work with images while capturing long-range dependencies and spatial information efficiently.


Q1. Why and how channel_RGB x H x W to kernel_num x H x W?
```python
self.proj = nn.Conv2d(in_chans=3, out_channels=768, kernel_size=5, stride=2)
```
[A good answer](https://stackoverflow.com/questions/46480699/why-are-my-keras-conv2d-kernels-3-dimensional)

In the typical jargon, when someone refers to a conv layer with N kernels of size (x, y), it is implied that the kernels actually have size (x, y, z), where z is the depth of the input volume to that layer.

Imagine what happens when the input image to the network has R, G, and B channels: each of the initial kernels itself has 3 channels. Subsequent layers are the same, treating the input volume as a multi-channel image, where the channels are now maps of some other feature.

The motion of that 3D kernel as it "sweeps" across the input is only 2D, so it is still referred to as a 2D convolution, and the output of that convolution is a 2D feature map.

A good quote about this in a recent paper, https://arxiv.org/pdf/1809.02601v1.pdf

"In a convolutional layer, the input feature map X is a W1 × H1 × D1 cube, with W1, H1 and D1 indicating its width, height and depth (also referred to as the number of channels), respectively. The output feature map, similarly, is a cube Z with W2 × H2 × D2 entries. The convolution Z = f(X) is parameterized by D2 convolutional kernels, each of which is a S × S × D1 cube."