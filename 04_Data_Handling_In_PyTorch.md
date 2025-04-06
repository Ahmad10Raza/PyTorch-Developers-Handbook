# Data Handling in PyTorch


### **Chapter 4: Data Handling in PyTorch**

#### *The Foundation of Effective Model Training*

## **Why Data Handling Matters**

> "Your model is only as good as the data it learns from."

In deep learning, **data preparation** often takes 60-80% of a project's time. This chapter teaches you how to:

- Efficiently load and process large datasets
- Implement custom data pipelines
- Handle real-world challenges like class imbalance
- Accelerate training with optimized data loading

## **Core Topics Overview**

### **4.1 `Dataset` and `DataLoader` Classes**

The backbone of PyTorch data pipelines:

```python
from torch.utils.data import Dataset, DataLoader

dataset = MyDataset(...)  # Your custom dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Key Features**:

- Automatic batching
- Multi-process loading
- Memory-efficient shuffling

### **4.2 Custom Dataset Implementation**

Create datasets for non-standard data formats:

```python
class CustomDataset(Dataset):
    def __init__(self, ...):
        # Initialize file paths/lists
    def __getitem__(self, idx):
        # Return (sample, label) at idx
    def __len__(self):
        # Return total samples
```

### **4.3 Data Augmentation**

Artificially expand your dataset:

```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # For images
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor()
])
```

### **4.4 Batch Processing & Shuffling**

Optimize GPU utilization:

```python
DataLoader(..., 
    batch_size=64,   # Samples per batch
    shuffle=True,    # For epoch-wise shuffling
    num_workers=4    # Parallel data loading
)
```

### **4.5 Handling Imbalanced Datasets**

Solutions for uneven class distributions:

- **Weighted Sampling**
- **Class-weighted Loss Functions**
- **Synthetic Minority Oversampling (SMOTE)**

## **Real-World Impact**

1. **Performance Boost**: Proper data loading can **2-5x training speed**
2. **Model Quality**: Augmentation can improve accuracy by 10-30%
3. **Edge Cases**: Handling imbalance prevents model bias

## **What You'll Gain**

By the end of this chapter, you'll be able to:

- Build **custom data pipelines** for any data type (images, text, audio)
- Implement **GPU-optimized loading** to avoid bottlenecks
- Solve **real-world data challenges** (missing values, imbalance)

---



### **4.1 `Dataset` and `DataLoader` Classes in PyTorch**

#### *The Foundation of Efficient Data Handling*

## **1. Core Concepts**

### **A. The `Dataset` Class**

- **What**: A blueprint for accessing your data
- **Why**: Standardizes how models interact with different data formats
- **Key Methods**:
  ```python
  from torch.utils.data import Dataset

  class CustomDataset(Dataset):
      def __init__(self, ...):  # Initialize paths/transformations
      def __getitem__(self, idx):  # Return (sample, label) at index
      def __len__(self):  # Return total samples
  ```

### **B. The `DataLoader` Class**

- **What**: Manages batching, shuffling, and parallel loading
- **Why**: Optimizes GPU utilization and training speed
- **Key Features**:
  ```python
  DataLoader(
      dataset,
      batch_size=32,
      shuffle=True,
      num_workers=4  # Parallel loading
  )
  ```

## **2. Complete Implementation Example**

### **Image Dataset Template**

```python
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.ToTensor()

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        return self.transform(image), label  # (C,H,W) tensor

    def __len__(self):
        return len(self.image_paths)

# Usage
dataset = ImageDataset(image_paths, labels, 
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

## **3. Critical Features**

### **A. Memory Efficiency**

- **Lazy Loading**: Only loads data when needed (via `__getitem__`)
- **Memory Mapping**: For large files (e.g., numpy `memmap`)

### **B. Parallel Loading**

```python
DataLoader(..., num_workers=4)  # Uses 4 CPU cores
```

**Best Practices**:

- Set `num_workers` to # of CPU cores
- Avoid excessive workers (memory overhead)

### **C. Custom Batching**

```python
def collate_fn(batch):
    # batch = list of (sample, label) tuples
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return images, labels

DataLoader(..., collate_fn=collate_fn)
```

**Use Cases**:

- Variable-length sequences
- Mixed data types

## **4. Performance Optimization**

| Technique                    | Implementation              | Benefit                 |
| ---------------------------- | --------------------------- | ----------------------- |
| **Pin Memory**         | `pin_memory=True`         | Faster GPU transfer     |
| **Prefetching**        | `prefetch_factor=2`       | Overlap loading/compute |
| **Persistent Workers** | `persistent_workers=True` | Reduce process spawning |

**Optimal Setup**:

```python
DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

## **5. Debugging Tips**

### **A. Verify Data Shapes**

```python
for batch in dataloader:
    print(batch[0].shape, batch[1].shape)  # Should match model input
    break
```

### **B. Check Data Loading Bottlenecks**

```python
from torch.utils.data import IterableDataset

class BenchmarkDataset(IterableDataset):
    def __iter__(self):
        while True:
            yield torch.rand(3, 224, 224), 0  # Fake data

# Measure iterations/second
loader = DataLoader(BenchmarkDataset(), batch_size=256)
```

## **Key Takeaways**

1. **`Dataset`** defines how to access your data
2. **`DataLoader`** handles batching/optimization
3. **Parallel loading** (`num_workers`) is critical for performance
4. **Always verify** shapes and loading speed

---



### **4.2 Custom Dataset Implementation in PyTorch**

#### *Handling Non-Standard Data Formats and Complex Pipelines*

## **1. Core Implementation Pattern**

Every custom dataset requires 3 essential methods:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, ...):
        """Initialize paths, transforms, etc."""
        pass
  
    def __getitem__(self, idx):
        """Return (sample, label) at index `idx`"""
        pass
  
    def __len__(self):
        """Return total number of samples"""
        pass
```

## **2. Practical Examples**

### **A. Image Dataset with Augmentation**

```python
from PIL import Image
import torchvision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or T.Compose([
            T.Resize(256),
            T.ToTensor()
        ])
  
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        return self.transform(image), label  # Returns (C,H,W) tensor
  
    def __len__(self):
        return len(self.image_paths)
```

**Usage**:

```python
dataset = ImageDataset(
    image_paths=["img1.jpg", "img2.jpg"],
    labels=[0, 1],
    transform=T.RandomHorizontalFlip()
)
```

### **B. Text Dataset for NLP**

```python
from torchtext.vocab import build_vocab_from_iterator

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab or self.build_vocab()
  
    def build_vocab(self):
        return build_vocab_from_iterator(
            [self.tokenizer(text) for text in self.texts],
            specials=['<unk>', '<pad>']
        )
  
    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        return torch.tensor(self.vocab(tokens)), self.labels[idx]
  
    def __len__(self):
        return len(self.texts)
```

## **3. Advanced Techniques**

### **A. Lazy Loading for Large Datasets**

```python
class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')  # Keeps file open
        self.keys = list(self.file.keys())
  
    def __getitem__(self, idx):
        data = self.file[self.keys[idx]]
        return torch.from_numpy(data['image']), data['label'][()]
  
    def __len__(self):
        return len(self.keys)
  
    def __del__(self):
        self.file.close()  # Cleanup
```

### **B. On-the-Fly Augmentation**

```python
class AudioAugmentDataset(Dataset):
    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.paths[idx])
      
        # Random time stretch
        if random.random() > 0.5:
            rate = 0.9 + 0.2*random.random()
            audio = torchaudio.functional.time_stretch(audio, sr, rate)
      
        return audio, self.labels[idx]
```

### **C. Handling Missing Data**

```python
class RobustImageDataset(Dataset):
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx])
            return self.transform(img), self.labels[idx]
        except Exception:
            return torch.zeros(3,256,256), -1  # Fallback
          
    def __len__(self):
        return len(self.paths)
```

## **4. Performance Optimizations**

| Technique                  | Implementation                | Benefit            |
| -------------------------- | ----------------------------- | ------------------ |
| **Memory Mapping**   | `np.memmap`                 | Large numeric data |
| **Pre-loading**      | Cache in `__init__`         | Faster access      |
| **Parallel Loading** | `DataLoader(num_workers>1)` | Utilize CPU cores  |

**Optimal Setup**:

```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True
)
```

## **5. Debugging Custom Datasets**

### **A. Sanity Check**

```python
# Verify first sample
sample, label = dataset[0]
print("Shape:", sample.shape)
print("Label:", label)

# Check for NaNs
assert not torch.isnan(sample).any()
```

### **B. Visualization (Images)**

```python
import matplotlib.pyplot as plt

plt.imshow(dataset[0][0].permute(1,2,0))
plt.show()
```

### **C. Benchmark Loading**

```python
from timeit import timeit

def test_speed():
    for i in range(100):
        _ = dataset[i]

print(f"Avg load time: {timeit(test_speed, number=1)/100:.4f}s")
```

## **Key Takeaways**

1. **`__getitem__`** must return (sample, label) as tensors
2. **Lazy loading** saves memory for large datasets
3. **Handle edge cases** (missing files, corrupt data)
4. **Optimize I/O** with parallel loading and prefetching

---



### **4.3 Data Augmentation in PyTorch**

#### *Artificially Expand Your Dataset Without Collecting New Data*

## **1. Core Concepts**

**What**: Techniques to generate modified versions of your data**Why**:

- Prevents overfitting
- Improves model robustness
- Works like "free" additional training samples

**Key Principle**:

> "Create realistic variations that preserve semantic meaning"

## **2. Image Augmentation (with `torchvision.transforms`)**

### **A. Common Transformations**

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### **B. Advanced Techniques**

```python
augment = transforms.Compose([
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 3)),
    ], p=0.3),
    transforms.RandomChoice([
        transforms.AutoAugment(),
        transforms.RandAugment()
    ]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
])
```

## **3. Domain-Specific Augmentations**

### **A. Text Data (NLP)**

```python
import nlpaug.augmenter.word as naw

text_aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = text_aug.augment("The quick brown fox")
```

### **B. Audio Data**

```python
import torchaudio

def audio_augment(waveform, sample_rate):
    # Time stretching
    if random.random() > 0.5:
        rate = 0.9 + 0.2*random.random()
        waveform = torchaudio.functional.time_stretch(waveform, sample_rate, rate)
  
    # Pitch shifting
    waveform = torchaudio.functional.pitch_shift(waveform, sample_rate, n_steps=2)
    return waveform
```

### **C. Tabular Data**

```python
def tabular_augment(batch):
    # Gaussian noise injection
    noise = torch.randn_like(batch) * 0.01
    return batch + noise
```

## **4. Implementation Best Practices**

### **A. Dataset Integration**

```python
class AugmentedDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
```

### **B. GPU-Accelerated Augmentation**

```python
# Using Kornia (for batch processing on GPU)
import kornia.augmentation as K

aug = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomGaussianBlur(kernel_size=(3,3), p=0.3),
    data_keys=["input"]  # Specify which tensors to augment
)

batch = torch.rand(16, 3, 224, 224, device='cuda')  # Batch on GPU
augmented_batch = aug(batch)
```

## **5. Special Cases & Debugging**

### **A. Handling Edge Cases**

```python
transform = transforms.Compose([
    transforms.ToPILImage(),  # Ensure input is PIL Image
    transforms.Resize(256),
    transforms.Lambda(lambda x: x if random.random() > 0.1 else x.rotate(90)),
    transforms.ToTensor()
])
```

### **B. Visual Inspection**

```python
import matplotlib.pyplot as plt

def show_augmented_samples(dataset, n=5):
    fig, axes = plt.subplots(1, n, figsize=(15,3))
    for i in range(n):
        img, _ = dataset[i]
        axes[i].imshow(img.permute(1,2,0))
    plt.show()
```

## **6. Performance Considerations**

| Technique                           | When to Use         | Benefit               |
| ----------------------------------- | ------------------- | --------------------- |
| **CPU Augmentation**          | Small datasets      | Simple implementation |
| **GPU Batch Augmentation**    | Large batches       | 5-10x faster          |
| **Pre-computed Augmentation** | Very large datasets | No runtime overhead   |

**Benchmark Example**:

```python
# Compare CPU vs GPU augmentation speed
cpu_time = timeit(lambda: [transform(img) for img in batch], number=10)
gpu_time = timeit(lambda: aug(batch.to('cuda')), number=10)
print(f"CPU: {cpu_time:.2f}s | GPU: {gpu_time:.2f}s")
```

## **Key Takeaways**

1. **Image Domains**: Use `torchvision.transforms`
2. **Other Modalities**: Implement custom augmentations
3. **GPU Acceleration**: Critical for large-scale training
4. **Always Verify**: Visually check augmented samples

---



### **4.4 Batch Processing & Shuffling in PyTorch**

#### *Optimizing Data Flow for Efficient Training*

## **1. Core Concepts**

### **A. Batch Processing**

- **What**: Grouping samples into chunks
- **Why**:
  - Enables GPU parallelism
  - Provides smoother gradient estimates
- **Key Parameters**:
  ```python
  DataLoader(..., batch_size=32)  # Typical values: 32-512
  ```

### **B. Shuffling**

- **What**: Randomizing sample order
- **Why**:
  - Breaks correlated sequences
  - Prevents batch-wise bias
- **Implementation**:
  ```python
  DataLoader(..., shuffle=True)  # For training sets
  ```

## **2. Advanced Batch Handling**

### **A. Custom Batching Logic**

```python
def collate_fn(batch):
    # batch = list of (sample, label) tuples
    images = torch.stack([x[0] for x in batch])
    labels = torch.tensor([x[1] for x in batch])
    return images, labels

DataLoader(..., collate_fn=collate_fn)
```

**Use Cases**:

- Variable-length sequences (pad to max length)
- Mixed data types (images + text)

### **B. Memory-Efficient Shuffling**

```python
# For datasets too large to fit in memory
class ShuffledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = torch.randperm(len(dataset))
  
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
```

## **3. Performance Optimization**

### **A. Parallel Loading**

```python
DataLoader(..., 
    num_workers=4,          # CPU cores for loading
    prefetch_factor=2,      # Batches to preload
    persistent_workers=True  # Maintain workers between epochs
)
```

**Recommendations**:

- Set `num_workers` to # of CPU cores - 1
- Increase `prefetch_factor` for large batches

### **B. GPU Acceleration**

```python
DataLoader(...,
    pin_memory=True  # Faster GPU transfer
)
```

**Mechanism**:
![Data Loading Pipeline](https://pytorch.org/tutorials/_images/sphx_glr_data_loading_tutorial_001.png)
*Source: PyTorch Official Docs*

## **4. Special Cases**

### **A. Uneven Batch Sizes**

```python
# For the last incomplete batch
DataLoader(..., drop_last=False)  # Keep (default) or discard
```

### **B. Weighted Sampling**

```python
weights = [0.1 if label==0 else 0.9 for label in labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights))
DataLoader(..., sampler=sampler)
```

## **5. Debugging Tips**

### **A. Verify Batch Shapes

```python
for batch in dataloader:
    print("Images:", batch[0].shape)  # Should be [B,C,H,W]
    print("Labels:", batch[1].shape)  # Should be [B]
    break
```

### **B. Check Shuffling

```python
# Compare first batches across epochs
epoch1 = next(iter(dataloader))[1]  # Labels
dataloader.dataset.reshuffle()
epoch2 = next(iter(dataloader))[1]
print("Same order?", torch.allclose(epoch1, epoch2))
```

## **Key Takeaways**

1. **Batch Size**: Balance GPU memory and gradient quality
2. **Shuffling**: Critical for training, disable for validation
3. **Parallel Loading**: Essential for GPU-bound training
4. **Custom Logic**: Handle edge cases via `collate_fn`

---



### **Vision (V) in Deep Learning: Core Concepts & PyTorch Implementation**

#### **1. What is Vision in Deep Learning?**

Vision refers to **computer vision (CV)** - the field of enabling machines to interpret and process visual data (images/videos) using neural networks. Key tasks include:

- **Image Classification** (e.g., ResNet)
- **Object Detection** (e.g., YOLO, Faster R-CNN)
- **Semantic Segmentation** (e.g., U-Net)
- **Image Generation** (e.g., GANs, Diffusion Models)

---

#### **2. Fundamental PyTorch Tools for Vision**

| Component                      | Purpose                  | PyTorch Implementation                                |
| ------------------------------ | ------------------------ | ----------------------------------------------------- |
| **Convolutional Layers** | Extract spatial features | `nn.Conv2d(in_channels, out_channels, kernel_size)` |
| **Pooling Layers**       | Downsample feature maps  | `nn.MaxPool2d(kernel_size)`                         |
| **Pretrained Models**    | Transfer learning        | `torchvision.models.resnet18(pretrained=True)`      |
| **Vision Transforms**    | Data augmentation        | `torchvision.transforms`                            |

**Example CNN Architecture**:

```python
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (B,3,H,W) -> (B,16,H,W)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,16,H/2,W/2)
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)                       # (B,32,1,1)
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))
```

#### **3. Key Innovations in Vision**

1. **Convolutional Neural Networks (CNNs)**

   - Local receptive fields (kernel windows)
   - Weight sharing across spatial dimensions
   - Hierarchical feature learning
2. **Vision Transformers (ViTs)**

   - Treat images as patches (e.g., 16x16)
   - Self-attention for global relationships

   ```python
   from transformers import ViTModel
   vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
   ```
3. **Specialized Architectures**

   - **U-Net**: Skip connections for segmentation
   - **YOLO**: Real-time object detection
   - **StyleGAN**: High-resolution image generation

#### **4. Data Handling for Vision**

```python
from torchvision import datasets, transforms

# Augmentation pipeline
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset setup
train_data = datasets.ImageFolder(
    root="path/to/train",
    transform=transform
)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
```

#### **5. Performance Optimization**

| Technique                 | Implementation                                  | Benefit                 |
| ------------------------- | ----------------------------------------------- | ----------------------- |
| **Mixed Precision** | `torch.cuda.amp`                              | 2-3x speedup            |
| **Channels-Last**   | `x = x.to(memory_format=torch.channels_last)` | Better GPU utilization  |
| **TensorRT**        | `torch2trt`                                   | Deployment acceleration |

#### **6. Practical Applications**

1. **Medical Imaging** (X-ray classification)
2. **Autonomous Vehicles** (Object detection)
3. **AR/VR** (3D scene understanding)

**Example Inference**:

```python
model.eval()
with torch.no_grad():
    output = model(test_image.unsqueeze(0))
pred_class = output.argmax().item()
```

#### **Key Takeaways**

1. **CNNs** dominate spatial feature extraction
2. **Transformers** are gaining traction for vision
3. **Data augmentation** is critical for generalization
4. **Pretrained models** accelerate development

---
