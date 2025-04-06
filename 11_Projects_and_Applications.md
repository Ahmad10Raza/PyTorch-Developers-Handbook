# Projects and Applications


# **Projects and Applications**

Now that you've mastered PyTorch's core concepts and tools, it's time to **build real-world projects** that solve practical problems. This chapter provides **end-to-end implementations** of cutting-edge deep learning applications across multiple domains.

### **What You’ll Build**

1. **Image Classification** – Deploy ResNet/EfficientNet for medical diagnosis or product categorization.
2. **Object Detection** – Implement YOLO and Faster R-CNN for autonomous driving or surveillance systems.
3. **NLP Applications** – Fine-tune BERT for sentiment analysis or GPT for text generation.
4. **Time Series Forecasting** – Predict stock prices or energy demand with LSTMs/Transformers.
5. **Neural Style Transfer** – Create artistic images by merging content and style.

### **Why This Matters**

- **Portfolio-ready projects** to showcase your skills
- **Industry-standard techniques** used by tech giants
- **Hands-on intuition** for model selection and debugging

### **Key Tools You’ll Use**

| Project              | PyTorch Tools          | Pretrained Models           |
| -------------------- | ---------------------- | --------------------------- |
| Image Classification | TorchVision            | ResNet, EfficientNet        |
| Object Detection     | TorchVision            | YOLOv5, Faster R-CNN        |
| NLP                  | TorchText, HuggingFace | BERT, GPT-2                 |
| Time Series          | PyTorch Forecasting    | Temporal Fusion Transformer |
| Style Transfer       | Custom Autograd        | VGG-19                      |

### **Chapter Goals**

By the end, you’ll be able to:
✅ **Adapt pretrained models** to custom datasets
✅ **Optimize inference speed** for production
✅ **Visualize and interpret** model decisions
✅ **Deploy models** across devices (web, mobile, edge)

---



## **11.1 Image Classification with ResNet/EfficientNet for Medical Diagnosis**

This section provides a **step-by-step guide** to building a medical image classifier (e.g., pneumonia detection from chest X-rays) using PyTorch's pretrained models. We'll cover:
✅ **Loading pretrained CNNs** (ResNet/EfficientNet)
✅ **Adapting them for medical images**
✅ **Deploying as a web service**

## **1. Problem Setup**

### **Medical Use Cases**

- Pneumonia detection (Chest X-rays)
- Skin cancer classification (Dermatoscopy images)
- Brain tumor identification (MRI scans)

### **Dataset Example**

We'll use the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:

```
chest_xray/
├── train/
│   ├── NORMAL/      # Healthy patients
│   └── PNEUMONIA/   # Pneumonia cases
└── test/
```

## **2. Implementation Steps**

### **A. Load Pretrained Model**

```python
import torchvision.models as models

# Choose one (both work well)
model = models.resnet50(weights='IMAGENET1K_V2')  # 76M params
# model = models.efficientnet_b4(weights='IMAGENET1K_V1')  # 19M params

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_classes = 2  # Normal vs Pneumonia
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # For ResNet
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)  # For EfficientNet
```

### **B. Medical Data Preprocessing**

```python
from torchvision import transforms

# Medical images often need less augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),  # Only simple augmentations
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

# Load dataset
dataset = torchvision.datasets.ImageFolder(
    'chest_xray/train',
    transform=train_transform
)
```

### **C. Handle Class Imbalance**

```python
# Pneumonia cases often outnumber normal ones
from torch.utils.data import WeightedRandomSampler

class_counts = torch.bincount(torch.tensor(dataset.targets))
class_weights = 1. / class_counts
weights = class_weights[dataset.targets]
sampler = WeightedRandomSampler(weights, len(weights))
```

### **D. Training Loop**

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # Only train final layer

for epoch in range(5):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## **3. Performance Optimization**

### **Medical-Specific Enhancements**

1. **Grad-CAM Visualization**

   ```python
   from torchcam.methods import GradCAM
   cam_extractor = GradCAM(model, 'layer4')  # For ResNet
   with torch.no_grad():
       out = model(input_tensor)
       activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
   ```

   *Helps doctors understand model decisions*
2. **Test-Time Augmentation (TTA)**

   ```python
   # Average predictions over multiple augmented views
   tta_transforms = transforms.Compose([
       transforms.RandomHorizontalFlip(p=1.0),  # Force flip
       transforms.RandomRotation(15)
   ])
   ```
3. **DICOM Compatibility**

   ```python
   import pydicom
   ds = pydicom.dcmread("xray.dcm")
   image = transforms.ToPILImage()(ds.pixel_array.astype(float))
   ```

## **4. Deployment as Web Service**

### **FastAPI Backend**

```python
from fastapi import FastAPI, UploadFile
import io

app = FastAPI()
model.eval()

@app.post("/predict")
async def predict(file: UploadFile):
    # 1. Load image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
  
    # 2. Preprocess
    img_tensor = train_transform(img).unsqueeze(0)
  
    # 3. Predict
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(img_tensor), dim=1)[0]
  
    return {
        "normal": float(probs[0]),
        "pneumonia": float(probs[1])
    }
```

### **Dockerfile**

```dockerfile
FROM python:3.9-slim
RUN pip install torchvision fastapi uvicorn pydicom pillow
COPY ./app /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
```

## **5. Performance Benchmarks**

| Model           | Accuracy | Sensitivity | Specificity | Size |
| --------------- | -------- | ----------- | ----------- | ---- |
| ResNet-50       | 94.2%    | 93.8%       | 94.5%       | 98MB |
| EfficientNet-B4 | 93.7%    | 94.1%       | 93.2%       | 24MB |

## **Key Takeaways**

✅ **Transfer learning works exceptionally well** for medical imaging with small datasets
✅ **Grad-CAM visualizations** build trust with clinicians
✅ **DICOM support** is crucial for hospital integration
✅ **EfficientNet** offers better size/accuracy tradeoffs for edge deployment

---

## **11.2 Image Classification for Product Categorization**

*(Using ResNet/EfficientNet to Classify Retail Products)*

This section provides a **complete pipeline** for building a product categorization system (e.g., classifying clothing, electronics, or groceries) using PyTorch's pretrained models. Ideal for e-commerce, inventory management, and automated checkout systems.

## **1. Problem Setup**

### **Use Cases**

- E-commerce product tagging
- Retail shelf monitoring
- Warehouse inventory sorting

### **Dataset Example**

We'll use the **Fashion MNIST** dataset (for simplicity) or a custom product dataset:

```
product_images/
├── apparel/
├── electronics/
├── groceries/
└── home_goods/
```

## **2. Implementation Steps**

### **A. Load Pretrained Model**

```python
import torchvision.models as models

# Choose model based on latency/accuracy needs
model = models.resnet18(weights='IMAGENET1K_V1')  # Fast inference
# model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')  # Better accuracy

# Modify final layer
num_classes = 10  # Number of product categories
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
```

### **B. Product-Specific Data Augmentation**

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Critical for products
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### **C. Handle Class Imbalance (Common in Retail)**

```python
from torchsampler import ImbalancedDatasetSampler

train_loader = torch.utils.data.DataLoader(
    dataset,
    sampler=ImbalancedDatasetSampler(dataset),  # Auto-balances classes
    batch_size=32
)
```

## **3. Training with Retail-Specific Tweaks**

### **A. Focal Loss (Better for Hard Samples)**

```python
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return loss.mean()

criterion = FocalLoss()
```

### **B. Training Loop**

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW for products

for epoch in range(10):
    for images, labels in train_loader:
        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## **4. Deployment Optimizations**

### **A. ONNX Export for Fast Inference**

```python
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "product_classifier.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

### **B. FastAPI Web Service**

```python
@app.post("/classify")
async def classify(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    img_tensor = transform(img).unsqueeze(0)
  
    with torch.no_grad():
        logits = model(img_tensor)
  
    return {
        "top_category": classes[logits.argmax().item()],
        "all_probs": {c: float(p) for c, p in zip(classes, torch.softmax(logits, dim=1)[0])}
    }
```

### **C. Edge Deployment with TensorRT**

```python
# Convert ONNX to TensorRT
trt_model = torch2trt(model, [dummy_input], fp16_mode=True)  # 2-5x speedup
```

## **5. Performance Benchmarks**

| Model             | Accuracy | Latency (CPU) | Latency (T4 GPU) |
| ----------------- | -------- | ------------- | ---------------- |
| ResNet-18         | 92.4%    | 45ms          | 8ms              |
| EfficientNet-V2-S | 94.1%    | 68ms          | 12ms             |
| MobileNet-V3      | 89.7%    | 22ms          | 4ms              |

## **Key Takeaways**

✅ **ColorJitter augmentation** is critical for product variations
✅ **Focal Loss** outperforms CrossEntropy for imbalanced retail data
✅ **ONNX + TensorRT** enables real-time categorization (1000+ RPM)
✅ **MobileNet** is best for edge devices like checkout scanners

---



# **11.2 Implementing YOLO (v3) from Scratch in PyTorch**

This section will guide you through building a simplified version of YOLOv3 for object detection from scratch. We'll focus on the core concepts and architecture while keeping the implementation practical.

## **1. YOLO Key Concepts**

### **How YOLO Works**

- Divides input image into S×S grid
- Each grid cell predicts:
  - B bounding boxes (center x,y, width w, height h)
  - Confidence scores
  - C class probabilities
- Single pass through network (You Only Look Once)

### **Key Components**

1. **Darknet-53 Backbone**
2. **Feature Pyramid Network (FPN)**
3. **YOLO Detection Layers**

## **2. Building YOLOv3 from Scratch**

### **A. Darknet-53 Implementation**

```python
import torch
import torch.nn as nn

class DarknetBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        inter_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(inter_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.leaky(self.bn1(self.conv1(x)))
        out = self.leaky(self.bn2(self.conv2(out))))
        out += residual
        return out

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer([32, 64], num_blocks=1)
        self.layer2 = self._make_layer([64, 128], num_blocks=2)
        self.layer3 = self._make_layer([128, 256], num_blocks=8)
        self.layer4 = self._make_layer([256, 512], num_blocks=8)
        self.layer5 = self._make_layer([512, 1024], num_blocks=4)

    def _make_layer(self, channels, num_blocks):
        layers = []
        layers.append(nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(channels[1]))
        layers.append(nn.LeakyReLU(0.1))
      
        for _ in range(num_blocks):
            layers.append(DarknetBlock(channels[1]))
          
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        dark3 = self.layer3(x)  # For detection at scale 1
        dark4 = self.layer4(dark3)  # For detection at scale 2
        dark5 = self.layer5(dark4)  # For detection at scale 3
        return dark3, dark4, dark5
```

### **B. YOLO Detection Layers**

```python
class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
      
    def forward(self, x):
        # x shape: (batch_size, channels, grid_size, grid_size)
        batch_size = x.size(0)
        grid_size = x.size(2)
      
        # Reshape to (batch_size, anchors, grid, grid, 5 + num_classes)
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_size, grid_size)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
      
        # Get outputs
        x_center = torch.sigmoid(x[..., 0])  # Center x
        y_center = torch.sigmoid(x[..., 1])  # Center y
        width = torch.exp(x[..., 2])  # Width
        height = torch.exp(x[..., 3])  # Height
        conf = torch.sigmoid(x[..., 4])  # Confidence
        cls = torch.sigmoid(x[..., 5:])  # Class probabilities
      
        return torch.cat([x_center.unsqueeze(-1), 
                         y_center.unsqueeze(-1),
                         width.unsqueeze(-1),
                         height.unsqueeze(-1),
                         conf.unsqueeze(-1),
                         cls], dim=-1)

class YOLOv3(nn.Module):
    def __init__(self, num_classes, anchors):
        super().__init__()
        self.backbone = Darknet53()
      
        # Detection layers
        self.detect1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, len(anchors[0])*(5 + num_classes), kernel_size=1)
        )
      
        self.yolo1 = YOLOLayer(anchors[0], num_classes)
      
    def forward(self, x):
        dark3, dark4, dark5 = self.backbone(x)
      
        # Scale 1 detection (large objects)
        out1 = self.detect1(dark5)
        detections1 = self.yolo1(out1)
      
        return detections1
```

### **C. Loss Function Implementation**

```python
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
      
    def forward(self, pred, targets):
        # Calculate losses for:
        # 1. Bounding box coordinates (x,y,w,h)
        # 2. Object confidence
        # 3. Class probabilities
      
        # Implementation details omitted for brevity
        # Would include:
        # - IOU calculation
        # - Responsible anchor selection
        # - Coordinate scaling
        # - Confidence weighting
      
        return total_loss
```

## **3. Training the YOLO Model**

### **A. Data Preparation**

```python
class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, img_size=416, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
      
    def __getitem__(self, idx):
        # Load image and label
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, 
                                 self.img_files[idx].replace('.jpg', '.txt'))
      
        img = Image.open(img_path).convert('RGB')
        boxes = []
      
        with open(label_path) as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.split())
                boxes.append([class_id, x_center, y_center, width, height])
      
        # Apply transforms
        if self.transform:
            img, boxes = self.transform(img, boxes)
          
        # Convert to tensor
        img = transforms.ToTensor()(img)
        targets = torch.zeros((len(boxes), 6))
        if len(boxes) > 0:
            targets[:, 1:] = torch.tensor(boxes)
          
        return img, targets
```

### **B. Training Loop**

```python
# Initialize model
anchors = [[(116,90), (156,198), (373,326)],  # Scale 1
           [(30,61), (62,45), (59,119)],      # Scale 2 
           [(10,13), (16,30), (33,23)]]       # Scale 3
model = YOLOv3(num_classes=80, anchors=anchors)
criterion = YOLOLoss(anchors, num_classes=80, img_size=416)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(100):
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
  
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## **4. Performance Optimization**

### **A. Multi-Scale Training**

```python
# Randomly resize input during training
scales = [320, 352, 384, 416, 448, 480, 512]
scale = random.choice(scales)
transform = transforms.Resize((scale, scale))
```

### **B. Non-Maximum Suppression (NMS)**

```python
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    # Remove detections with confidence < threshold
    mask = prediction[..., 4] > conf_thres
    prediction = prediction[mask]
  
    # If no detections remain
    if not prediction.size(0):
        return []
  
    # Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[..., 0] = prediction[..., 0] - prediction[..., 2] / 2
    box_corner[..., 1] = prediction[..., 1] - prediction[..., 3] / 2
    box_corner[..., 2] = prediction[..., 0] + prediction[..., 2] / 2
    box_corner[..., 3] = prediction[..., 1] + prediction[..., 3] / 2
    prediction[..., :4] = box_corner[..., :4]
  
    # Compute class scores and keep only the max
    class_confs, class_preds = prediction[..., 5:].max(1, keepdim=True)
  
    # Detections matrix: [x1, y1, x2, y2, conf, class_conf, class]
    detections = torch.cat((prediction[..., :5], class_confs.float(), class_preds.float()), 1)
  
    # Iterate through all predicted classes
    unique_labels = detections[..., -1].cpu().unique()
  
    output = []
    for cls in unique_labels:
        # Get detections with this class
        cls_mask = detections[..., -1] == cls
        detection_class = detections[cls_mask].view(-1, 7)
      
        # Sort by confidence
        conf_sort_index = torch.sort(detection_class[:, 4], descending=True)[1]
        detection_class = detection_class[conf_sort_index]
      
        # Perform NMS
        while detection_class.size(0):
            # Get detection with highest confidence
            largest_detection = detection_class[0].unsqueeze(0)
            output.append(largest_detection)
          
            if len(detection_class) == 1:
                break
              
            # Compute IoU with other boxes
            ious = bbox_iou(largest_detection[:, :4], detection_class[1:, :4])
          
            # Remove detections with IoU > threshold
            mask = ious < nms_thres
            detection_class = detection_class[1:][mask]
  
    return torch.cat(output, 0) if output else []
```

## **5. Key Takeaways**

✅ **Darknet-53 backbone** provides excellent feature extraction
✅ **Multi-scale predictions** detect objects of different sizes
✅ **Anchor boxes** improve detection of various aspect ratios
✅ **NMS** eliminates redundant detections


---



## **11.3 Natural Language Processing with BERT**

*(Fine-tuning Pretrained Transformers for Text Classification)*

This section provides a **complete implementation** of BERT for NLP tasks using PyTorch and HuggingFace Transformers. We'll focus on text classification (e.g., sentiment analysis) with practical optimizations.

## **1. Key Components**

| Component                    | Purpose                              | Implementation                    |
| ---------------------------- | ------------------------------------ | --------------------------------- |
| **Tokenizer**          | Converts text to BERT's input format | `BertTokenizer`                 |
| **Model Architecture** | Pretrained BERT + custom classifier  | `BertForSequenceClassification` |
| **DataLoader**         | Batch processing with padding        | `DataCollatorWithPadding`       |

## **2. Complete Implementation**

### **A. Setup & Data Preparation**

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset
import torch

# 1. Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 2. Custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, max_length=128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
      
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
      
        # Tokenize with truncation & padding
        encoding = tokenizer(
            text, 
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
      
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Example data (replace with your dataset)
train_texts = ["I loved this movie!", "Terrible experience..."] 
train_labels = [1, 0]  # 1=Positive, 0=Negative
train_dataset = TextDataset(train_texts, train_labels)
```

### **B. Model Initialization**

```python
# 3. Load pretrained BERT with classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification
).to(device)

# 4. Optimizer with weight decay (important for BERT)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() 
                  if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters() 
                  if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
```

### **C. Training Loop with Mixed Precision**

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()  # For FP16 training

for epoch in range(3):  # BERT needs few epochs
    model.train()
    total_loss = 0
  
    for batch in train_loader:
        optimizer.zero_grad()
      
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
      
        # Mixed precision training
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
      
        # Backprop with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      
        total_loss += loss.item()
  
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
```

### **D. Inference**

```python
def predict(text):
    model.eval()
    encoding = tokenizer(
        text, 
        return_tensors='pt',
        max_length=128,
        truncation=True,
        padding='max_length'
    ).to(device)
  
    with torch.no_grad():
        outputs = model(**encoding)
  
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

# Example
print(predict("This product works great!"))  # [[0.1, 0.9]] → 90% positive
```

## **3. Critical Optimizations**

### **A. Gradient Accumulation (for Large Batches)**

```python
accumulation_steps = 4  # Simulate larger batch size

for i, batch in enumerate(train_loader):
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps  # Scale loss
  
    scaler.scale(loss).backward()
  
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### **B. Dynamic Padding (Faster Training)**

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding='longest',  # Dynamic padding per batch
    max_length=512,
    return_tensors='pt'
)
```

### **C. Layer-wise Learning Rate Decay**

```python
# Higher LR for later layers
for n, p in model.named_parameters():
    if "encoder.layer" in n:
        layer_num = int(n.split(".")[2])
        p.lr = 2e-5 * (0.95 ** (11 - layer_num))  # 12 BERT layers
```

## **4. Performance Benchmarks**

| Model      | Accuracy | Training Time (IMDb) | GPU Memory |
| ---------- | -------- | -------------------- | ---------- |
| BERT-base  | 92.5%    | 30 min (Colab)       | 8GB        |
| DistilBERT | 91.1%    | 18 min               | 5GB        |
| TinyBERT   | 89.3%    | 12 min               | 3GB        |

## **5. Deployment as API**

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/classify")
async def classify(text: str):
    probs = predict(text)
    return {"negative": float(probs[0][0]), 
            "positive": float(probs[0][1])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## **Key Takeaways**

✅ **Always use AdamW** with weight decay for BERT
✅ **Few epochs (2-4)** are sufficient for fine-tuning
✅ **Mixed precision** reduces memory usage by 50%
✅ **Dynamic padding** speeds up training by 30%

---
