#  **Neural Network Basics**

#### *From Mathematical Foundations to PyTorch Implementation*

### **Introduction**

Neural networks are the building blocks of modern deep learning. This chapter bridges the gap between **theory** (how neurons learn) and **practice** (implementing networks in PyTorch). You'll learn to:

- Design network architectures from scratch
- Properly initialize and regularize models
- Debug common training issues

**Key Philosophy**:

> "Understand the math, then let PyTorch handle the derivatives."

### **Chapter Topics**

#### **3.1 The Perceptron: Building Block of Neural Nets**

- Biological inspiration vs. mathematical abstraction
- Implementing a perceptron in PyTorch
- Limitations of linear decision boundaries

#### **3.2 Activation Functions**

| Function  | Use Case              | PyTorch Implementation       |
| --------- | --------------------- | ---------------------------- |
| ReLU      | Hidden layers         | `torch.nn.ReLU()`          |
| Sigmoid   | Binary classification | `torch.nn.Sigmoid()`       |
| Tanh      | RNNs/Hidden layers    | `torch.nn.Tanh()`          |
| LeakyReLU | Avoid dead neurons    | `torch.nn.LeakyReLU(0.01)` |

#### **3.3 Designing Network Architectures**

- Layer stacking with `nn.Sequential`
- Custom `nn.Module` subclassing
- Skip connections (ResNet basics)

#### **3.4 Weight Initialization**

```python
# Best practices
nn.init.kaiming_normal_(layer.weight, mode='fan_out')  
nn.init.constant_(layer.bias, 0)  
```

#### **3.5 Loss Functions**

| Task                       | Loss Function        | PyTorch Class             |
| -------------------------- | -------------------- | ------------------------- |
| Regression                 | Mean Squared Error   | `nn.MSELoss()`          |
| Binary Classification      | Binary Cross-Entropy | `nn.BCELoss()`          |
| Multi-class Classification | Cross-Entropy        | `nn.CrossEntropyLoss()` |

#### **3.6 Training Loop Anatomy**

```python
for epoch in range(epochs):
    for X, y in dataloader:
        optimizer.zero_grad()
        outputs = model(X)  
        loss = criterion(outputs, y)
        loss.backward()  
        optimizer.step()
```

#### **3.7 Debugging Neural Nets**

- Gradient checking (`torch.autograd.gradcheck`)
- Overfitting a small batch test
- Visualizing activations with TensorBoard

### **Why This Matters**

1. **Foundation**: All advanced architectures (CNNs, RNNs, Transformers) build on these basics.
2. **Debugging Skills**: 80% of model failures stem from improper basics (e.g., bad initialization).
3. **Flexibility**: Custom `nn.Module` lets you implement cutting-edge research.

---



### **3.1 The Perceptron: Building Block of Neural Nets**

#### *From Biological Inspiration to Mathematical Model*

## **1. Biological Inspiration**

![p](https://res.cloudinary.com/dyl5ibyvg/image/upload/v1743883126/ymeanbj3k5qp58ajwwnz.png)

![Perceptron Diagram](https://res.cloudinary.com/dyl5ibyvg/image/upload/v1743009819/ew79trka9nsj6xutkfi5.png)

## **2. Mathematical Formulation**

![Perceptron](https://res.cloudinary.com/dyl5ibyvg/image/upload/v1743882934/aekbxrdytqisd29iaaby.png)

**PyTorch Implementation**:

```python
import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # w*x + b
        self.step = lambda x: (x > 0).float()  # Step activation

    def forward(self, x):
        return self.step(self.linear(x))
```

## **3. Key Properties**

### **A. Decision Boundary**

- For 2D inputs: (w_1x_1 + w_2x_2 + b = 0) defines a **line**
- For 3D+: A **hyperplane** separating classes

**Example**:

```python
# Manually set weights (OR gate)
perceptron = Perceptron(input_dim=2)
perceptron.linear.weight.data = torch.tensor([[1., 1.]])
perceptron.linear.bias.data = torch.tensor([-0.5])

# Test
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
print(perceptron(inputs))  # tensor([[0.], [1.], [1.], [1.]])
```

### **B. Limitations**

- **Cannot solve non-linearly separable problems** (e.g., XOR)
- **No probabilistic output** (just 0 or 1)

## **4. Training the Perceptron**

### **A. Perceptron Learning Algorithm**

![P](https://res.cloudinary.com/dyl5ibyvg/image/upload/v1743883126/nwnp8yzrgs4te3xmf3md.png)

**PyTorch Training Loop**:

```python
def train_perceptron(model, X, y, lr=0.1, epochs=100):
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model.linear(X)  # Raw logits (no step)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
```

### **B. Modern Equivalent**

Today, we use:

- **Logistic regression** (probabilistic output)
- **Multi-layer perceptrons (MLPs)** for non-linear problems

## **5. From Perceptron to Neural Networks**

- **Stack perceptrons** → Hidden layers
- **Replace step function** with differentiable activations (ReLU, Sigmoid)
- **Train with backpropagation**

**Upgrade to MLP**:

```python
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4),  # Hidden layer
            nn.ReLU(),
            nn.Linear(4, 1)          # Output layer
        )

    def forward(self, x):
        return torch.sigmoid(self.layers(x))  # Probabilistic output
```

## **Key Takeaways**

1. Perceptrons are **linear binary classifiers**
2. Limited to **linearly separable problems**
3. Modern networks stack perceptrons with:
   - Non-linear activations
   - Multiple hidden layers
4. Training uses **gradient descent** (not just perceptron rule)

---



### **1. Standard Activations (Native PyTorch)**

```python
import torch
import torch.nn as nn

# Input tensor
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Native implementations
sigmoid = torch.sigmoid(x)            # 1/(1+e^(-x))
tanh = torch.tanh(x)                  # (e^x - e^(-x))/(e^x + e^(-x))
relu = torch.relu(x)                  # max(0, x)
leaky_relu = nn.LeakyReLU(0.1)(x)     # max(0.1*x, x)
softmax = torch.softmax(x, dim=-1)    # e^x / sum(e^x)
gelu = nn.GELU()(x)                   # x * Φ(x) (used in Transformers)
```

### **2. Custom Activation Implementations**

```python
# Swish (self-gated)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Mish (smooth alternative to ReLU)
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

# ELU (Exponential Linear Unit)
class ELU(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

# Quick test
x = torch.tensor([-1.0, 0.0, 1.0])
print(Swish()(x))  # tensor([-0.2689,  0.0000,  0.7311])
```

### **3. Usage in Neural Networks**

```python
# Example 1: Sequential model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),          # Standard choice for hidden layers
    nn.Linear(256, 64),
    Mish(),             # Custom activation
    nn.Linear(64, 10),
    nn.Softmax(dim=1)   # Output layer for classification
)

# Example 2: Custom forward pass
def forward(self, x):
    x = self.linear1(x)
    x = torch.where(x > 0, x, 0.01 * x)  # LeakyReLU manual implementation
    return self.linear2(x)
```

### **4. Advanced: Memory-Efficient Variants**

```python
# Memory-efficient Swish (saves memory during training)
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))

def swish(x):
    return SwishImplementation.apply(x)
```

### **5. Key Notes for Production**

1. **Prefer Native Functions** when possible (optimized in PyTorch):
   ```python
   nn.ReLU()  # Faster than manual torch.where(x > 0, x, 0)
   ```
2. **For Custom Activations**:
   - Inherit from `nn.Module` for easy integration
   - Use `torch.autograd.Function` for memory optimization
3. **Debug NaN Values**:
   ```python
   def safe_activation(x, eps=1e-6):
       return torch.clamp(nn.ReLU()(x), min=0, max=1/eps)
   ```

### **Complete Example: Activation Benchmark**

```python
import time

activations = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(0.1),
    "Swish": Swish(),
    "Mish": Mish()
}

x = torch.randn(1_000_000)  # Large tensor

for name, activation in activations.items():
    start = time.time()
    y = activation(x)
    torch.cuda.synchronize()  # For GPU timing
    print(f"{name}: {time.time() - start:.4f} sec")
```

This gives you **production-ready implementations** with performance considerations. For most cases, stick with native PyTorch (`nn.ReLU`, `nn.Sigmoid`, etc.), and use custom classes only for research or specific needs.


---



### **1. Sequential Models (`nn.Sequential`)**

**What**: A linear stack of layers where output of one layer feeds directly into the next.**Why**:

- Simplest way to build feedforward networks
- Easy to read and debug
  **Code**:

```python
model = nn.Sequential(
    nn.Linear(784, 256),  # Layer 1: 784 inputs → 256 outputs
    nn.ReLU(),            # Activation
    nn.Linear(256, 10)    # Layer 2: 256 → 10 outputs
)
```

**Analogy**: Like an assembly line - data moves straight through each station (layer).

### **2. Custom `nn.Module` Class**

**What**: A flexible template for complex architectures.**Why**:

- Full control over forward pass logic
- Can implement non-sequential flows (skips, branches, etc.)
  **Code**:

```python
class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)  # Define layers
        self.layer2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))  # Custom forward logic
        return self.layer2(x)
```

**Key Insight**: The `forward()` method defines *how data flows*, separate from layer definitions.

### **3. Skip Connections (ResNet-style)**

**What**: Shortcut paths that bypass one or more layers.**Why**:

- Solves vanishing gradients in deep networks
- Preserves information across layers
  **Code**:

```python
def forward(self, x):
    residual = x  # Save input
    x = F.relu(self.conv1(x))
    x = self.conv2(x)
    return F.relu(x + residual)  # Add original input
```

**Visualization**:

```
Input → Conv1 → Conv2 → + → Output
  ˉˉˉˉˉˉˉˉˉˉˉˉˉ↑
```

**Impact**: Allows training networks with 1000+ layers.

### **4. Multi-Branch Networks**

**What**: Parallel processing paths that merge later.**Why**:

- Extract different features simultaneously
- Common in vision (e.g., Inception)
  **Code**:

```python
def forward(self, x):
    branch1 = self.conv1(x)  # Path 1
    branch2 = self.conv2(x)  # Path 2
    return torch.cat([branch1, branch2], dim=1)  # Merge
```

**Use Case**: Processing an image at different scales simultaneously.

### **5. Weight-Sharing (Siamese Nets)**

**What**: Multiple inputs processed by the same layer weights.**Why**:

- Compare two inputs (e.g., face verification)
- Efficient feature extraction
  **Code**:

```python
def forward(self, x1, x2):
    feat1 = self.encoder(x1)  # Same weights
    feat2 = self.encoder(x2)  # Used again
    return self.head(feat1), self.head(feat2)
```

**Example**: Twin networks for similarity learning.

### **6. Dynamic Computation**

**What**: Networks that adjust depth/computation on-the-fly.**Why**:

- Save computation on easy inputs
- Adaptive inference speed
  **Code**:

```python
def forward(self, x, depth=3):
    for layer in self.layers[:depth]:  # Only use 'depth' layers
        x = F.relu(layer(x))
    return x
```

**Application**: Conditional computation for edge devices.

### **Why These Patterns Matter**

1. **Sequential**: Baseline for simple tasks
2. **Custom Modules**: Research/new architectures
3. **Skip Connections**: Deep networks that actually train
4. **Branches**: Capture complex patterns
5. **Weight-Sharing**: Efficient comparison tasks
6. **Dynamic**: Real-world efficiency

Each pattern solves specific challenges in deep learning. The code shows *how to implement them in PyTorch*, while the explanations clarify *when and why* to use them.


---



### **3.4 Weight Initialization in PyTorch**

#### *Why It Matters and How to Do It Right*

## **1. The Importance of Initialization**

**Problem**: Poor initialization leads to:

- **Vanishing gradients** (weights → 0)
- **Exploding gradients** (weights → ∞)
- **Dead neurons** (e.g., ReLUs never activate)

**Solution**: Proper initialization sets weights to optimal starting values for training.

## **2. Common Initialization Methods**

![Weight](https://res.cloudinary.com/dyl5ibyvg/image/upload/v1743884716/qrczzbh4xwnticuzqusa.png)

## **3. PyTorch Implementation**

### **A. Manual Initialization**

```python
# For a Linear layer
layer = nn.Linear(100, 50)

# Xavier/Glorot (uniform)
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)  # Common for biases

# Kaiming/He (normal)
nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
```

### **B. Custom Initialization Function**

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

model.apply(init_weights)  # Applies to all layers
```

## **4. Special Cases**

### **A. LSTM/GRU Initialization**

```python
for name, param in lstm.named_parameters():
    if 'weight_ih' in name:  # Input-hidden weights
        nn.init.xavier_uniform_(param)
    elif 'weight_hh' in name: # Hidden-hidden weights
        nn.init.orthogonal_(param)
    elif 'bias' in name:
        param.data.fill_(0)
        # Forget gate bias often initialized to 1
        param.data[hidden_size:hidden_size*2].fill_(1)
```

### **B. Pretrained Weights**

```python
# Load pretrained weights but initialize new head
model = resnet18(pretrained=True)
nn.init.kaiming_normal_(model.fc.weight)  # New final layer
```

## **5. Debugging Initialization**

### **A. Weight Statistics**

```python
def print_weight_stats(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")

print_weight_stats(model)
```

**Ideal Output**:

- Mean near 0
- Std dev matches initialization scheme (e.g., ~0.01 for Kaiming)

### **B. Gradient Checking**

```python
# After first backward pass
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}!")
```

## **6. Best Practices**

1. **Default to Kaiming** for ReLU networks
2. **Use zeros for biases** (except forget gates in LSTMs)
3. **Match initialization to activation**:
   - Xavier for Tanh/Sigmoid
   - Kaiming for ReLU/LeakyReLU
4. **Verify gradients** after first batch

## **Key Takeaways**

1. Initialization determines if training succeeds or fails
2. **Kaiming (He)** is the modern default for ReLU networks
3. Always **check weight statistics** before training
4. **LSTMs need special care** (orthogonal for hidden weights)

---



### **What Comes Next?**

After mastering **weight initialization**, here's what you should explore to build a complete deep learning pipeline:

## **1. Immediate Next Steps**

### **3.5 Loss Functions**

*How to quantify "how wrong" your model is*

- **Classification**: Cross-Entropy, Focal Loss
- **Regression**: MSE, MAE, Huber Loss
- **Custom Losses**: Implementing domain-specific objectives

**Key Question**:
*"Are we punishing the model for the right mistakes?"*

### **3.6 Optimizers**

*Algorithms that update weights based on gradients*

- **SGD** (with/without momentum)
- **Adam/AdamW** (adaptive learning rates)
- **Advanced**: LAMB, RAdam

**Key Insight**:
*"The optimizer determines how efficiently you descend the loss landscape."*

## **2. Connecting the Dots**

| Concept                  | Why It Matters                       |
| ------------------------ | ------------------------------------ |
| **Initialization** | Sets starting point for optimization |
| **Loss Functions** | Defines what "good" looks like       |
| **Optimizers**     | Determines how to improve            |

**Example Training Loop**:

```python
# Initialization
model.apply(init_weights)  # 3.4

# Loss
criterion = nn.CrossEntropyLoss()  # 3.5  

# Optimizer
optimizer = torch.optim.Adam(model.parameters())  # 3.6

for x, y in dataloader:
    optimizer.zero_grad()
    outputs = model(x)  
    loss = criterion(outputs, y)  
    loss.backward()  
    optimizer.step()  
```

## **3. Beyond the Basics**

### **3.7 Debugging Neural Nets**

- Gradient checking (`torch.autograd.gradcheck`)
- Overfitting a small batch
- Visualizing training with TensorBoard

### **3.8 Regularization Techniques**

- Dropout (`nn.Dropout`)
- Weight decay (L2 regularization)
- Batch Normalization (`nn.BatchNorm1d`)

---



### **3.6 Training Loop Anatomy**

#### *The Step-by-Step Engine of Neural Network Learning*

## **1. Core Components of a Training Loop**

Every PyTorch training loop consists of **5 critical steps**:

```python
for epoch in range(epochs):          # 1. Epoch loop
    for batch in dataloader:        # 2. Batch iteration
        # ------ Critical Steps ------
        optimizer.zero_grad()       # 3. Reset gradients
        outputs = model(batch)      # 4. Forward pass
        loss = criterion(outputs, targets)  
        loss.backward()             # 5. Backpropagation
        optimizer.step()            # 6. Update weights
        # ---------------------------
```

## **2. Breakdown of Each Step**

### **Step 1: Epoch Loop**

- **What**: One full pass through the entire dataset
- **Why**: Multiple epochs allow gradual learning
- **Code**:
  ```python
  for epoch in range(num_epochs):
      print(f"Epoch {epoch+1}/{num_epochs}")
  ```

### **Step 2: Batch Iteration**

- **What**: Process data in chunks (batches)
- **Why**:
  - Fits larger datasets in memory
  - Noisy updates help escape local minima
- **Code**:
  ```python
  for images, labels in train_loader:  # From DataLoader
      # images.shape = [batch_size, channels, height, width]
  ```

### **Step 3: Zeroing Gradients**

- **What**: Reset gradient buffers
- **Why**: PyTorch **accumulates** gradients by default
- **Code**:
  ```python
  optimizer.zero_grad()  # Clear old gradients
  # Equivalent to model.zero_grad()
  ```

### **Step 4: Forward Pass**

- **What**: Compute predictions
- **Why**: Generate loss for backpropagation
- **Code**:
  ```python
  outputs = model(images)  # Calls model.forward()
  loss = criterion(outputs, labels)
  ```

### **Step 5: Backward Pass**

- **What**: Calculate gradients via chain rule
- **Why**: Know how to update weights
- **Code**:
  ```python
  loss.backward()  # Populates .grad attributes
  ```

### **Step 6: Weight Update**

- **What**: Adjust weights using gradients
- **Why**: Minimize loss function
- **Code**:
  ```python
  optimizer.step()  # Applies gradients (e.g.: w = w - lr*w.grad)
  ```

## **3. Advanced Loop Features**

### **A. Gradient Clipping**

Prevents exploding gradients in RNNs:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(), 
    max_norm=1.0
)
```

### **B. Mixed Precision Training**

Faster training with GPU optimization:

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### **C. Learning Rate Scheduling**

Dynamic learning rate adjustment:

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in epochs:
    # Training loop...
    scheduler.step()  # Update LR after epoch
```

## **4. Complete Training Template**

```python
def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
          
            # Reset gradients
            optimizer.zero_grad()
          
            # Forward + backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
          
            # Optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          
            # Update weights
            optimizer.step()
          
            # Track statistics
            running_loss += loss.item()
      
        print(f"Epoch {epoch+1} Loss: {running_loss/len(dataloader):.4f}")
```

## **5. Debugging Checklist**

| Issue                    | Solution                                     |
| ------------------------ | -------------------------------------------- |
| **NaN Loss**       | Check input data, reduce learning rate       |
| **Zero Gradients** | Verify `requires_grad=True` for parameters |
| **Slow Training**  | Enable CUDA, use larger batches              |
| **Overfitting**    | Add dropout/regularization                   |

## **Key Takeaways**

1. **Always zero gradients** before `.backward()`
2. **Batch processing** enables large-scale training
3. **Monitor loss** to detect training issues
4. **Advanced techniques** (clipping, scheduling) stabilize training

---



### **3.7 Debugging Neural Networks in PyTorch**

#### *How to Diagnose and Fix Common Training Problems*

## **1. Core Debugging Tools**

### **A. Gradient Flow Analysis**

```python
# Check gradient statistics
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: mean={param.grad.mean():.4f}, std={param.grad.std():.4f}")
```

**What to look for**:

- **Vanishing gradients**: All values ≈ 0
- **Exploding gradients**: Values > 1e3

### **B. Activation Monitoring**

```python
# Hook to record activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.layer1.register_forward_hook(get_activation('layer1'))
```

## **2. Common Problems & Solutions**

| Symptom                  | Diagnosis                | Fix                                                 |
| ------------------------ | ------------------------ | --------------------------------------------------- |
| **Loss = NaN**     | Exploding gradients      | Gradient clipping (`nn.utils.clip_grad_norm_`)    |
| **Zero accuracy**  | Wrong loss function      | Verify `nn.CrossEntropyLoss()` for classification |
| **Plateaued loss** | Poor initialization      | Use Kaiming/Xavier initialization                   |
| **Overfitting**    | High train, low test acc | Add dropout (`nn.Dropout(0.5)`)                   |
| **Slow training**  | Small LR/bad optimizer   | Try Adam with `lr=3e-4`                           |

## **3. Step-by-Step Debugging Protocol**

### **1. Overfit a Tiny Dataset**

```python
# Test with 1-10 samples
small_data = train_dataset[:10]
train_loader = DataLoader(small_data, batch_size=2)

# Should reach 100% accuracy quickly
if not can_overfit(model, train_loader):
    print("Architecture problem!")
```

### **2. Verify Forward Pass**

```python
# Check tensor shapes
x, y = next(iter(train_loader))
print("Input shape:", x.shape)
try:
    out = model(x)
    print("Output shape:", out.shape) 
except RuntimeError as e:
    print("Shape mismatch:", e)
```

### **3. Check Parameter Updates**

```python
# Track weight changes
before = model.fc1.weight.clone()
optimizer.step()
after = model.fc1.weight
print("Weight delta:", (after - before).abs().mean())
```

## **4. Advanced Techniques**

### **A. Gradient Checking (Numerical vs Analytical)**

```python
from torch.autograd import gradcheck

input = torch.randn(3, dtype=torch.double, requires_grad=True)
test = gradcheck(model, input, eps=1e-6)
print("Gradient check passed:", test)
```

### **B. Visualizing Learning**

```python
# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

for epoch in epochs:
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_histogram('Weights/fc1', model.fc1.weight, epoch)
```

## **5. Debugging Checklist**

1. **Data Issues**

   - Check for NaN/inf in inputs (`torch.isnan(x).any()`)
   - Verify label ranges (e.g., 0-9 for MNIST)
2. **Model Issues**

   - Print layer output shapes
   - Test with dummy input (`torch.randn(batch_size, input_dim)`)
3. **Optimization Issues**

   - Monitor gradient norms
   - Try different learning rates (log scale: 1e-4 to 1e-2)

## **Key Takeaways**

1. **Start small**: Overfit a mini-batch first
2. **Monitor distributions**: Weights, gradients, activations
3. **Use PyTorch tools**: `gradcheck`, hooks, TensorBoard
4. **Systematic approach**: Isolate data/model/optimizer issues

---



### **3.8 Regularization Techniques in PyTorch**

#### *Preventing Overfitting and Improving Generalization*

## **1. Core Regularization Methods**

### **A. Dropout**

Randomly deactivates neurons during training:

```python
nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # 50% deactivation
    nn.Linear(256, 10)
)
```

**Key Points**:

- Only active during `model.train()`
- Disabled during `model.eval()`
- Typical rates: 0.2-0.5 for hidden layers

### **B. Weight Decay (L2 Regularization)**

Penalizes large weights in optimizer:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5  # L2 penalty
)
```

**Math**: Loss = Original Loss + λ∑w²
**Effect**: Encourages smaller, distributed weights

## **2. Advanced Techniques**

### **A. Batch Normalization**

Normalizes layer outputs:

```python
nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),  # Normalize over batch
    nn.ReLU()
)
```

**Benefits**:

- Allows higher learning rates
- Reduces sensitivity to initialization
- Acts as mild regularizer

### **B. Data Augmentation**

Artificially expands training data:

```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # For images
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
```

### **C. Early Stopping**

```python
best_loss = float('inf')
patience = 3
no_improve = 0

for epoch in epochs:
    val_loss = validate(model)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best.pt')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break  # Stop training
```

## **3. PyTorch Implementation Guide**

### **A. Combined Regularization**

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)
```

### **B. Custom Regularization**

```python
def l1_regularization(model, lambda_l1=0.01):
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

loss = criterion(outputs, labels) + l1_regularization(model)
```

## **4. When to Use Which Technique**

| Scenario                  | Recommended Techniques            |
| ------------------------- | --------------------------------- |
| **Small dataset**   | Dropout + Data Augmentation       |
| **Deep networks**   | BatchNorm + Weight Decay          |
| **Computer Vision** | Data Augmentation + Dropout       |
| **Overfitting**     | Increase Dropout + Early Stopping |

## **5. Debugging Regularization**

### **A. Checking Dropout Activation**

```python
print(model.training)  # Should be True during training
```

### **B. Verifying Weight Decay**

```python
for group in optimizer.param_groups:
    print("Weight decay:", group['weight_decay'])
```

### **C. BatchNorm Statistics**

```python
print("BatchNorm running mean:", model.bn1.running_mean)
```

## **Key Takeaways**

1. **Dropout**: Random deactivation → ensemble effect
2. **Weight Decay**: Prevents large weights
3. **BatchNorm**: Stabilizes training + implicit regularization
4. **Combine techniques**: Often more effective than single methods

---
