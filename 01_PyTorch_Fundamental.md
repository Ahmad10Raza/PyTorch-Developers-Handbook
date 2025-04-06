# 1. PyTorch Fundamental

### **Chapter 1: PyTorch Fundamentals**

**Introduction**

PyTorch has emerged as one of the most popular open-source frameworks for deep learning, renowned for its flexibility, dynamic computation graphs, and Python-first philosophy. Developed by Meta (formerly Facebook) and now part of the Linux Foundation, PyTorch powers cutting-edge research and production systems alike—from computer vision to natural language processing.

This chapter lays the groundwork for your PyTorch journey. You’ll start by understanding PyTorch’s core data structure: **tensors**, which are the building blocks of all operations. You’ll learn how to create, manipulate, and optimize tensors, while mastering key concepts like **broadcasting**, **memory management**, and **GPU acceleration**. By the end, you’ll be equipped to:

- Install PyTorch and verify CPU/GPU support.
- Perform tensor operations with NumPy-like syntax.
- Efficiently handle memory and device transfers.
- Apply indexing and broadcasting rules to avoid common pitfalls.

Whether you’re transitioning from NumPy or starting fresh, these fundamentals will serve as the foundation for every neural network you build. Let’s begin!

### **Key Topics Covered**

1. **Introduction to PyTorch**
   - History, use cases, and comparison with TensorFlow.
2. **Installation and Setup**
   - Configuring PyTorch for CPU/GPU environments.
3. **Tensors**
   - Creation methods, math operations, and properties.
4. **Indexing & Slicing**
   - Subsetting tensors (inspired by NumPy).
5. **Broadcasting Rules**
   - How PyTorch handles operations between tensors of different shapes.
6. **Memory Management**
   - Views vs. copies, and GPU-CPU transfers.


### **1.1 Introduction to PyTorch**

#### **What is PyTorch?**

PyTorch is an **open-source deep learning framework** developed by Meta (Facebook) and now maintained by the Linux Foundation. It provides a flexible and intuitive platform for building, training, and deploying machine learning models, particularly neural networks.

Unlike static computation graph frameworks (like early TensorFlow), PyTorch uses a **dynamic computation graph (define-by-run)**, meaning the graph is built on-the-fly as operations are executed. This makes debugging easier and allows for more natural Python-like coding.

#### **Key Features of PyTorch**

| Feature                                                       | Why It Matters                                                     |
| ------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Dynamic Computation Graphs**                          | Enables flexible model architectures (e.g., variable-length RNNs). |
| **Pythonic & NumPy-like Syntax**                        | Easy to learn if you know Python/NumPy.                            |
| **GPU Acceleration (CUDA)**                             | Speeds up deep learning computations significantly.                |
| **Autograd (Automatic Differentiation)**                | Automatically computes gradients for backpropagation.              |
| **TorchScript & ONNX Support**                          | Allows model deployment in production environments.                |
| **Rich Ecosystem (TorchVision, TorchText, TorchAudio)** | Prebuilt tools for CV, NLP, and audio tasks.                       |
| **Distributed Training**                                | Scales training across multiple GPUs/machines.                     |

#### **PyTorch vs. TensorFlow**

| Aspect                        | PyTorch                              | TensorFlow (2.x+)                     |
| ----------------------------- | ------------------------------------ | ------------------------------------- |
| **Graph Type**          | Dynamic (eager execution by default) | Static (but supports eager execution) |
| **Debugging**           | Easier (Python-like execution)       | More complex (historically)           |
| **Deployment**          | TorchScript, ONNX, Flask             | TensorFlow Serving, TFLite            |
| **Research Popularity** | Dominant in academia                 | More common in industry               |
| **Learning Curve**      | More intuitive                       | Steeper initially                     |

**When to Choose PyTorch?**
✔ Research & prototyping
✔ Dynamic architectures (e.g., attention models)
✔ If you prefer Pythonic code

**When to Consider TensorFlow?**
✔ Production pipelines (e.g., Google Cloud TPUs)
✔ Mobile/edge deployment (TFLite)

#### **PyTorch in the Real World**

- **Research**: Used in most AI papers (e.g., GPT, Stable Diffusion).
- **Industry**: Meta, Tesla, and OpenAI rely on PyTorch.
- **Education**: Preferred for teaching due to simplicity.

#### **Code Example: Why PyTorch Feels Like NumPy**

```python
import torch

# Similar to NumPy, but with GPU support
x = torch.rand(3, 3)  # Random tensor
y = torch.ones(3, 3)  
z = x + y  # Element-wise addition (broadcasting supported)

print(z)
```

**Output**:

```
tensor([[1.2314, 1.8421, 1.5732],
        [1.9923, 1.4821, 1.3356],
        [1.6701, 1.3832, 1.9021]])
```



### **1.2 Installation and Setup (CPU/GPU)**

PyTorch can run on both **CPU and GPU (CUDA)** devices. Setting it up correctly ensures optimal performance, especially for deep learning tasks.

## **Step 1: Install PyTorch**

### **Official Installation (via PyTorch.org)**

The easiest way is to use the [official PyTorch installer](https://pytorch.org/get-started/locally/):

1. Go to **https://pytorch.org/get-started/locally/**
2. Select:
   - **OS** (Windows, Linux, macOS)
   - **Package Manager** (`pip` or `conda`)
   - **Python Version**
   - **CUDA Version** (if using GPU)

Example:

```bash
# For CUDA 12.1 (latest stable GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (no GPU)
pip install torch torchvision torchaudio
```

### **Alternative: Conda (Anaconda/Miniconda)**

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## **Step 2: Verify Installation**

Check if PyTorch is installed correctly:

```python
import torch

# Check PyTorch version
print(torch.__version__)  # e.g., "2.3.0"

# Check if CUDA (GPU support) is available
print(torch.cuda.is_available())  # True if GPU is detected
```

✅ **Expected Output:**

```
2.3.0  
True  # (If GPU is available)
```

## **Step 3: GPU Setup (NVIDIA CUDA & cuDNN)**

### **Requirements for GPU Support**

- **NVIDIA GPU** (Check compatibility [here](https://developer.nvidia.com/cuda-gpus))
- **CUDA Toolkit** (Installs GPU drivers)
- **cuDNN** (Optimized deep learning library)

### **Install CUDA & cuDNN**

1. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Example (Linux):
     ```bash
     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
     sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
     sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
     sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
     sudo apt-get update
     sudo apt-get -y install cuda-12.1
     ```
2. **Install cuDNN**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) (requires NVIDIA account)
   - Extract and copy files to CUDA directory:
     ```bash
     tar -xzvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
     sudo cp cuda/include/* /usr/local/cuda/include/
     sudo cp cuda/lib64/* /usr/local/cuda/lib64/
     ```

## **Step 4: Troubleshooting Common Issues**

| Problem                                                   | Solution                                                                                     |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **`torch.cuda.is_available()` returns `False`** | 1. Check GPU compatibility`<br>` 2. Reinstall CUDA/cuDNN `<br>` 3. Update NVIDIA drivers |
| **"CUDA out of memory"**                            | Reduce batch size or use `torch.cuda.empty_cache()`                                        |
| **Slow performance on GPU**                         | Ensure `tensor.to(device)` is used correctly                                               |
| **DLL load failed (Windows)**                       | Install Microsoft Visual C++ Redistributable                                                 |

## **Step 5: Switching Between CPU & GPU**

PyTorch allows seamless switching between devices:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move tensors/models to GPU
x = torch.rand(3, 3).to(device)
model = MyModel().to(device)
```

## **Final Check: GPU Benchmarking**

Test if GPU acceleration works:

```python
import time

# CPU test
start = time.time()
a = torch.rand(10000, 10000)
b = torch.rand(10000, 10000)
c = a @ b  # Matrix multiplication
print(f"CPU Time: {time.time() - start:.2f}s")

# GPU test (if available)
if torch.cuda.is_available():
    start = time.time()
    a = a.to("cuda")
    b = b.to("cuda")
    c = a @ b
    print(f"GPU Time: {time.time() - start:.2f}s")
```

✅ **Expected Output:**

```
CPU Time: 3.45s  
GPU Time: 0.12s  # (Much faster!)
```


### **1.2 Installation and Setup (CPU/GPU)**

PyTorch can run on both **CPU and GPU (CUDA)** devices. Setting it up correctly ensures optimal performance, especially for deep learning tasks.

## **Step 1: Install PyTorch**

### **Official Installation (via PyTorch.org)**

The easiest way is to use the [official PyTorch installer](https://pytorch.org/get-started/locally/):

1. Go to **https://pytorch.org/get-started/locally/**
2. Select:
   - **OS** (Windows, Linux, macOS)
   - **Package Manager** (`pip` or `conda`)
   - **Python Version**
   - **CUDA Version** (if using GPU)

Example:

```bash
# For CUDA 12.1 (latest stable GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (no GPU)
pip install torch torchvision torchaudio
```

### **Alternative: Conda (Anaconda/Miniconda)**

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## **Step 2: Verify Installation**

Check if PyTorch is installed correctly:

```python
import torch

# Check PyTorch version
print(torch.__version__)  # e.g., "2.3.0"

# Check if CUDA (GPU support) is available
print(torch.cuda.is_available())  # True if GPU is detected
```

✅ **Expected Output:**

```
2.3.0  
True  # (If GPU is available)
```

## **Step 3: GPU Setup (NVIDIA CUDA & cuDNN)**

### **Requirements for GPU Support**

- **NVIDIA GPU** (Check compatibility [here](https://developer.nvidia.com/cuda-gpus))
- **CUDA Toolkit** (Installs GPU drivers)
- **cuDNN** (Optimized deep learning library)

### **Install CUDA & cuDNN**

1. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Example (Linux):
     ```bash
     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
     sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
     sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
     sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
     sudo apt-get update
     sudo apt-get -y install cuda-12.1
     ```
2. **Install cuDNN**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) (requires NVIDIA account)
   - Extract and copy files to CUDA directory:
     ```bash
     tar -xzvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
     sudo cp cuda/include/* /usr/local/cuda/include/
     sudo cp cuda/lib64/* /usr/local/cuda/lib64/
     ```

## **Step 4: Troubleshooting Common Issues**

| Problem                                                   | Solution                                                                                     |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **`torch.cuda.is_available()` returns `False`** | 1. Check GPU compatibility`<br>` 2. Reinstall CUDA/cuDNN `<br>` 3. Update NVIDIA drivers |
| **"CUDA out of memory"**                            | Reduce batch size or use `torch.cuda.empty_cache()`                                        |
| **Slow performance on GPU**                         | Ensure `tensor.to(device)` is used correctly                                               |
| **DLL load failed (Windows)**                       | Install Microsoft Visual C++ Redistributable                                                 |

## **Step 5: Switching Between CPU & GPU**

PyTorch allows seamless switching between devices:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move tensors/models to GPU
x = torch.rand(3, 3).to(device)
model = MyModel().to(device)
```

## **Final Check: GPU Benchmarking**

Test if GPU acceleration works:

```python
import time

# CPU test
start = time.time()
a = torch.rand(10000, 10000)
b = torch.rand(10000, 10000)
c = a @ b  # Matrix multiplication
print(f"CPU Time: {time.time() - start:.2f}s")

# GPU test (if available)
if torch.cuda.is_available():
    start = time.time()
    a = a.to("cuda")
    b = b.to("cuda")
    c = a @ b
    print(f"GPU Time: {time.time() - start:.2f}s")
```

✅ **Expected Output:**

```
CPU Time: 3.45s  
GPU Time: 0.12s  # (Much faster!)
```



### **1.4 Tensor Indexing and Slicing**

*(Advanced techniques for accessing and modifying tensor elements)*

PyTorch's indexing syntax is heavily inspired by **NumPy**, offering powerful ways to extract, modify, and manipulate tensor data. Here's a complete breakdown:

## **1. Basic Indexing (Single Elements)**

```python
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

# Get single element (row 1, column 2)
print(tensor[1, 2])  # Output: 6

# Modify element
tensor[0, 1] = 99    # Now: [[1, 99, 3], ...]
```

## **2. Slicing (Sub-Tensors)**

| Syntax                     | Description                   | Example               |
| -------------------------- | ----------------------------- | --------------------- |
| `tensor[start:end]`      | Range of indices              | `tensor[0:2, 1:]`   |
| `tensor[start:end:step]` | With step size                | `tensor[::2, ::-1]` |
| `tensor[...]`            | Ellipsis (auto-complete dims) | `tensor[..., 0]`    |

**Examples:**

```python
x = torch.arange(9).view(3, 3)  # 3x3 tensor

# Get first 2 rows, last 2 columns
print(x[:2, -2:])  # tensor([[1, 2], [4, 5]])

# Reverse every other row
print(x[::2, :])   # tensor([[0, 1, 2], [6, 7, 8]])
```

## **3. Advanced Indexing Techniques**

### **A. Boolean Masking**

```python
mask = x > 4
print(x[mask])  # tensor([5, 6, 7, 8]) - Flattened result

# Modify masked elements
x[x > 4] = 0    # Sets all >4 values to 0
```

### **B. Integer Array Indexing**

```python
rows = torch.tensor([0, 2])
cols = torch.tensor([1, 0])
print(x[rows, cols])  # tensor([1, 6]) - (x[0,1], x[2,0])
```

### **C. Combined Indexing**

```python
# First column where rows > 1
print(x[x[:, 0] > 1, 0])  # tensor([3, 6])
```

## **4. Special Cases**

### **A. `torch.where()` (Conditional Selection)**

```python
y = torch.where(x > 2, x, -1)  # Replace values <=2 with -1
```

### **B. `torch.masked_select()`**

```python
selected = torch.masked_select(x, x > 2)  # Returns 1D tensor
```

### **C. `torch.take()` (Flattened Indexing)**

```python
indices = torch.tensor([2, 5])
print(x.take(indices))  # tensor([2, 5]) - Treats x as 1D
```

## **5. In-Place Modification**

Use `_` suffix to modify tensors directly:

```python
x[1:, 1:] += 10  # Adds 10 to bottom-right 2x2 submatrix
```

## **6. Dimension-Specific Operations**

| Operation                  | Syntax             | Example                          |
| -------------------------- | ------------------ | -------------------------------- |
| **Select along dim** | `torch.select()` | `x.select(1, 2)` (3rd column)  |
| **Narrow**           | `torch.narrow()` | `x.narrow(0, 1, 2)` (rows 1-2) |
| **Index fill**       | `.index_fill_()` | `x.index_fill_(0, idx, value)` |

**Example:**

```python
# Fill specific indices with value
x.index_fill_(1, torch.tensor([0, 2]), -1)  # Columns 0 & 2 set to -1
```

## **7. Practical Applications**

1. **Cropping Images**:

   ```python
   image_tensor = torch.rand(3, 256, 256)  # (C, H, W)
   cropped = image_tensor[:, 50:200, 30:-30]
   ```
2. **Batch Processing**:

   ```python
   batch = torch.rand(32, 10)  # (batch_size, features)
   first_5_samples = batch[:5] 
   ```
3. **Attention Masks (NLP)**:

   ```python
   seq_len = 50
   mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
   ```

## **Common Pitfalls**

1. **Clone Before Modifying Slices**:

   ```python
   slice = x[0:2].clone()  # Avoids modifying original
   ```
2. **Avoid Mixed Index Types**:

   ```python
   # Bad: x[0, [1, 2]] 
   # Good: x[[0, 0], [1, 2]]  # Explicit indices
   ```
3. **Non-Contiguous Slices**:

   ```python
   y = x.t()  # Transpose makes non-contiguous
   z = y.contiguous()  # Ensures memory layout
   ```

## **Cheatsheet Summary**

| Task               | Syntax                                |
| ------------------ | ------------------------------------- |
| Get single element | `tensor[i,j]`                       |
| Slice submatrix    | `tensor[start:end:step, dim]`       |
| Conditional select | `tensor[mask]` or `torch.where()` |
| Fancy indexing     | `tensor[[row_idx], [col_idx]]`      |
| Modify in-place    | `tensor[idx] = value`               |



### **1.5 Broadcasting Rules in PyTorch**

*(How PyTorch automatically expands tensors for operations with mismatched shapes)*

## **1. What is Broadcasting?**

Broadcasting is PyTorch's mechanism to perform **element-wise operations** on tensors with **different shapes** by automatically expanding the smaller tensor to match the larger one **without copying data**.

**Key Idea**:

- Operations like `+`, `-`, `*`, `/` support broadcasting.
- Works from **right to left** (trailing dimensions).

## **2. Broadcasting Rules**

Two tensors are "broadcastable" if:

1. **Rule 1**: Trailing dimensions must match **or be 1**.
2. **Rule 2**: Missing dimensions are treated as size **1**.

PyTorch compares shapes **dimension-wise** starting from the right:

- If dimensions are **equal**, proceed.
- If one dimension is **1**, stretch it to match.
- If **dimension is missing**, pretend it’s **1**.

## **3. Step-by-Step Examples**

### **Example 1: Vector + Scalar**

```python
a = torch.tensor([1, 2, 3])  # Shape: (3)
b = 5                        # Shape: () → (1) → (3) via broadcasting

result = a + b  # tensor([6, 7, 8])
```

**Steps**:

1. `b` (scalar) is treated as `torch.tensor(5, shape=(1,))`.
2. `b` is stretched to `(3,)` to match `a`.

### **Example 2: Matrix + Vector (Common Case)**

```python
a = torch.rand(3, 4)  # Shape: (3, 4)
b = torch.rand(4)      # Shape: (4) → (1, 4) → (3, 4)

result = a + b  # Valid
```

**Steps**:

1. `b`’s shape becomes `(1, 4)` (Rule 2).
2. `b` is stretched along dim=0 to `(3, 4)`.

### **Example 3: Incompatible Shapes**

```python
a = torch.rand(3, 4, 5)
b = torch.rand(3, 5)    # Shape mismatch at dim=1 (4 vs 5)

# This will ERROR:
# result = a + b  
```

**Why?**:

- Dim 1: `4` (a) != `5` (b) and neither is 1 → **Not broadcastable**.

## **4. Explicit Expansion with `expand()` and `unsqueeze()`**

If broadcasting fails, manually reshape tensors:

### **A. `unsqueeze()`: Add Singleton Dimensions**

```python
a = torch.rand(3, 4)
b = torch.rand(4)       # Shape: (4)

# Manually match shapes
b = b.unsqueeze(0)     # Shape: (1, 4)
result = a + b.expand(3, 4)  # Explicit expansion
```

### **B. `expand()`: Repeat Data (No Copy)**

```python
b = torch.tensor([1, 2, 3])  # Shape: (3)
b_expanded = b.expand(2, 3)   # Shape: (2, 3)  
# Repeats rows: [[1, 2, 3], [1, 2, 3]]
```

## **5. Common Use Cases**

### **A. Normalizing a Batch of Images**

```python
images = torch.rand(32, 3, 256, 256)  # (N, C, H, W)
mean = torch.tensor([0.5, 0.5, 0.5])  # Shape: (3)

# Broadcasting: mean becomes (1, 3, 1, 1)
normalized = images - mean.view(1, 3, 1, 1)  
```

### **B. Adding Bias to Linear Layer Output**

```python
output = torch.rand(10, 50)  # (batch_size, features)
bias = torch.rand(50)        # Shape: (50)

# Broadcasting: bias becomes (1, 50) → (10, 50)
output += bias  
```

## **6. Debugging Broadcasting Errors**

**Error Message**:
`RuntimeError: The size of tensor a (4) must match the size of tensor b (5) at dimension 1`

**Solution**:

1. Check shapes with `.shape`.
2. Use `.unsqueeze()` or `.reshape()` to align dimensions.
3. Verify trailing dimensions match (or are 1).

**Example Fix**:

```python
a = torch.rand(3, 4)
b = torch.rand(3, 5)

# Option 1: Reshape b to be broadcastable
b = b[:, :4]  # Slice to match dim=1

# Option 2: Expand via unsqueeze + expand
b = b.unsqueeze(1).expand(3, 4, 5)  # New shape: (3, 4, 5)
```

## **7. Broadcasting Cheatsheet**

| Scenario                           | How to Fix                                    |
| ---------------------------------- | --------------------------------------------- |
| **Scalar + Tensor**          | Automatic (e.g.,`tensor + 5`)               |
| **Vector + Matrix**          | Ensure vector is `(..., 1)` or `(1, ...)` |
| **Mismatched trailing dims** | Slice or pad tensors                          |
| **Missing dimensions**       | Use `unsqueeze()`                           |

**Key Functions**:

- `tensor.unsqueeze(dim)` - Add dimension at `dim`.
- `tensor.expand(new_shape)` - Repeat data to match shape.
- `tensor.view()` or `reshape()` - Align dimensions.

## **8. Performance Considerations**

- Broadcasting is **memory-efficient** (no data copied).
- But **overuse** can lead to confusion. Prefer explicit shapes when possible.

#### **visual diagram**

### **Broadcasting Visualization**

*(PyTorch expands tensors from right to left)*

#### **Example 1: Vector + Scalar**

```python
Tensor A: [1, 2, 3]  # Shape (3)
Scalar B: 5           # Shape () → (1) → (3)
```

**Step-by-Step Expansion**:

```
A: [1, 2, 3]  
B: 5 → [5, 5, 5]  # Stretched to match A's shape
Result: [1+5, 2+5, 3+5] = [6, 7, 8]
```

#### **Example 2: Matrix + Vector**

```python
Tensor A: [[1, 2, 3],   # Shape (2, 3)
           [4, 5, 6]]  
Tensor B: [10, 20, 30]   # Shape (3) → (1, 3) → (2, 3)
```

**Expansion Process**:

```
B: [10, 20, 30]  
   → [[10, 20, 30],     # Dim 0 stretched
      [10, 20, 30]]  
Result: [[1+10, 2+20, 3+30],  
         [4+10, 5+20, 6+30]]
```

#### **Example 3: Incompatible Shapes**

```python
Tensor A: [[1, 2],  # Shape (2, 2)
           [3, 4]]  
Tensor B: [1, 2, 3]  # Shape (3) → ERROR!
```

**Why?**:

```
Mismatch at dim=1: 
A's dim=1 is 2, B's dim=0 is 3 → Not broadcastable!
```

### **Broadcasting Rules Diagram**

```
Step 1: Align shapes right-to-left
        A (4, 3, 2)
        B    (3, 1)   ← Pad missing dims with 1 → (1, 3, 1)

Step 2: Compare dimensions:
        Dim 2: A=2, B=1 → B stretches to 2
        Dim 1: A=3, B=3 → Match
        Dim 0: A=4, B=1 → B stretches to 4

Final Shape: (4, 3, 2)
```

### **When Broadcasting Fails**

```
Tensor A: (3, 4, 5)  
Tensor B:    (2, 5)  

Error: Dim 1 is 4 (A) vs 2 (B) → Neither is 1!
```

### **Key Takeaways**

1. **Right-to-Left**: PyTorch compares shapes from the last dimension backward.
2. **Stretch 1s**: Dimensions of size 1 are copied to match larger tensors.
3. **No Magic**: If dimensions aren’t 1 or equal, broadcasting fails.

**Visual Mnemonic**:
Imagine "expanding" the smaller tensor like a rubber band to fit the larger one’s shape, but only if the dimensions are compatible!



### **1.6 Memory Management in PyTorch**

*(Views vs. Copies, GPU/CPU Transfers, and Optimization Techniques)*

## **1. Views vs. Copies**

PyTorch tensors can share storage (views) or create independent copies (clones).

| Operation                 | Memory Shared? | Syntax                                | Use Case                        |
| ------------------------- | -------------- | ------------------------------------- | ------------------------------- |
| **View**            | ✅ Yes         | `.view()`, `.reshape()`, `.t()` | Fast shape changes              |
| **Shallow Copy**    | ✅ Yes         | `a = b`                             | Reference assignment            |
| **Clone**           | ❌ No          | `.clone()`                          | Isolate tensor for modification |
| **Contiguous Copy** | ❌ No          | `.contiguous()`                     | Fix non-contiguous layouts      |

**Example**:

```python
a = torch.tensor([[1, 2], [3, 4]])

# View (shares memory)
b = a.view(4)       # b is [1, 2, 3, 4]
a[0, 0] = 99       # b also becomes [99, 2, 3, 4]

# Clone (independent copy)
c = a.clone()       # c gets new memory
a[0, 0] = 0        # c remains unchanged
```

## **2. Checking Memory Properties**

| Method                    | Purpose              | Example                                              |
| ------------------------- | -------------------- | ---------------------------------------------------- |
| `.is_contiguous()`      | Checks memory layout | `a.is_contiguous()`                                |
| `.storage().data_ptr()` | Memory address       | `a.storage().data_ptr() == b.storage().data_ptr()` |
| `id()`                  | Python object ID     | `id(a) == id(b)`                                   |

**Example**:

```python
a = torch.rand(3, 3)
b = a.t()  # Transpose (non-contiguous)

print(a.is_contiguous())  # True  
print(b.is_contiguous())  # False  
```

## **3. GPU/CPU Transfers**

| Operation             | Syntax            | Notes                                 |
| --------------------- | ----------------- | ------------------------------------- |
| **Move to GPU** | `.to('cuda')`   | Requires CUDA GPU                     |
| **Move to CPU** | `.to('cpu')`    | Default device                        |
| **Pin Memory**  | `.pin_memory()` | Faster GPU transfers (for DataLoader) |

**Example**:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transfer tensor
x = torch.rand(3, 3).to(device)  

# Pin memory (for DataLoader)
y = torch.rand(100, 3).pin_memory()  # Used with CUDA
```

## **4. Memory Optimization Techniques**

### **A. Avoid Unnecessary Copies**

```python
# Bad: Creates temporary copy
x = x.contiguous().view(-1)  

# Good: Use reshape() which handles non-contiguous tensors
x = x.reshape(-1)  
```

### **B. Reuse Memory with `torch.no_grad()`**

```python
with torch.no_grad():  # Disables gradient tracking
    y = model(x)       # Reduces memory overhead
```

### **C. Gradient Clearing**

```python
optimizer.zero_grad()  # Clear old gradients to free memory
```

### **D. Mixed Precision Training**

```python
scaler = torch.cuda.amp.GradScaler()  # Reduces GPU memory usage
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

## **5. Common Pitfalls & Fixes**

| Issue                                   | Solution                                  |
| --------------------------------------- | ----------------------------------------- |
| **"CUDA out of memory"**          | Reduce batch size, use `.empty_cache()` |
| **Non-contiguous errors**         | Call `.contiguous()` before operations  |
| **Accidental view modifications** | Use `.clone()` for independent tensors  |
| **Slow DataLoader**               | Enable `pin_memory=True`                |

**Example (Free GPU Memory)**:

```python
torch.cuda.empty_cache()  # Releases unused GPU memory
```

## **6. Visual Guide: Memory Sharing**

```
Original Tensor (a): [1, 2, 3, 4]  
    │
    ├─ View (b = a.view(2, 2)): Shares memory  
    │   └─ Modifying b affects a  
    │
    └─ Clone (c = a.clone()): New memory  
        └─ Independent of a
```

## **7. Summary Cheatsheet**

| Task                         | Code                                      |
| ---------------------------- | ----------------------------------------- |
| **Create view**        | `b = a.view(...)` or `a.reshape(...)` |
| **Force copy**         | `b = a.clone()`                         |
| **Fix non-contiguous** | `a = a.contiguous()`                    |
| **Move to GPU**        | `a = a.to('cuda')`                      |
| **Free GPU memory**    | `torch.cuda.empty_cache()`              |

### **Deep Dive: PyTorch Memory Management Mastery**

*(Advanced optimization techniques, low-level control, and real-world GPU/CPU workflows)*

#### **1. Memory Layout: Strides & Contiguity**

PyTorch tensors use **strides** to map indices to memory locations. Understanding this is key for performance.

```python
x = torch.rand(3, 4)  
print(x.stride())  # (4, 1) ← Elements in dim=0 are 4 apart in memory

# Transpose breaks contiguity
y = x.t()         # Stride becomes (1, 4)
print(y.is_contiguous())  # False

# Fix with contiguous() (copies memory)
z = y.contiguous()  # Stride (3, 1)
```

**When to care?**

- Before operations like `view()` that require contiguous tensors
- When passing data to C/CUDA extensions

#### **2. Advanced View Operations**

Beyond `view()`/`reshape()`, PyTorch offers precise control:

```python
x = torch.rand(2, 3, 4)

# Unfold (sliding window)
unfolded = x.unfold(1, 2, 1)  # Dim=1, size=2, step=1 → shape (2, 2, 4, 2)

# As_strided (manual stride control)
custom_view = x.as_strided(size=(2, 6), stride=(12, 2))  # Risky but powerful
```

**Use Case**: Implementing custom kernels without memory copies.

#### **3. Zero-Copy Transfers (Advanced GPU Optimization)**

##### **A. Pinned Memory for Async GPU Transfers**

```python
# Allocate pinned (page-locked) CPU memory
pinned_tensor = torch.rand(1000, 1000).pin_memory()

# Async transfer to GPU (faster for DataLoader)
with torch.cuda.stream(torch.cuda.Stream()):
    gpu_tensor = pinned_tensor.to('cuda', non_blocking=True)
```

##### **B. Direct GPU-GPU Copies**

```python
# Between GPUs (multi-GPU setups)
tensor_gpu1 = torch.rand(3, 3, device='cuda:0')
tensor_gpu2 = tensor_gpu1.to('cuda:1')  # Implicit copy

# Peer-to-peer (faster for frequent transfers)
torch.cuda.set_device(0)
tensor_gpu2 = tensor_gpu1.cuda(1)  # Uses NVLink if available
```

#### **4. Memory Profiling & Debugging**

##### **A. Tracking Allocations**

```python
# Enable memory snapshot
torch.cuda.memory._record_memory_history()

# Your code here
x = torch.rand(10000, 10000, device='cuda')

# Get snapshot
print(torch.cuda.memory._snapshot())  # Shows allocations/frees
```

##### **B. Finding Memory Leaks**

```python
# Wrap code in memory checker
with torch.autograd.profiler.profile(profile_memory=True) as prof:
    train_model()

print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
```

#### **5. In-Place Operations (Advanced Use)**

Use with caution (breaks autograd if not careful):

```python
x = torch.rand(3, 3, requires_grad=True)

# Safe in-place (PyTorch-managed)
x.add_(1)  # Keeps gradient history

# Unsafe in-place (manual)
x.data += 1  # Bypasses autograd (risk of incorrect gradients)
```

**Golden Rule**: Prefer out-of-place ops unless memory-bound.

#### **6. Custom Memory Allocators**

For extreme optimization (C++ required):

```cpp
#include <torch/cuda/caching_host_allocator.h>

// Register custom allocator
c10::cuda::CUDACachingAllocator::set_allocator_settings("expandable_segments:False");
```

**Use Case**:

- Real-time systems with strict latency requirements
- Avoiding CUDA fragmentation

#### **7. Real-World Optimization Pipeline**

```python
def train_optimized():
    # 1. Pin memory for DataLoader
    loader = DataLoader(dataset, pin_memory=True, num_workers=4)
  
    # 2. Mixed precision
    scaler = torch.cuda.amp.GradScaler()
  
    # 3. Gradient accumulation
    for i, (inputs, targets) in enumerate(loader):
        with torch.cuda.amp.autocast():
            outputs = model(inputs.cuda(non_blocking=True))
            loss = criterion(outputs, targets.cuda(non_blocking=True))
      
        # 4. Overlap compute/data transfer
        next_inputs, next_targets = prefetch_next_batch(loader)
      
        # 5. Memory-efficient backward
        scaler.scale(loss).backward()
        if i % 2 == 0:  # Accumulate gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # Saves memory
```

### **Pro-Level Cheatsheet**

| Technique                        | When to Use             | Code Snippet                          |
| -------------------------------- | ----------------------- | ------------------------------------- |
| **Pinned Memory**          | High-speed data loading | `tensor.pin_memory()`               |
| **Non-blocking Xfer**      | Hide PCIe latency       | `.to(device, non_blocking=True)`    |
| **Gradient Checkpointing** | Memory-heavy models     | `torch.utils.checkpoint.checkpoint` |
| **TensorRT Integration**   | Production deployment   | `torch2trt`                         |
| **Custom CUDA Kernels**    | Ultimate performance    | `torch.cuda.jit.script`             |

### **Debugging Memory Issues**

1. **OOM Errors**:
   ```python
   torch.cuda.empty_cache()  # Temporary fix
   ```
2. **Fragmentation**:
   ```python
   torch.cuda.reset_peak_memory_stats()  # Track true usage
   ```
3. **Memory Snapshot**:
   ```python
   from torch.cuda import memory_summary
   print(memory_summary())
   ```
