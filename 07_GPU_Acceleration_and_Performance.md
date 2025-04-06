# GPU Acceleration & Performance


### **GPU Acceleration & Performance**

In this chapter, we explore techniques to **accelerate and optimize** PyTorch models using **GPU computing** and advanced performance enhancements. As deep learning models grow in complexity, efficient computation becomes critical. This chapter covers:

1. **Moving Tensors to GPU** – Leverage CUDA-enabled GPUs for faster operations using `to(device)`.
2. **Mixed Precision Training** – Speed up training with reduced memory usage via `torch.cuda.amp`.
3. **Distributed Training** – Scale training across multiple GPUs with `DataParallel` and `DistributedDataParallel`.
4. **Profiling & Optimization** – Identify bottlenecks and optimize code with `torch.profiler`.

By the end of this chapter, you’ll be equipped to **maximize performance** and **reduce training time** for large-scale deep learning tasks.

---



### **Moving Tensors to GPU in PyTorch (`to(device)`)**

PyTorch allows you to accelerate computations by **moving tensors and models to a GPU** (if available). This is crucial for deep learning, as GPUs significantly speed up matrix operations compared to CPUs.

## **1. Checking GPU Availability**

Before moving data to GPU, check if CUDA (NVIDIA GPU support) is available:

```python
import torch

# Check if CUDA (GPU support) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

- If a GPU is available, `device` will be set to `"cuda"`.
- If not, it defaults to `"cpu"`.

## **2. Moving Tensors to GPU**

Use `.to(device)` or `.cuda()` (legacy) to transfer tensors:

### **Method 1: Using `to(device)` (Recommended)**

```python
x = torch.randn(3, 3)  # Create a tensor on CPU by default
x_gpu = x.to(device)   # Move to GPU if available
print(x_gpu.device)    # Output: cuda:0 (if GPU is available)
```

### **Method 2: Using `.cuda()` (Legacy)**

```python
x_gpu = x.cuda()  # Explicitly moves tensor to GPU
```

### **Method 3: Creating Tensors Directly on GPU**

```python
x = torch.randn(3, 3, device=device)  # Creates tensor directly on GPU
```

## **3. Moving Models to GPU**

Neural network models must also be moved to GPU for acceleration:

```python
model = torch.nn.Linear(10, 5).to(device)  # Moves model to GPU
```

### **Example: Training Loop with GPU**

```python
model = MyNeuralNetwork().to(device)  # Assume a custom model
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    inputs, labels = data  # Assume data is loaded
    inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
  
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
```

## **4. Moving Data Back to CPU**

If needed (e.g., for plotting or CPU-based operations), move tensors back:

```python
x_cpu = x_gpu.cpu()  # Moves tensor back to CPU
```

---

## **5. Handling Multiple GPUs**

If multiple GPUs are available, specify which one to use:

```python
device = torch.device("cuda:0")  # Use first GPU
device = torch.device("cuda:1")  # Use second GPU
```

## **Key Notes**

✅ **Always move both model and data to GPU** for training.
✅ **Operations between GPU and CPU tensors will fail** (ensure all tensors are on the same device).
✅ **Use `.to(device)` for cleaner code** (instead of `.cuda()`).
✅ **Check memory usage** (`nvidia-smi` in terminal) to avoid GPU OOM errors.

### **Example: Full GPU Workflow**

```python
import torch

# 1. Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Create model and move to GPU
model = torch.nn.Linear(10, 5).to(device)

# 3. Generate random data and move to GPU
x = torch.randn(100, 10).to(device)
y = model(x)

print(y.device)  # Output: cuda:0
```

By efficiently moving tensors to GPU, you can **dramatically speed up training and inference** in PyTorch! 🚀

---



# **Mixed Precision Training in PyTorch (`torch.cuda.amp`)**

Mixed Precision Training is a technique that **combines 16-bit (half) and 32-bit (float) precision** to speed up training while maintaining model accuracy. It leverages **NVIDIA Tensor Cores** (available in modern GPUs like V100, A100, RTX) to perform faster matrix operations with reduced memory usage.

## **1. Why Use Mixed Precision?**

✅ **Faster Training** – 16-bit operations are ~2-4x faster on Tensor Cores.
✅ **Lower Memory Usage** – Reduces GPU memory consumption, allowing larger batch sizes.
✅ **No Significant Accuracy Loss** – Critical computations remain in 32-bit for stability.

## **2. Key Components**

PyTorch provides **Automatic Mixed Precision (AMP)** via `torch.cuda.amp`:

- **`GradScaler`**: Prevents underflow in 16-bit gradients by dynamically scaling them.
- **`autocast`**: Automatically selects precision (16/32-bit) for operations.

## **3. How Mixed Precision Works**

| Operation      | Precision Used      | Why?                        |
| -------------- | ------------------- | --------------------------- |
| Forward Pass   | 16-bit (FP16)       | Faster computation          |
| Backward Pass  | 16-bit (FP16)       | Faster gradients            |
| Weight Updates | 32-bit (FP32)       | Numerical stability         |
| Loss Scaling   | Scaled FP16 → FP32 | Prevents gradient underflow |

## **4. Implementation in PyTorch**

### **Step 1: Import AMP Modules**

```python
from torch.cuda.amp import autocast, GradScaler
```

### **Step 2: Initialize GradScaler**

```python
scaler = GradScaler()  # Dynamically scales gradients to prevent underflow
```

### **Step 3: Modify Training Loop**

```python
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters())

for inputs, labels in dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
  
    optimizer.zero_grad()
  
    # 1. Forward pass (automatic 16/32-bit selection)
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
  
    # 2. Backward pass (scaled gradients)
    scaler.scale(loss).backward()
  
    # 3. Optimizer step (unscales gradients)
    scaler.step(optimizer)
  
    # 4. Update scaler for next iteration
    scaler.update()
```

## **5. When to Use Mixed Precision**

✔ **NVIDIA GPUs with Tensor Cores** (Volta, Ampere, RTX series).
✔ **Large batch training** (reduces memory usage).
✔ **Training speed bottlenecks**.

## **6. When to Avoid Mixed Precision**

❌ **Non-NVIDIA GPUs** (no Tensor Core support).
❌ **Models extremely sensitive to precision** (some GANs, reinforcement learning).

## **7. Verifying Mixed Precision**

Check if your GPU is being utilized efficiently:

```bash
nvidia-smi  # Look for "fp16" or "Tensor Core" usage
```

## **8. Performance Gains**

| Model/Dataset        | FP32 Time | FP16 Time | Speedup |
| -------------------- | --------- | --------- | ------- |
| ResNet-50 (ImageNet) | 10 hrs    | 3.5 hrs   | ~3x     |
| BERT-Large           | 7 days    | 2 days    | ~3.5x   |

## **9. Common Pitfalls & Fixes**

🚨 **Gradient Underflow** → Increase `GradScaler`'s `init_scale`.
🚨 **NaN Losses** → Reduce batch size or skip AMP for unstable layers.
🚨 **No Speedup** → Ensure Tensor Cores are enabled (`torch.backends.cudnn.benchmark = True`).

### **Full Example: Mixed Precision Training**

```python
from torch.cuda.amp import autocast, GradScaler

model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for epoch in range(epochs):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
      
        optimizer.zero_grad()
      
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
      
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

By using mixed precision, you can **train models faster and with less memory**, making it essential for large-scale deep learning. 🚀


---



# **Distributed Training in PyTorch (`DataParallel`, `DistributedDataParallel`)**

Distributed training allows you to **scale deep learning models across multiple GPUs/nodes**, dramatically reducing training time for large models and datasets. PyTorch provides two main approaches:

## **1. Key Concepts**

| Concept                     | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| **Data Parallelism**  | Splits batches across GPUs (each GPU gets a subset). |
| **Model Parallelism** | Splits model layers across GPUs (for huge models).   |
| **Process Group**     | Manages communication between GPUs/nodes.            |
| **RPC (Remote Call)** | For more complex distributed setups.                 |

PyTorch supports:

- **Single-machine multi-GPU** (`DataParallel`, `DistributedDataParallel`)
- **Multi-machine multi-GPU** (`DistributedDataParallel` + `torch.distributed`)

## **2. `DataParallel` (DP) - Simple but Limited**

### **How it Works**

- **One process controls all GPUs** (master GPU gathers gradients).
- **Batch is split across GPUs** (e.g., batch 1024 → 256 per GPU).
- **Easy to use but inefficient** due to GIL and master GPU bottleneck.

### **Code Example**

```python
model = nn.DataParallel(model)  # Wrap the model
outputs = model(inputs)         # Forward pass (auto-split batch)
loss.backward()                 # Gradients synchronized automatically
```

### **Pros & Cons**

| Pros                        | Cons                          |
| --------------------------- | ----------------------------- |
| Easy to use (1 line change) | Master GPU becomes bottleneck |
| Works out of the box        | Slower due to Python GIL      |
| Good for quick prototyping  | Not scalable across machines  |

## **3. `DistributedDataParallel` (DDP) - Recommended**

### **How it Works**

- **Each GPU runs a separate process** (no Python GIL limitation).
- **Uses NCCL/RPC for fast GPU-GPU communication**.
- **Supports multi-node training** (across servers).

### **Code Example (Single Machine)**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. Initialize process group
dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 2. Move model to GPU and wrap with DDP
model = MyModel().to(rank)  # rank = GPU ID
model = DDP(model, device_ids=[rank])

# 3. Train (each GPU processes a different batch subset)
for inputs, labels in dataloader:
    inputs, labels = inputs.to(rank), labels.to(rank)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
```

### **Key Steps for DDP**

1. **Initialize `ProcessGroup`** (`init_process_group`).
2. **Split data** using `DistributedSampler`.
3. **Wrap model in `DDP`**.
4. **Launch script with `torchrun`**.

### **Launching Training**

```bash
# Single machine, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-machine (requires SSH setup)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 train.py  # Machine 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 train.py  # Machine 2
```

## **4. `DistributedSampler` (For Data Splitting)**

Ensures each GPU gets a unique batch subset:

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

## **5. DDP vs DataParallel**

| Feature                   | `DataParallel`      | `DistributedDataParallel` |
| ------------------------- | --------------------- | --------------------------- |
| **Speed**           | Slower (GIL)          | Faster (multi-process)      |
| **Scalability**     | Single node           | Multi-node support          |
| **GPU Utilization** | Master GPU bottleneck | All GPUs used efficiently   |
| **Ease of Use**     | Very easy             | Requires setup              |
| **Recommended**     | ❌ No                 | ✅ Yes                      |

## **6. Advanced Topics**

### **Gradient Accumulation (for Large Batches)**

```python
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
  
    if (i + 1) % 4 == 0:  # Accumulate 4 batches
        optimizer.step()
        optimizer.zero_grad()
```

### **Model Parallelism (for Huge Models)**

Split layers across GPUs:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 100).to('cuda:0')
        self.layer2 = nn.Linear(100, 10).to('cuda:1')

    def forward(self, x):
        x = self.layer1(x.to('cuda:0'))
        x = self.layer2(x.to('cuda:1'))
        return x
```

### **Multi-Node Training**

- Requires **NCCL backend** and **SSH setup**.
- Uses `init_process_group(backend="nccl", init_method="env://")`.

## **7. Performance Comparison**

| Method                 | Batch 1024 (4 GPUs) | Multi-Node Support |
| ---------------------- | ------------------- | ------------------ |
| **Single GPU**   | 100% time           | ❌ No              |
| **DataParallel** | ~70% time           | ❌ No              |
| **DDP**          | ~30% time           | ✅ Yes             |

## **8. Common Issues & Fixes**

🚨 **Deadlocks** → Ensure all processes synchronize (e.g., same `DistributedSampler` shuffle).
🚨 **GPU Memory Imbalance** → Use `find_unused_parameters=True` in DDP.
🚨 **NCCL Errors** → Set `NCCL_DEBUG=INFO` for debugging.

### **Full DDP Training Script**

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main(rank, world_size):
    # 1. Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
  
    # 2. Create model and move to GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
  
    # 3. Prepare data with DistributedSampler
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
  
    # 4. Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Ensures shuffling
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
```

### **Launch Script**

```bash
torchrun --nproc_per_node=4 train.py
```

## **Conclusion**

- **Use `DDP` for serious training** (faster, scalable).
- **Use `DataParallel` only for quick tests**.
- **Mixed Precision + DDP = Best Performance** 🚀.

---



# **Profiling & Optimization in PyTorch (`torch.profiler`)**

Profiling helps identify **bottlenecks** in your training loop (e.g., slow GPU ops, CPU-GPU sync issues). PyTorch provides tools to analyze and optimize performance.

## **1. Key Profiling Tools**

| Tool                                  | Purpose                                  |
| ------------------------------------- | ---------------------------------------- |
| **`torch.profiler`**          | Official PyTorch profiler (recommended). |
| **`torch.autograd.profiler`** | Legacy profiler (deprecated).            |
| **TensorBoard Plugin**          | Visualize profiling results.             |
| **`cProfile` (Python)**       | CPU-level profiling.                     |

## **2. Using `torch.profiler`**

### **Basic Setup**

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Track CPU/GPU
    record_shapes=True,  # Log tensor shapes
    profile_memory=True,  # Track memory usage
    with_stack=True,  # Include call stack
) as prof:
    # Code to profile (e.g., training step)
    with record_function("forward_pass"):
        outputs = model(inputs)
    loss.backward()
    optimizer.step()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### **Key Metrics**

| Metric                               | Description                 |
| ------------------------------------ | --------------------------- |
| **`cpu_time`**               | Time spent on CPU.          |
| **`cuda_time`**              | Time spent on GPU.          |
| **`self_cuda_memory_usage`** | GPU memory used by op.      |
| **`occurrences`**            | How often an op was called. |

## **3. Analyzing Results**

### **Example Output**

```
--------------------------------------------------------
Name                  Self CPU %   Self CPU   CPU total %  CPU total  CUDA total
--------------------------------------------------------
conv2d                50.0%       1.2ms      50.0%        1.2ms      0.8ms
matmul                30.0%       0.7ms      30.0%        0.7ms      0.5ms
loss_backward         10.0%       0.2ms      10.0%        0.2ms      0.1ms
```

### **Common Bottlenecks**

1. **CPU-GPU Data Transfer** (`to(device)` calls).
2. **Kernel Launch Overhead** (too many small ops).
3. **Inefficient Ops** (e.g., unoptimized `for` loops).

## **4. Optimization Techniques**

### **A. Reduce CPU-GPU Overhead**

- **Prefetch Data**: Use `DataLoader` with `num_workers > 0`.
- **Pin Memory**: Enable `pin_memory=True` for faster transfers.

```python
dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```

### **B. Optimize GPU Ops**

- **Fuse Kernels**: Replace multiple small ops with fused versions.
  ```python
  # Bad: Two separate ops
  x = torch.relu(x)
  x = torch.matmul(x, W)

  # Good: Fused op (if available)
  x = torch.nn.functional.linear(x, W)
  ```
- **Enable CuDNN Benchmarks**:
  ```python
  torch.backends.cudnn.benchmark = True  # Auto-optimizes conv ops
  ```

### **C. Memory Optimization**

- **Gradient Checkpointing**: Trade compute for memory.
  ```python
  from torch.utils.checkpoint import checkpoint
  def forward(x):
      return checkpoint(model.block, x)  # Recompute activations in backward
  ```
- **Use `torch.empty_cache()`** (if memory fragmentation occurs):
  ```python
  torch.cuda.empty_cache()  # Clear unused GPU memory
  ```

## **5. TensorBoard Visualization**

```python
prof.export_chrome_trace("trace.json")  # Export for TensorBoard
```

Then run:

```bash
tensorboard --logdir=./logs  # View in browser
```

![TensorBoard Profiling](https://pytorch.org/tutorials/_images/profiler_overview.png)

## **6. Advanced: Custom Profiling**

### **Time Specific Code Blocks**

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# Code to measure
end.record()
torch.cuda.synchronize()  # Wait for GPU to finish
print(f"Time: {start.elapsed_time(end)} ms")
```

### **Memory Snapshot**

```python
print(torch.cuda.memory_summary())  # Detailed GPU memory usage
```

## **7. Real-World Example**

### **Optimized Training Loop**

```python
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
dataloader = DataLoader(..., num_workers=4, pin_memory=True)

# Warm-up (avoid initial CUDA overhead)
for _ in range(3):
    dummy_input = torch.randn(64, 3, 224, 224).cuda()
    model(dummy_input)

# Profile
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    for inputs, labels in dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---
