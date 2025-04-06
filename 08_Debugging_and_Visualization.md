# Debugging and Visualization


# **Introduction to Chapter 8: Debugging and Visualization**

Training deep learning models can often feel like navigating a maze—**hidden errors, mysterious gradient explosions, and silent failures** can derail your progress. In this chapter, we explore essential **debugging techniques and visualization tools** to diagnose and fix common issues in PyTorch.

### **What You’ll Learn**

1. **TensorBoard Integration** – Visualize training metrics, model graphs, and embeddings.
2. **Gradient Checking** – Verify numerical stability in custom layers.
3. **Common Errors** – Fix shape mismatches, NaN gradients, and CUDA memory issues.
4. **Debugging Tools** – Use `torch.autograd.detect_anomaly` to catch faulty backpropagation.

### **Why This Matters**

- **Spot errors early** before they cascade into training failures.
- **Understand model behavior** with real-time visual feedback.
- **Save hours of frustration** by mastering systematic debugging.

By the end of this chapter, you’ll be equipped to **troubleshoot like a pro** and keep your training runs on track. Let’s dive in!

---



# **TensorBoard Integration in PyTorch**

*(Section 8.1 of Chapter 8: Debugging and Visualization)*

TensorBoard is a powerful **visualization toolkit** from TensorFlow that PyTorch seamlessly supports. It helps you:
✅ **Track metrics** (loss, accuracy) in real time
✅ **Visualize model architectures**
✅ **Profile training performance**
✅ **Debug gradients and embeddings**

## **1. Setup & Installation**

```bash
pip install tensorboard
```

Launch TensorBoard (run in terminal):

```bash
tensorboard --logdir=./runs  # Default log directory
```

Access at `http://localhost:6006`.

## **2. Core Features with PyTorch**

### **A. Logging Scalars (Loss, Accuracy)**

```python
from torch.utils.tensorboard import SummaryWriter

# Initialize writer
writer = SummaryWriter("./runs/experiment_1")

for epoch in range(epochs):
    loss = train_one_epoch(model, dataloader)
    writer.add_scalar("Loss/train", loss, epoch)  # Log loss

writer.close()  # Always close!
```

![Scalar Dashboard](https://pytorch.org/docs/stable/_images/tensorboard_scalars.png)

### **B. Visualizing Models**

```python
# Log model architecture
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Example input shape
writer.add_graph(model, dummy_input)
```

![Model Graph](https://pytorch.org/docs/stable/_images/tensorboard_graph.png)

### **C. Histograms (Weights/Gradients)**

```python
for name, param in model.named_parameters():
    writer.add_histogram(f"weights/{name}", param, epoch)
    writer.add_histogram(f"grads/{name}", param.grad, epoch)
```

![Histograms](https://pytorch.org/docs/stable/_images/tensorboard_histograms.png)

### **D. Embeddings (Dimensionality Reduction)**

```python
# Log feature embeddings (e.g., for MNIST)
images, labels = next(iter(dataloader))
features = model.encoder(images.to(device))
writer.add_embedding(features, metadata=labels, label_img=images)
```

![Embeddings](https://pytorch.org/docs/stable/_images/tensorboard_embedding.png)

## **3. Advanced Features**

### **A. Hyperparameter Tuning**

```python
writer.add_hparams(
    {"lr": 0.01, "batch_size": 64},
    {"hparam/accuracy": 0.92, "hparam/loss": 0.03}
)
```

### **B. Profiling Integration**

```python
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs/profiler"),
    record_shapes=True
) as prof:
    for step, data in enumerate(dataloader):
        train_step(data)
        prof.step()
```

![Profiler](https://pytorch.org/tutorials/_images/profiler_overview.png)

## **4. Best Practices**

1. **Organize Logs**: Use subdirectories like `./runs/exp1_lr0.01`.
2. **Tag Consistently**: Prefix related metrics (e.g., `Loss/train`, `Loss/val`).
3. **Avoid Overlogging**: Log every 100 steps, not every batch.
4. **Compare Experiments**: Overlay plots by selecting multiple runs.

## **5. Full Training Loop Example**

```python
writer = SummaryWriter("./runs/mnist_experiment")

for epoch in range(epochs):
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
      
        train_loss += loss.item()
      
        # Log every 100 batches
        if batch_idx % 100 == 0:
            writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(train_loader) + batch_idx)
  
    # Log epoch-level metrics
    writer.add_scalar("Loss/train_epoch", train_loss / len(train_loader), epoch)
    writer.add_scalar("Accuracy/train", compute_accuracy(model, train_loader), epoch)

writer.close()
```

## **Key Takeaways**

- Use `SummaryWriter` to **log training dynamics**.
- Visualize **model architectures** and **weight distributions**.
- Profile **performance bottlenecks** directly in TensorBoard.
- Compare **multiple experiments** for hyperparameter tuning.

---



# **8.2 Gradient Checking in PyTorch**

*(Next Section in Chapter 8: Debugging and Visualization)*

Gradient checking is a **numerical validation technique** to verify that your manually implemented gradients (e.g., in custom layers or loss functions) are correct. This prevents subtle bugs that can silently degrade model performance.

## **1. Why Gradient Checking Matters**

- Catches errors in **custom autograd Functions** (`torch.autograd.Function`)
- Validates **numerical stability** of complex operations
- Essential when:
  - Implementing **new research layers**
  - Debugging **NaN gradients** or training divergence
  - Porting models from other frameworks

## **2. How Gradient Checking Works**

Compare **analytical gradients** (from `.backward()`) with **numerical gradients** (finite differences):

| Method                            | Pros                     | Cons                         |
| --------------------------------- | ------------------------ | ---------------------------- |
| **Analytical** (Autograd)   | Exact                    | May have implementation bugs |
| **Numerical** (Finite Diff) | No implementation needed | Approximate, slow            |

## **3. Implementation Steps**

### **A. Define a Test Function**

```python
import torch

def gradcheck(layer, test_input, tol=1e-6):
    """Numerically validate gradients of a layer."""
    analytical_grad = torch.autograd.gradcheck(
        layer, 
        test_input, 
        eps=1e-6,       # Small perturbation
        atol=tol,       # Absolute tolerance
        rtol=tol,       # Relative tolerance
        raise_exception=False
    )
    return analytical_grad
```

### **B. Test a Custom Layer**

```python
class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)
        return x @ weight.t() + bias

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_output @ weight
        grad_weight = grad_output.t() @ x
        grad_bias = grad_output.sum(0)
        return grad_x, grad_weight, grad_bias

# Test
x = torch.randn(10, 5, requires_grad=True, dtype=torch.double)
weight = torch.randn(3, 5, requires_grad=True, dtype=torch.double)
print("Gradient check:", gradcheck(CustomLinear.apply, (x, weight, None)))
```

**Output**:
`Gradient check: True` (if correct) or `False` (if buggy).

## **4. Key Parameters**

| Parameter           | Purpose           | Recommended Value    |
| ------------------- | ----------------- | -------------------- |
| `eps`             | Perturbation size | `1e-6` to `1e-8` |
| `atol`/`rtol`   | Error tolerance   | `1e-5` for float32 |
| `raise_exception` | Fail fast         | `True` in debug    |

## **5. When Gradient Checking Fails**

1. **Check Implementation**:
   - Verify **forward pass** math matches **backward pass**.
   - Ensure all tensors are `requires_grad=True`.
2. **Adjust Tolerance**:
   - Increase `atol/rtol` for float32 (due to numerical precision).
3. **Isolate Components**:
   - Test sub-operations individually.

## **6. Real-World Example: Debugging a Buggy Layer**

```python
class BuggyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x > 0).float()  # Bug: Forgot to clone()

# Test - Will FAIL!
x = torch.randn(10, dtype=torch.double, requires_grad=True)
print(gradcheck(BuggyReLU.apply, (x,)))  # False
```

**Fix**:

```python
return grad_output.clone() * (x > 0).float()  # Corrected
```

## **7. Limitations**

- **Slow**: Not for routine training (use only during development).
- **Precision Issues**: May fail for float32 due to numerical noise.
- **Non-Differentiable Ops**: Breaks on `argmax`, `sort`, etc.

## **8. Best Practices**

1. **Use Double Precision** (`dtype=torch.double`) for reliable checks.
2. **Test Small Inputs** (e.g., 5x5 tensors) first.
3. **Combine with `detect_anomaly`** (Section 8.4) for full debugging.

### **Full Gradient Check Pipeline**

```python
def test_custom_layer():
    # 1. Create test input (double precision!)
    x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
  
    # 2. Run gradient check
    passed = torch.autograd.gradcheck(
        CustomLayer.apply, 
        (x,), 
        eps=1e-6, 
        atol=1e-4
    )
  
    # 3. Validate
    assert passed, "Gradient check failed! Review backward() implementation."

test_custom_layer()
```

## **Key Takeaways**

✅ Always validate **custom autograd code** with gradient checks.
✅ Use **double precision** and **small inputs** for reliability.
✅ Combine with **TensorBoard histograms** (Section 8.1) for full gradient debugging.


---



# **8.3 Common Errors in PyTorch**

*(Next Section in Chapter 8: Debugging and Visualization)*

Even experienced PyTorch developers encounter these frustrating errors. Here’s how to diagnose and fix them systematically:

## **1. Shape Mismatch Errors**

### **Classic Symptoms**

```
RuntimeError: Expected size [N, C, H, W], got [N, H, W, C]  # Channel-last vs channel-first
RuntimeError: mat1 and mat2 shapes cannot be multiplied (a×b vs c×d)
```

### **Debugging Steps**

1. **Print Tensor Shapes**
   ```python
   print(f"Input shape: {x.shape}, Weight shape: {layer.weight.shape}")
   ```
2. **Use Shape Assertions**
   ```python
   assert x.dim() == 4, f"Expected 4D tensor, got {x.dim()}D"
   ```
3. **Fix Common Cases**
   - **CNN Inputs**: Add channel dimension with `unsqueeze(1)`
   - **Matrix Multiplication**: Transpose with `.t()` or `permute()`

### **Example Fix**

```python
# Before (crashes)
x = torch.randn(32, 28, 28)  # Missing channel
conv = nn.Conv2d(1, 32, 3)
out = conv(x)  # Error!

# After (fixed)
x = x.unsqueeze(1)  # Shape: [32, 1, 28, 28]
out = conv(x)  # Works!
```

## **2. NaN Gradients**

### **Root Causes**

- **Exploding Gradients** (→ Add gradient clipping)
- **Numerical Instability** (→ Use `torch.autograd.detect_anomaly`)
- **Buggy Custom Layers** (→ Use gradient checking from 8.2)

### **Debugging Tools**

```python
# 1. Enable anomaly detection
with torch.autograd.detect_anomaly():
    loss.backward()  # Will show exact op where NaN occurs

# 2. Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. NaN Sniffer
def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}!")
```

## **3. CUDA Memory Errors**

### **Error Messages**

```
CUDA out of memory: Tried to allocate... (OutOfMemoryError)  
Misaligned address in memcpy... (Often due to CPU-GPU tensor mixing)
```

### **Solutions**

| Problem                 | Fix                                                        |
| ----------------------- | ---------------------------------------------------------- |
| **OOM**           | Reduce batch size or use gradient accumulation             |
| **Memory Leak**   | Call `torch.cuda.empty_cache()`                          |
| **Device Mixing** | Ensure all tensors are on same device with `.to(device)` |

### **Memory Debugging Script**

```python
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")  
print(f"Cached:    {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

## **4. Autograd Graph Issues**

### **Common Pitfalls**

- **Accidental Detach**: `x.detach()` breaks gradient flow
- **In-Place Modification**: `x[0] = 1` corrupts gradients
- **Non-Leaf Updates**: Modifying `weight.data` without `requires_grad_()`

### **Detection Pattern**

```python
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}! Graph is broken.")
```

## **5. Dataloader Deadlocks**

### **Symptoms**

- Training hangs at epoch start
- Multi-process dataloader crashes

### **Fixes**

```python
DataLoader(
    dataset,
    num_workers=4,         # Optimal: 4-8 per GPU
    pin_memory=True,       # Faster GPU transfers
    persistent_workers=True  # Avoid reloading workers
)
```

## **6. Debugging Checklist**

When your model fails:

1. **Shape Sanity Check**: Print tensor shapes at key points
2. **Gradient Inspection**: Log `param.grad` statistics
3. **Numerical Checks**: Add `assert not torch.isnan(x).any()`
4. **Minimal Test Case**: Reproduce with batch_size=1

### **Full Error Handling Example**

```python
try:
    output = model(inputs)
    loss = criterion(output, targets)
  
    # Check for NaN before backward
    assert not torch.isnan(loss).any(), "Loss is NaN!"
  
    loss.backward()
  
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  
    optimizer.step()
  
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("🔴 OOM Error: Reduce batch size")
    elif "shape" in str(e):
        print("🔴 Shape mismatch:", inputs.shape)
    raise
```

## **Key Takeaways**

✅ **Shape Errors**: Always verify tensor dimensions in model hooks
✅ **NaN Gradients**: Use `detect_anomaly` + gradient clipping
✅ **Memory Issues**: Profile with `torch.cuda.memory_summary()`
✅ **Prevention**: Add runtime assertions in critical code paths


---



# **8.4 Debugging Tools in PyTorch**

*(Final Section in Chapter 8: Debugging and Visualization)*

When your model fails silently or produces mysterious results, PyTorch provides **powerful debugging tools** to uncover hidden issues. This section covers professional-grade techniques used in research and production.

## **1. `torch.autograd.detect_anomaly` (Critical for NaN/Inf Detection)**

### **What It Catches**

- NaN/inf values in gradients
- Broken computation graphs
- Illegal in-place modifications

### **How to Use**

```python
with torch.autograd.detect_anomaly():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()  # Will throw error at FIRST occurrence of anomaly
```

**Output Example**:

```
RuntimeError: Function 'AddBackward0' returned nan values in its 0th output.
```

### **Advanced Usage**

```python
# Customize detection (PyTorch ≥2.0)
with torch.autograd.detect_anomaly(
    check_nan=True,       # Default: True
    check_inf=True,       # Check infinite values
    stacklevel=4          # Show more context in traceback
):
    training_step()
```

## **2. CUDA Debugging Tools**

### **A. Synchronous Execution (Catch Silent CUDA Errors)**

```python
torch.cuda.set_sync_debug_mode(1)  # 0=off, 1=warn, 2=error
# Now all CUDA ops will validate sync immediately
```

### **B. Memory Snapshot (For OOM Crashes)**

```python
from torch.cuda.memory import snapshot
import pickle

# Take snapshot before crash
snap = snapshot()
with open('mem_snapshot.pkl', 'wb') as f:
    pickle.dump(snap, f)

# Analyze later
print(snap.stats())  # Shows allocation hotspots
```

## **3. Debugging Distributed Training**

### **A. NCCL Async Error Handling**

```python
import os
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Crash on first collective error
```

### **B. Process Group Inspector**

```python
from torch.distributed import Watchdog

# Monitor stuck processes
watchdog = Watchdog(timeout=timedelta(minutes=5))
watchdog.start()

# Your training loop
try:
    train()
finally:
    watchdog.stop()
```

## **4. Advanced Debugging Techniques**

### **A. Computation Graph Inspection**

```python
def graph_hook(grad):
    print(f"Gradient shape: {grad.shape}, mean: {grad.mean().item()}")
    return grad

x = torch.randn(3, requires_grad=True)
y = x * 2
y.register_hook(graph_hook)  # Prints gradient info during backward()
```

### **B. Conditional Breakpoints**

```python
from pdb import set_trace

class DebugLayer(nn.Module):
    def forward(self, x):
        if torch.isnan(x).any():
            set_trace()  # Drop into debugger
        return x
```

## **5. Real-World Debugging Pipeline**

```python
def train():
    # 1. Enable full debugging
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.set_sync_debug_mode(1)
  
    try:
        # 2. Training loop with assertions
        for inputs, targets in dataloader:
            assert not torch.isnan(inputs).any()
          
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
          
            # 3. Gradient inspection
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Warning: {name} has no gradient!")
          
            loss.backward()
            optimizer.step()
          
    except Exception as e:
        # 4. Post-mortem analysis
        print("CUDA memory summary:")
        print(torch.cuda.memory_summary())
        raise
```

## **Key Takeaways**

✅ **Always use `detect_anomaly`** when developing new architectures
✅ **CUDA sync debugging** catches hidden GPU errors
✅ **Memory snapshots** are crucial for OOM diagnosis
✅ **Distributed training** requires special monitoring

**Pro Tip**: Combine these tools with **TensorBoard logging** (Section 8.1) for comprehensive debugging.

### **Debugging Cheat Sheet**

| Tool                      | When to Use                        | Command/Pattern                                 |
| ------------------------- | ---------------------------------- | ----------------------------------------------- |
| `detect_anomaly`        | NaN gradients, training divergence | `with torch.autograd.detect_anomaly():`       |
| CUDA sync debug           | Silent CUDA errors                 | `torch.cuda.set_sync_debug_mode(1)`           |
| Memory snapshot           | OOM crashes                        | `torch.cuda.memory.snapshot()`                |
| NCCL async error handling | Distributed training hangs         | `os.environ['NCCL_ASYNC_ERROR_HANDLING']='1'` |
| Gradient hooks            | Inspect backprop flow              | `tensor.register_hook(print)`                 |

With these tools, you'll transform from **"Why isn't this working?"** to **"Ah-ha, here's the exact problem!"** 🕵️♂️

---
