# Training Pipeline


### **Chapter 5: Training Pipeline in PyTorch**

#### *From Raw Data to Deployable Models*

## **Why This Chapter Matters**

> "A well-structured training pipeline is the difference between experimental code and production-ready AI."

In this chapter, you'll learn to build **end-to-end training systems** that:

- Reproducibly train models
- Automate hyperparameter tuning
- Monitor and debug training in real-time
- Scale across multiple GPUs

## **Core Topics Overview**

### **5.1 Training Loop Implementation**

- Anatomy of a PyTorch training loop
- Gradient accumulation
- Mixed precision training

### **5.2 Validation & Metrics**

- Creating robust validation splits
- Custom metric implementations
- Early stopping strategies

### **5.3 Hyperparameter Optimization**

- Manual tuning vs automated (Optuna, Ray Tune)
- Learning rate scheduling
- Batch size selection

### **5.4 Distributed Training**

- DataParallel vs DistributedDataParallel
- Multi-GPU training strategies
- Cluster computing with PyTorch Lightning

### **5.5 Debugging & Profiling**

- Identifying bottlenecks
- Memory usage optimization
- Gradient flow analysis

### **5.6 Model Saving & Deployment**

- Exporting to ONNX/TorchScript
- Production-serving considerations

## **Real-World Impact**

1. **Reproducibility**: Versioned pipelines ensure identical results
2. **Efficiency**: Proper batching and distribution **2-5x** training speed
3. **Model Quality**: Systematic validation prevents overfitting

## **What You'll Build**

By the end of this chapter, you'll create:
✅ A **configurable training pipeline** with logging
✅ **Automated hyperparameter sweeps**
✅ A **production-ready export** pipeline


---



### **5.1 Training Loop Implementation in PyTorch**

#### *The Engine of Deep Learning Model Optimization*

## **1. Core Components of a Training Loop**

### **Basic Training Loop Structure**

```python
model.train()  # Set model to training mode
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Step 1: Move data to device (GPU/CPU)
        data, targets = data.to(device), targets.to(device)
      
        # Step 2: Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
      
        # Step 3: Backward pass
        optimizer.zero_grad()  # Clear existing gradients
        loss.backward()        # Compute gradients
      
        # Step 4: Update weights
        optimizer.step()
```

## **2. Key Elements Explained**

### **A. Model Training Mode**

```python
model.train()  # Enables dropout/batch norm
model.eval()   # Disables them for inference
```

### **B. Gradient Accumulation**

For large batches that don't fit in GPU memory:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, targets) in enumerate(train_loader):
    outputs = model(data)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
  
    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### **C. Mixed Precision Training**

2-3x speedup with NVIDIA GPUs:

```python
scaler = torch.cuda.amp.GradScaler()

for data, targets in train_loader:
    optimizer.zero_grad()
  
    with torch.cuda.amp.autocast():
        outputs = model(data)
        loss = criterion(outputs, targets)
  
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## **3. Advanced Techniques**

### **A. Gradient Clipping**

Prevents exploding gradients in RNNs:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # Threshold
)
```

### **B. Learning Rate Scheduling**

Dynamic LR adjustment:

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1  # Multiply LR by 0.1 every 5 epochs
)

for epoch in range(epochs):
    # Training loop...
    scheduler.step()
```

### **C. Custom Loss Weighting**

For imbalanced datasets:

```python
class_weights = torch.tensor([0.1, 0.9]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

## **4. Debugging Tools**

### **A. Gradient Monitoring**

```python
# Check gradient flow
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
print(f"Gradient norm: {total_norm ** 0.5:.4f}")
```

### **B. Nan Detection**

```python
def check_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in {name}")
```

## **5. Complete Training Template**

```python
def train(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
          
            # Forward
            outputs = model(data)
            loss = criterion(outputs, targets)
          
            # Backward
            optimizer.zero_grad()
            loss.backward()
          
            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          
            # Update
            optimizer.step()
          
            # Logging
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
      
        print(f"Epoch {epoch} Avg Loss: {running_loss/len(train_loader):.4f}")
    return model
```

## **Key Takeaways**

1. **Always** call `zero_grad()` before `backward()`
2. Use **mixed precision** for GPU speedups
3. **Gradient clipping** stabilizes RNNs
4. **Monitor gradients** to debug training

---



### **5.2 Validation & Metrics in PyTorch**

#### *Measuring Model Performance Beyond Training Loss*

## **1. Core Validation Pipeline**

### **Basic Validation Loop**

```python
model.eval()  # Disable dropout/batchnorm
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation
    for data, targets in val_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
      
        # Loss calculation
        val_loss += criterion(outputs, targets).item()
      
        # Accuracy calculation
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

val_loss /= len(val_loader)
val_acc = 100 * correct / total
print(f'Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%')
```

## **2. Essential Metrics**

### **Classification Metrics**

| Metric    | Formula                                 | PyTorch Implementation               |
| --------- | --------------------------------------- | ------------------------------------ |
| Accuracy  | (TP+TN)/(TP+TN+FP+FN)                   | `(preds == labels).float().mean()` |
| Precision | TP/(TP+FP)                              | `torchmetrics.Precision()`         |
| Recall    | TP/(TP+FN)                              | `torchmetrics.Recall()`            |
| F1-Score  | 2*(Precision*Recall)/(Precision+Recall) | `torchmetrics.F1Score()`           |

### **Regression Metrics**

| Metric | Formula                       | PyTorch Implementation              |
| ------ | ----------------------------- | ----------------------------------- |
| MAE    | 𝚺                            | ŷ-y                               |
| MSE    | 𝚺(ŷ-y)²/n                 | `torchmetrics.MeanSquaredError()` |
| R²    | 1 - (𝚺(y-ŷ)²/𝚺(y-ȳ)²) | `torchmetrics.R2Score()`          |

## **3. Advanced Validation Techniques**

### **A. Cross-Validation**

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)
  
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)
  
    # Train and validate for this fold
```

### **B. Custom Metric Calculation**

```python
class DiceScore(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        return (2. * intersection + self.smooth) / 
               (preds.sum() + targets.sum() + self.smooth)

dice = DiceScore()
print(f"Dice Score: {dice(predictions, masks):.4f}")
```

### **C. Early Stopping**

```python
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=3)
if early_stopping(val_loss):
    print("Early stopping triggered!")
    break
```

## **4. Monitoring & Visualization**

### **A. TensorBoard Logging**

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
for epoch in range(epochs):
    # Training...
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_histogram('Layer1_weights', model.layer1.weight, epoch)
```

### **B. Confusion Matrix**

```python
from torchmetrics import ConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay

cm = ConfusionMatrix(num_classes=10)
cm.update(preds, targets)
fig, ax = plt.subplots(figsize=(10,10))
ConfusionMatrixDisplay(cm.compute().numpy()).plot(ax=ax)
plt.savefig('confusion_matrix.png')
```

## **5. Best Practices**

1. **Separate Validation Set**: Never use test data for validation
2. **Metric Selection**: Choose metrics aligned with business goals
3. **Statistical Significance**: Run multiple validation cycles
4. **Class Imbalance**: Use weighted metrics for skewed datasets

## **Key Takeaways**

1. Always use `model.eval()` and `torch.no_grad()` during validation
2. Track multiple metrics beyond just accuracy/loss
3. Implement early stopping to prevent overfitting
4. Visualize results with TensorBoard/confusion matrices

---



### **5.3 Hyperparameter Optimization in PyTorch**

#### *Systematically Tuning Your Model for Peak Performance*

## **1. Core Hyperparameters to Optimize**

| Hyperparameter   | Typical Range    | Importance                 |
| ---------------- | ---------------- | -------------------------- |
| Learning Rate    | 1e-5 to 1e-2     | Most critical              |
| Batch Size       | 16 to 1024       | Affects memory/performance |
| Network Depth    | 2 to 100+ layers | Model capacity             |
| Dropout Rate     | 0.0 to 0.5       | Regularization strength    |
| Optimizer Choice | Adam/SGD/RMSprop | Convergence behavior       |

## **2. Manual Tuning Strategies**

### **A. Learning Rate Finder**

```python
from torch_lr_finder import LRFinder

optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
criterion = nn.CrossEntropyLoss()
lr_finder = LRFinder(model, optimizer, criterion)

lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot()  # Identify optimal LR valley
best_lr = lr_finder.suggestion()
```

### **B. Batch Size Selection**

```python
for batch_size in [32, 64, 128, 256]:
    loader = DataLoader(dataset, batch_size=batch_size)
    # Measure GPU memory usage and throughput
```

## **3. Automated Optimization**

### **A. Grid Search**

```python
from sklearn.model_selection import ParameterGrid

params = {
    'lr': [1e-3, 1e-4],
    'batch_size': [32, 64]
}

for config in ParameterGrid(params):
    train_model(**config)
```

### **B. Random Search (More Efficient)**

```python
from scipy.stats import loguniform

for _ in range(20):
    lr = loguniform.rvs(1e-5, 1e-2)
    batch_size = np.random.choice([32, 64, 128])
    train_model(lr=lr, batch_size=batch_size)
```

### **C. Bayesian Optimization (Optuna)**

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
  
    model = build_model(dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  
    return train_and_validate(model, optimizer)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("Best params:", study.best_params)
```

## **4. Advanced Techniques**

### **A. Population-Based Training (PBT)**

```python
# Using Ray Tune
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

scheduler = PopulationBasedTraining(
    time_attr='training_iteration',
    perturbation_interval=5,
    hyperparam_mutations={
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": [32, 64, 128]
    })

tune.run(train_fn, scheduler=scheduler)
```

### **B. Multi-Fidelity Optimization**

```python
# Hyperband early stopping
from ray.tune.schedulers import HyperBandScheduler

hyperband = HyperBandScheduler(
    time_attr='training_iteration',
    max_t=100,
    reduction_factor=3)

tune.run(train_fn, scheduler=hyperband)
```

## **5. Optimization Best Practices**

1. **Start with Learning Rate**: Use LR finder first
2. **Freeze Other Params**: Tune one parameter at a time initially
3. **Track Experiments**: Use tools like Weights & Biases
4. **Compute Budget**: Allocate more trials to important parameters

## **Key Takeaways**

1. **Manual Tuning** works for small parameter spaces
2. **Automated Methods** (Optuna, Ray) scale better
3. **Learning Rate** is the most critical parameter
4. **Parallelization** speeds up the search

---



### **What Comes After Hyperparameter Optimization?**

#### *The Path to Production-Ready Models*

## **1. Immediate Next Steps**

### **5.4 Distributed Training**

*Scaling your optimized model across multiple devices*

- **Multi-GPU Training**: `DataParallel` vs `DistributedDataParallel`
- **Cluster Computing**: PyTorch Lightning/Horovod integration
- **Mixed Precision**: `torch.cuda.amp` for 2-3x speedups

**Key Question**:
*"How can we train faster without sacrificing model quality?"*

### **5.5 Debugging & Profiling**

*Identifying bottlenecks in your training pipeline*

- **GPU Utilization**: `nvtop` / `nvidia-smi` monitoring
- **Memory Profiling**: `torch.cuda.memory_summary()`
- **Gradient Flow**: Visualizing with TensorBoard

## **2. End-to-End Training Pipeline**

| Step                            | Key Technologies           | Outcome             |
| ------------------------------- | -------------------------- | ------------------- |
| **Data Prep**             | `Dataset`/`DataLoader` | Optimized data flow |
| **Training Loop**         | Custom training logic      | Trained model       |
| **Hyperparameter Tuning** | Optuna/Ray Tune            | Optimized config    |
| **Distributed Training**  | DDP/FSDP                   | Faster training     |
| **Debugging**             | Profilers/Metrics          | Stable convergence  |

**Example Accelerated Pipeline**:

```python
# Multi-GPU training with hyperparameter tuning
def train_func(config):
    model = build_model(config)
    trainer = pl.Trainer(
        gpus=4,
        strategy="ddp",
        precision=16
    )
    trainer.fit(model)

tuner = tune.Tuner(
    train_func,
    tune_config=tune.TuneConfig(num_samples=50)
)
results = tuner.fit()
```

---



### **5.5 Debugging & Profiling in PyTorch**

#### *Diagnosing and Fixing Training Pipeline Issues*

## **1. Core Debugging Tools**

### **A. Gradient Inspection**

```python
# Check for vanishing/exploding gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.mean():.4f}, grad_std={param.grad.std():.4f}")
```

### **B. Activation Monitoring**

```python
# Register forward hooks
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.layer1.register_forward_hook(get_activation('layer1'))
```

## **2. Advanced Profiling Techniques**

### **A. PyTorch Profiler**

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True
) as prof:
    for batch in dataloader:
        train_step(batch)
        prof.step()
```

### **B. Memory Usage Analysis**

```python
print(torch.cuda.memory_summary(device=None, abbreviated=False))
```

## **3. Common Issues & Fixes**

| Symptom                        | Diagnosis                        | Solution                                                         |
| ------------------------------ | -------------------------------- | ---------------------------------------------------------------- |
| **NaN Loss**             | Exploding gradients              | Gradient clipping (`nn.utils.clip_grad_norm_`)                 |
| **Zero Accuracy**        | Incorrect loss function          | Verify loss matches task (e.g., CrossEntropy for classification) |
| **GPU Underutilization** | Data loading bottleneck          | Increase `num_workers`, enable `pin_memory`                  |
| **Slow Training**        | Small batch size/inefficient ops | Mixed precision, larger batches                                  |

## **4. Performance Optimization Checklist**

1. **Data Loading**

   - Prefetch with `DataLoader(prefetch_factor=2)`
   - Enable `pin_memory=True` for CUDA
2. **Computation**

   - Use `torch.compile()` (PyTorch 2.0+)
   - Replace Python loops with vectorized ops
3. **GPU Utilization**

   - Monitor with `nvidia-smi -l 1`
   - Ensure >80% GPU usage during training

## **Key Takeaways**

1. **Profile before optimizing** - Find real bottlenecks
2. **Monitor gradients/activations** - Catch issues early
3. **Use PyTorch's built-in tools** - Profiler, memory debugger
4. **Iterative improvement** - Fix one issue at a time

---



### **5.6 Model Saving & Deployment in PyTorch**

#### *From Trained Models to Production Systems*

## **1. Model Saving & Loading**

### **A. Basic Model Checkpoints**

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'model_checkpoint.pth')

# Load
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### **B. Complete Model Archiving (TorchScript)**

```python
# Trace model
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, 'model_scripted.pt')

# Load for inference
model = torch.jit.load('model_scripted.pt')
```

## **2. Deployment Options**

### **A. ONNX Export (Cross-Platform)**

```python
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### **B. Web Deployment (FastAPI)**

```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.jit.load('model_scripted.pt')

@app.post("/predict")
def predict(input_data: list):
    tensor_input = torch.tensor(input_data)
    return {"prediction": model(tensor_input).tolist()}
```

### **C. Mobile Deployment (TorchMobile)**

```python
from torch.utils.mobile_optimizer import optimize_for_mobile

scripted_model = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_model)
optimized_model._save_for_lite_interpreter("model.ptl")
```

## **3. Production Best Practices**

### **A. Model Optimization**

| Technique     | Implementation                           | Impact            |
| ------------- | ---------------------------------------- | ----------------- |
| Quantization  | `torch.quantization.quantize_dynamic`  | 4x size reduction |
| Pruning       | `torch.nn.utils.prune.l1_unstructured` | Faster inference  |
| Kernel Fusion | `torch.jit.freeze`                     | 20-30% speedup    |

### **B. Monitoring**

```python
# Track input/output distributions
def predict(input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        log_stats(input_tensor.mean(), output.argmax())
    return output
```

## **4. Deployment Architecture**

**Recommended Stack**:

```
                  +-----------------+
                  |   Load Balancer |
                  +--------+--------+
                           |
           +---------------+---------------+
           |                               |
+----------+----------+       +-----------+-----------+
|  Model Server       |       |  Model Server        |
|  (TorchServe)       |       |  (FastAPI/Flask)     |
+----------+----------+       +-----------+-----------+
           |                               |
           +---------------+---------------+
                           |
                  +--------+--------+
                  |    Monitoring  |
                  |    (Prometheus)|
                  +----------------+
```

## **Key Takeaways**

1. **Save both weights AND architecture** for reproducibility
2. **Optimize for target platform** (mobile/web/embedded)
3. **Monitor production models** for data drift
4. **Standardize deployment** using containers (Docker)

---
