---
layout: post
title: "Graph Convolutional Networks for Protein Function Prediction: A PyTorch Geometric Tutorial"
date: 2025-11-01
categories: [deep-learning, graph-neural-networks, pytorch]
author: Aisylu Fattakhova, Azalia Alisheva
---

## Introduction

If you're comfortable with PyTorch and deep learning but haven't yet explored **Graph Neural Networks (GNNs)**, this tutorial is for you. We'll build a complete pipeline using **PyTorch Geometric (PyG)** to predict protein functions from their interaction networks—a problem where traditional CNNs and RNNs fall short.

**What makes this different?** Instead of working with images or sequences, we're working with **graphs**—data structures where nodes (proteins) are connected by edges (interactions). This structural information is crucial for understanding biological function, and it's exactly what graph neural networks excel at.

**Prerequisites:**
- ✅ Familiar with PyTorch and deep learning basics
- ✅ Understanding of neural network training loops
- ❓ New to graph neural networks and PyTorch Geometric

> **🚀 Try it yourself in Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/yourrepo/blob/main/notebook.ipynb)

---

## Table of Contents

<div class="toc-buttons">
  <a class="toc-button" href="#understanding-the-problem">Understanding the Problem</a>
  <a class="toc-button" href="#what-makes-graphs-different">What Makes Graphs Different?</a>
  <a class="toc-button" href="#exploring-the-dataset">Exploring the Dataset</a>
  <a class="toc-button" href="#graph-convolutional-networks-the-core-idea">GCN: The Core Idea</a>
  <a class="toc-button" href="#building-our-gcn-architecture">Building the GCN</a>
  <a class="toc-button" href="#training-and-evaluation">Training & Evaluation</a>
  <a class="toc-button" href="#results-and-insights">Results & Insights</a>
  <a class="toc-button" href="#visualizing-learned-representations">Visualizing Representations</a>
</div>

---

## Understanding the Problem

### Problem Formulation

We formulate protein function prediction as a **multi-label node classification** task on a graph:

- **Graph Structure:** \( G = (V, E) \) where:
  - \( V \) = Set of proteins (nodes)
  - \( E \) = Set of physical interactions between proteins (edges)

- **Input:** Node features {::nomarkdown}\( X \in \mathbb{R}^{N \times 50} \){:/} (50 biological descriptors per protein)

- **Output:** Multi-label predictions {::nomarkdown}\( Y \in \{0,1\}^{N \times 121} \){:/} (121 Gene Ontology terms)

### Why Multi-Label Classification?

Unlike standard classification (e.g., "this image is a cat"), a protein can have **multiple functions simultaneously**. For example, a single protein might be involved in:
- Immune response
- Cellular transport
- Signal transduction

This means we need to predict 121 independent binary classifications per protein.

### The Dataset: Protein-Protein Interaction (PPI)

The PPI dataset consists of **24 graphs**, each representing protein interactions in different human tissues:

<div class="stat-cards">
  <div class="stat-card">
    <div class="stat-number">20</div>
    <div class="stat-label">Training Graphs</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">2</div>
    <div class="stat-label">Validation Graphs</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">2</div>
    <div class="stat-label">Test Graphs</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">121</div>
    <div class="stat-label">Function Classes</div>
  </div>
</div>

Each graph contains:
- **Nodes:** Proteins (typically 1000-3000 per graph)
- **Edges:** Physical interactions (typically 10,000-30,000 per graph)
- **Node Features:** 50-dimensional vectors encoding:
  - Positional gene sets
  - Motif gene sets
  - Immunological signatures
- **Labels:** 121-dimensional binary vectors (one per Gene Ontology term)

---

## What Makes Graphs Different?

If you're coming from standard PyTorch, you're used to tensors with fixed shapes like `(batch_size, channels, height, width)` for images. **Graphs break this assumption.**

### Key Differences from Images

1. **Variable Structure:** Each graph has a different number of nodes and edges—you can't batch them like images
2. **Sparse Connectivity:** Most proteins don't interact with most others (sparse adjacency matrix)
3. **No Spatial Locality:** Unlike pixels, nodes have no inherent grid structure

### How PyTorch Geometric Handles This

PyG introduces two key tensors:

```python
# Node feature matrix: [Num_Nodes, Num_Features]
x = torch.tensor([[0.5, 0.2, ...],  # Node 0 features
                  [0.1, 0.9, ...],  # Node 1 features
                  ...])

# Edge connectivity: [2, Num_Edges] 
# First row: source nodes, Second row: destination nodes
edge_index = torch.tensor([[0, 1, 2, ...],  # Source nodes
                           [1, 0, 3, ...]])  # Destination nodes
```

The `edge_index` format is more efficient than a full adjacency matrix for sparse graphs.

### Visualizing a Protein Interaction Network

Let's visualize a protein's neighborhood to understand the graph structure:

![Subgraph Visualization]({{ site.baseurl }}/assets/subgraph_vis.png)

This visualization shows a 2-hop neighborhood around a central protein. Node size and color represent the number of connections (degree)—proteins with more interactions appear larger and brighter. The spring layout algorithm reveals natural clusters of interacting proteins.

---

## Exploring the Dataset

### Loading the PPI Dataset

Let's start by loading the dataset using PyTorch Geometric:

```python
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

# Load PPI Dataset
train_dataset = PPI(root='/tmp/PPI', split='train')
val_dataset = PPI(root='/tmp/PPI', split='val')
test_dataset = PPI(root='/tmp/PPI', split='test')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

print(f"Node Features: {train_dataset.num_features}")  # 50
print(f"Labels (Classes): {train_dataset.num_classes}")  # 121
print(f"Number of nodes in first graph: {train_dataset[0].num_nodes}")  # ~1767
print(f"Number of edges in first graph: {train_dataset[0].num_edges}")  # ~32318
```

**Key observation:** The first training graph has 1,767 proteins with 32,318 interactions—this is a dense interaction network!

### Class Distribution Analysis

Understanding the distribution of protein functions helps us interpret model performance:

![Class Distribution]({{ site.baseurl }}/assets/class_dist.png)

The class distribution shows that protein functions are **highly imbalanced**—some functions are common (appear in many proteins) while others are rare. This is typical in biological datasets and something we need to account for in our evaluation metrics.

---

## Graph Convolutional Networks: The Core Idea

### The Intuition: Message Passing

At its core, a GCN allows each node to **aggregate information from its neighbors**. The idea is based on **homophily**—nodes with similar neighbors tend to be similar themselves.

**Biological intuition:** If a protein's neighbors are all involved in cellular respiration, it's likely that protein is too. If its neighbors are all immune-related, it probably is as well.

### The Mathematics

The Graph Convolutional layer uses this propagation rule:

{::nomarkdown}
\[
H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
\]
{:/}

Where:

<ul>
  <li>\( \tilde{A} = A + I_N \) is the adjacency matrix with <strong>self-loops</strong> (nodes can use their own features)</li>
  <li>\( \tilde{D} \) is the <strong>degree matrix</strong> (normalizes by number of neighbors)</li>
  <li>\( W^{(l)} \) is the <strong>learnable weight matrix</strong> for layer \( l \)</li>
  <li>\( H^{(l)} \) is the <strong>node embeddings</strong> at layer \( l \)</li>
  <li>\( \sigma \) is the <strong>activation function</strong> (typically ReLU)</li>
</ul>

<div class="info-box tip">
  <strong>💡 Normalization Insight:</strong> The normalization \( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} \) ensures that nodes with many neighbors don't dominate the aggregation—it's like averaging contributions.
</div>

### Why Multiple Layers?

- **Layer 1:** Each node sees its immediate neighbors
- **Layer 2:** Each node sees neighbors of neighbors (2-hop)
- **Layer 3:** Each node sees 3-hop neighborhoods
- **Layer 4:** Each node sees 4-hop neighborhoods

<div class="info-box warning">
  <strong>⚠️ Over-smoothing Problem:</strong> Deeper networks allow information to propagate farther, but too many layers can lead to **over-smoothing** (all nodes become similar). We'll experiment with different depths to find the sweet spot.
</div>

---

## Building Our GCN Architecture

### The Model Definition

Here's our GCN implementation in PyTorch Geometric:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        # Input layer: maps 50 features -> 256 hidden units
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Hidden layers: maintain 256 dimensions
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Output layer: maps 256 -> 121 classes
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, return_embeds=False):
        # Pass through all layers except the last
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)  # Graph convolution
            x = F.relu(x)                      # Non-linearity
            x = F.dropout(x, p=0.5, training=self.training)  # Regularization

        # Get embeddings from last hidden layer
        embeddings = x

        # Final classifier layer (no activation - we'll apply sigmoid for binary classification)
        x = self.convs[-1](x, edge_index)

        if return_embeds:
            return x, embeddings
        return x
```

### Key Components Explained

1. **GCNConv:** PyG's graph convolution layer—handles the sparse matrix multiplication efficiently
2. **ReLU Activation:** Introduces non-linearity between layers
3. **Dropout (p=0.5):** Prevents overfitting, especially important with limited training graphs
4. **No activation on output:** We'll use BCEWithLogitsLoss which applies sigmoid internally

### The Loss Function: Binary Cross Entropy

Since this is multi-label classification, we use **Binary Cross Entropy with Logits Loss**:

{::nomarkdown}
\[
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{121}
\left[ y_{ij} \cdot \log(\sigma(\hat{y}_{ij})) + (1 - y_{ij}) \cdot \log(1 - \sigma(\hat{y}_{ij})) \right]
\]
{:/}

Each of the 121 classes is treated as an **independent binary classification problem**. This is different from standard CrossEntropyLoss, which assumes mutually exclusive classes.

```python
criterion = torch.nn.BCEWithLogitsLoss()  # Applies sigmoid internally
```

---

## Training and Evaluation

### Training Loop

Here's our complete training and evaluation function:

```python
from sklearn.metrics import f1_score
import time

def train_eval(model, optimizer, criterion, epochs=50):
    train_loss_hist = []
    val_f1_hist = []

    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_loss_hist.append(avg_loss)

        # --- VALIDATION PHASE ---
        model.eval()
        ys, preds = [], []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                
                # Convert logits to binary predictions (sigmoid > 0.5)
                pred_labels = (torch.sigmoid(out) > 0.5).float()
                ys.append(data.y.cpu())
                preds.append(pred_labels.cpu())

        # Concatenate all predictions
        y_true = torch.cat(ys, dim=0).numpy()
        y_pred = torch.cat(preds, dim=0).numpy()
        
        # Micro-averaged F1 score (standard for multi-label PPI)
        val_f1 = f1_score(y_true, y_pred, average='micro')
        val_f1_hist.append(val_f1)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"Training finished in {duration:.2f} minutes.")

    return train_loss_hist, val_f1_hist
```

### Experiment Setup: Comparing Network Depths

We'll compare 2, 3, and 4-layer architectures to find the optimal depth:

```python
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Experiment configurations
configs = [2, 3, 4]
results = {}
trained_models = {}

for depth in configs:
    print(f"\n=== Experiment: {depth}-Layer GCN ===")
    
    # Initialize model
    model = GCN(
        in_channels=train_dataset.num_features,      # 50
        hidden_channels=256,
        out_channels=train_dataset.num_classes,      # 121
        num_layers=depth
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train and evaluate
    loss_hist, f1_hist = train_eval(model, optimizer, criterion, epochs=50)
    results[f'{depth}-Layer'] = {'loss': loss_hist, 'f1': f1_hist}
    trained_models[f'{depth}-Layer'] = model
```

### Why Micro-Averaged F1 Score?

For multi-label classification, we use **micro-averaged F1** because:
- It treats each label prediction as an individual binary classification
- It's more robust to class imbalance than macro-averaged F1
- It's the standard metric for PPI function prediction tasks

---

## Results and Insights

### Training Curves

Let's examine how different architectures converge:

![Loss Curve]({{ site.baseurl }}/assets/loss_curve.png)

**Key observations:**
- **2-Layer GCN:** Converges smoothly and achieves the lowest training loss
- **3-Layer GCN:** Shows more variance in F1 scores during training
- **4-Layer GCN:** Struggles more—deeper networks can suffer from over-smoothing

### Performance Comparison

![Performance Bar Chart]({{ site.baseurl }}/assets/performance_bar.png)

**Results:**

<div class="results-table-wrapper">
<table class="results-table">
  <thead>
    <tr>
      <th>Architecture</th>
      <th>Micro-F1 Score</th>
      <th>Performance</th>
    </tr>
  </thead>
  <tbody>
    <tr class="best-result">
      <td><strong>2-Layer GCN</strong></td>
      <td><strong>0.549</strong></td>
      <td><span class="badge badge-success">Best</span></td>
    </tr>
    <tr>
      <td><strong>3-Layer GCN</strong></td>
      <td>0.522</td>
      <td><span class="badge badge-info">Good</span></td>
    </tr>
    <tr>
      <td><strong>4-Layer GCN</strong></td>
      <td>0.480</td>
      <td><span class="badge badge-warning">Lower</span></td>
    </tr>
  </tbody>
</table>
</div>

### Why Does a 2-Layer Network Perform Best?

This result aligns with common findings in graph neural networks:

1. **Over-smoothing:** As we add layers, node representations become more similar across the graph, losing discriminative power
2. **Task-specific optimal depth:** For protein function prediction, 2-hop neighborhoods (what a 2-layer GCN captures) provide sufficient context
3. **Small receptive field:** Proteins typically interact with functionally related neighbors, so distant information may add noise rather than signal

**Takeaway:** Deeper isn't always better for GNNs. The optimal depth depends on:
- Graph structure (average path length, clustering)
- Task requirements (local vs. global patterns)
- Dataset size (deeper networks need more data)

---

## Visualizing Learned Representations

### t-SNE Visualization

To understand what our model learned, we can visualize the learned embeddings using t-SNE:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get the best model (2-Layer)
best_model = trained_models['2-Layer']
best_model.eval()

# Extract embeddings from validation set
data = val_dataset[0].to(device)
with torch.no_grad():
    _, embeddings = best_model(data.x, data.edge_index, return_embeds=True)

# Get labels for coloring
labels = data.y.cpu().numpy().argmax(axis=1)  # Most prominent function
embeds_np = embeddings.cpu().numpy()

# Reduce to 2D using t-SNE
print("Running t-SNE reduction...")
tsne = TSNE(n_components=2, random_state=42)
z = tsne.fit_transform(embeds_np[:3000])  # Sample for speed

# Plot
plt.figure(figsize=(10, 10))
plt.scatter(z[:, 0], z[:, 1], c=labels[:3000], cmap='tab20', s=10, alpha=0.6)
plt.title("t-SNE Visualization of Protein Embeddings")
plt.axis('off')
plt.savefig('assets/tsne_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

![t-SNE Plot]({{ site.baseurl }}/assets/tsne_plot.png)

The t-SNE visualization reveals that the GCN learned to **cluster proteins by function**. Proteins with similar functions are embedded closer together in the learned space, even though they may be distant in the original graph structure.

---

## Key Takeaways

1. **Graphs are powerful for relational data:** When entities have meaningful relationships, GNNs can leverage this structure effectively

2. **PyTorch Geometric makes it easy:** PyG handles sparse graph operations efficiently, letting you focus on architecture and experimentation

3. **Depth matters, but not always more:** For graph neural networks, finding the right depth is crucial—too shallow misses context, too deep causes over-smoothing

4. **Multi-label requires different losses:** Use BCEWithLogitsLoss for multi-label problems, not standard CrossEntropyLoss

5. **Visualization helps interpretation:** t-SNE and other dimensionality reduction techniques reveal what your model learned about the data structure

---

## Next Steps

- **Experiment with different architectures:** Try GraphSAGE, GAT (Graph Attention Networks), or GIN (Graph Isomorphism Networks)
- **Hyperparameter tuning:** Optimize learning rate, hidden dimensions, dropout rate
- **Feature engineering:** Try different node features or add edge features
- **Ensemble methods:** Combine predictions from multiple models

---

## References and Resources

- **PyTorch Geometric Documentation:** [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/)
- **GCN Paper:** Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)
- **PPI Dataset:** [PyG Datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.PPI)

---

## Try It Yourself!

> **🚀 Open the complete notebook in Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/yourrepo/blob/main/notebook.ipynb)

The complete code is available in the Google Colab notebook above. Clone it, run the experiments, and modify the architecture to see how different configurations affect performance!

---

**Authors:** Aisylu Fattakhova, Azalia Alisheva  
**Date:** November 2025

<!-- MathJax loader specific to this post -->
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
      ignoreHtmlClass: 'tex2jax_ignore',
      processHtmlClass: 'tex2jax_process'
    }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
