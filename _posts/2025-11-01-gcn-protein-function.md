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

<div style="text-align: center; margin: 2.5rem 0;">
  <a href="https://colab.research.google.com/drive/1ldaz7PJEzqI_cbFVz23U94Wngvsyll_H?usp=sharing" target="_blank" class="colab-button">
    <span style="font-size: 1.5rem; line-height: 1;">🚀</span>
    <span>Open in Google Colab</span>
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="flex-shrink: 0;">
      <path d="M7 17L17 7M17 7H7M17 7V17" stroke-linecap="round"/>
    </svg>
  </a>
</div>

---

## Table of Contents

<div class="toc-buttons">
  <a class="toc-button" href="#graph-basics">Graph Basics</a>
  <a class="toc-button" href="#pytorch-geometric-basics">Working with PyTorch Geometric</a>
  <a class="toc-button" href="#understanding-the-problem">Understanding the Problem</a>
  <a class="toc-button" href="#exploring-the-dataset">Exploring the Dataset</a>
  <a class="toc-button" href="#graph-convolutional-networks-the-core-idea">GCN: The Core Idea</a>
  <a class="toc-button" href="#building-our-gcn-architecture">Building the GCN</a>
  <a class="toc-button" href="#training-and-evaluation">Training & Evaluation</a>
  <a class="toc-button" href="#results-and-insights">Results & Insights</a>
  <a class="toc-button" href="#visualizing-learned-representations">Visualizing Representations</a>
</div>

## Graph Basics {#graph-basics}

### What is a Graph?

In machine learning, a **graph** is a data structure \( G = (V, E) \) where:

- **Nodes \(V\):** entities (proteins, papers, users, molecules)
- **Edges \(E\):** relationships (interactions, citations, friendships, chemical bonds)
- **Node features \(X\):** a feature vector for each node (e.g., biological descriptors, text embeddings)

Graphs are powerful when **relationships matter** as much as the individual data points—exactly the case for protein–protein interaction networks.

### Common Types of Graphs

- **Undirected graph:** edges have no direction (e.g., mutual friendship, protein interaction)
- **Directed graph:** edges have direction (e.g., citation network: paper A → paper B)
- **Weighted graph:** edges carry strengths or capacities (e.g., interaction confidence scores)
- **Bipartite graph:** two disjoint node sets with edges only across sets (e.g., users ↔ items in recommendation)

### Interactive: Different Graph Types

Explore four tiny graphs below—undirected, directed, weighted, and bipartite. You can **drag nodes** and **inspect different structures** interactively.

<div class="graph-row">
  <div class="graph-box">
    <h4>Undirected Graph</h4>
    <div id="graph-undirected" class="graph-canvas"></div>
  </div>
  <div class="graph-box">
    <h4>Directed Graph</h4>
    <div id="graph-directed" class="graph-canvas"></div>
  </div>
  <div class="graph-box">
    <h4>Weighted Graph</h4>
    <div id="graph-weighted" class="graph-canvas"></div>
  </div>
  <div class="graph-box">
    <h4>Bipartite Graph</h4>
    <div id="graph-bipartite" class="graph-canvas"></div>
  </div>
</div>

<!-- vis-network library (from CDN) -->
<script type="text/javascript" src="https://unpkg.com/vis-network@9.1.6/dist/vis-network.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/vis-network@9.1.6/styles/vis-network.min.css"/>

<script type="text/javascript">
  if (typeof window !== 'undefined') {
    function buildGraph(containerId, nodesData, edgesData, optionsOverrides) {
      const container = document.getElementById(containerId);
      if (!container || typeof vis === 'undefined') return;

      const nodes = new vis.DataSet(nodesData);
      const edges = new vis.DataSet(edgesData);

      const baseOptions = {
        physics: { stabilization: true },
        interaction: {
          zoomView: false,
          dragView: true,
          dragNodes: true,
          selectConnectedEdges: false
        },
        nodes: {
          shape: 'dot',
          size: 18,
          font: { size: 14, color: '#000000' }
        },
        edges: {
          width: 2,
          color: { color: '#667eea' },
          arrows: { to: false }
        }
      };

      const options = Object.assign({}, baseOptions, optionsOverrides || {});
      new vis.Network(container, { nodes, edges }, options);
    }

    // Undirected protein interaction toy graph
    buildGraph('graph-undirected',
      [
        { id: 1, label: 'Protein A' },
        { id: 2, label: 'Protein B' },
        { id: 3, label: 'Protein C' },
        { id: 4, label: 'Protein D' }
      ],
      [
        { from: 1, to: 2 },
        { from: 1, to: 3 },
        { from: 2, to: 4 },
        { from: 3, to: 4 }
      ]
    );

    // Directed citation toy graph
    buildGraph('graph-directed',
      [
        { id: 1, label: 'Paper 1' },
        { id: 2, label: 'Paper 2' },
        { id: 3, label: 'Paper 3' },
        { id: 4, label: 'Paper 4' }
      ],
      [
        { from: 1, to: 2 },
        { from: 1, to: 3 },
        { from: 2, to: 4 },
        { from: 3, to: 4 }
      ],
      { edges: { arrows: { to: { enabled: true, scaleFactor: 0.8 } } } }
    );

    // Weighted toy graph: interaction strengths
    buildGraph('graph-weighted',
      [
        { id: 1, label: 'A' },
        { id: 2, label: 'B' },
        { id: 3, label: 'C' }
      ],
      [
        { from: 1, to: 2, width: 1, label: '0.2' },
        { from: 1, to: 3, width: 4, label: '0.9' },
        { from: 2, to: 3, width: 2, label: '0.5' }
      ],
      {
        edges: {
          font: { size: 12, align: 'top', color: '#000000' }
        }
      }
    );

    // Bipartite toy graph: Proteins ↔ Functions
    buildGraph('graph-bipartite',
      [
        { id: 1, label: 'Protein X', group: 'protein' },
        { id: 2, label: 'Protein Y', group: 'protein' },
        { id: 3, label: 'Immune', group: 'function' },
        { id: 4, label: 'Transport', group: 'function' }
      ],
      [
        { from: 1, to: 3 },
        { from: 1, to: 4 },
        { from: 2, to: 3 }
      ],
      {
        groups: {
          protein: { color: { background: '#e0e7ff', border: '#4c51bf' }, font: { color: '#000000' } },
          function: { color: { background: '#ffedd5', border: '#ea580c' }, font: { color: '#000000' } }
        }
      }
    );
  }
</script>

Now that we understand what graphs are, let's see how they differ from traditional tensor-based data structures in PyTorch and how PyTorch Geometric makes working with them practical.

---

## Working with Graphs in PyTorch Geometric {#pytorch-geometric-basics}

If you're coming from standard PyTorch, you're used to tensors with fixed shapes like `(batch_size, channels, height, width)` for images. **Graphs break this assumption.** Let's explore the key differences and how PyTorch Geometric handles them.

### Key Differences from Images

1. **Variable Structure:** Each graph has a different number of nodes and edges—you can't batch them like images
2. **Sparse Connectivity:** Most proteins don't interact with most others (sparse adjacency matrix)
3. **No Spatial Locality:** Unlike pixels, nodes have no inherent grid structure

### How PyTorch Geometric Handles This

PyG introduces two key tensors that efficiently represent graph-structured data:

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

The `edge_index` format is more efficient than a full adjacency matrix for sparse graphs. Instead of storing an {::nomarkdown}\( N \times N \){:/} matrix where most entries are zero, we only store the edges that actually exist.

Now that we know how graphs work and how to represent them in PyTorch, let's dive into our specific problem: predicting protein functions from their interaction networks.

---

## Understanding the Problem {#understanding-the-problem}

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

Perfect! Now that we've formulated the problem, let's explore the dataset we'll use to solve it.

---

## Exploring the Dataset

### Overview: Protein-Protein Interaction (PPI) Dataset

The PPI dataset consists of **24 graphs**, each representing protein interactions in different human tissues. This is a real-world biological dataset commonly used for benchmarking graph neural networks.

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

### Dataset Structure

Each graph contains:
- **Nodes:** Proteins (typically 1,000-3,000 per graph)
- **Edges:** Physical interactions (typically 10,000-30,000 per graph)
- **Node Features:** 50-dimensional vectors encoding:
  - Positional gene sets
  - Motif gene sets
  - Immunological signatures
- **Labels:** 121-dimensional binary vectors (one per Gene Ontology term)

### Loading the Dataset with PyTorch Geometric

Let's load the dataset using PyTorch Geometric:

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

**Key observation:** The first training graph has 1,767 proteins with 32,318 interactions—this is a **dense interaction network**! Each protein connects to many others, which is biologically meaningful since proteins often work in complexes.

### Visualizing the Graph Structure

Let's visualize a protein's neighborhood to understand the graph structure:

![Subgraph Visualization]({{ site.baseurl }}/assets/subgraph_vis.png)

This visualization shows a 2-hop neighborhood around a central protein. Node size and color represent the number of connections (degree)—proteins with more interactions appear larger and brighter. The spring layout algorithm reveals natural clusters of interacting proteins.

### Class Distribution Analysis

Understanding the distribution of protein functions helps us interpret model performance:

![Class Distribution]({{ site.baseurl }}/assets/class_dist.png)

The class distribution shows that protein functions are **highly imbalanced**—some functions are common (appear in many proteins) while others are rare. This is typical in biological datasets and something we need to account for in our evaluation metrics. We'll use micro-averaged F1 score to handle this imbalance.

Now that we understand our data, let's build a Graph Convolutional Network to learn from these protein interactions and predict functions.

---

## Graph Convolutional Networks: The Core Idea

### The Intuition: Message Passing

At its core, a GCN allows each node to **aggregate information from its neighbors**. The idea is based on **homophily**—nodes with similar neighbors tend to be similar themselves.

**Biological intuition:** If a protein's neighbors are all involved in cellular respiration, it's likely that protein is too. If its neighbors are all immune-related, it probably is as well.

### Interactive: Message Passing in Action

Explore how message passing works in this detailed interactive visualization. Watch how a central protein **aggregates feature information** from its neighboring proteins to create an updated representation.

<div style="display: grid; grid-template-columns: 2fr 1fr; gap: 1rem; margin: 1.5rem 0;">
  <div id="message-passing-demo" style="height: 500px; border-radius: 10px; border: 2px solid #667eea; background: white; position: relative; box-sizing: border-box;"></div>
  
  <div id="msg-status-panel" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 10px; padding: 1.5rem; border: 2px solid #667eea; height: 500px; overflow-y: auto; box-sizing: border-box;">
    <h4 style="margin-top: 0; color: #667eea; font-size: 1.1rem;">📊 Message Passing Steps</h4>
    <div id="msg-status-text" style="font-size: 0.95rem; line-height: 1.6; color: #333;">
      <p><strong>Step 0: Initial State</strong></p>
      <p>Each protein has initial features (biological descriptors). Use "Next Step" to manually progress through the message passing process.</p>
      <hr style="border: none; border-top: 1px solid rgba(102, 126, 234, 0.3); margin: 1rem 0;">
      <div id="msg-formula" style="background: rgba(255, 255, 255, 0.7); padding: 1rem; border-radius: 5px; margin-top: 1rem; display: none;">
        <strong>Formula:</strong>
        <div style="font-size: 0.85rem; margin-top: 0.5rem; font-family: 'JetBrains Mono', monospace;">
          <div>h'<sub>i</sub> = σ(Σ<sub>j∈N(i)</sub> W·h<sub>j</sub> / √(d<sub>i</sub>·d<sub>j</sub>))</div>
        </div>
      </div>
    </div>
  </div>
</div>

<div style="display: flex; gap: 0.5rem; justify-content: center; margin-bottom: 1rem; flex-wrap: wrap;">
  <button id="reset-msg" style="padding: 0.75rem 1.5rem; border-radius: 8px; background: #f093fb; color: white; border: none; cursor: pointer; font-weight: 600; box-shadow: 0 4px 12px rgba(240, 147, 251, 0.4); transition: transform 0.2s;">🔄 Reset</button>
  <button id="step-msg" style="padding: 0.75rem 1.5rem; border-radius: 8px; background: #4facfe; color: white; border: none; cursor: pointer; font-weight: 600; box-shadow: 0 4px 12px rgba(79, 172, 254, 0.4); transition: transform 0.2s;">⏭ Next Step</button>
</div>

<script type="text/javascript">
  if (typeof window !== 'undefined') {
    let msgNetwork = null;
    let msgNodes = null;
    let msgEdges = null;
    let currentStep = 0;
    let isAnimating = false;
    
    // Feature vectors for each protein (simplified 3D for visualization)
    const nodeFeatures = {
      0: [0.5, 0.3, 0.8],  // Central protein
      1: [0.7, 0.2, 0.1],  // Neighbor 1 - Immune function
      2: [0.2, 0.9, 0.3],  // Neighbor 2 - Metabolic function
      3: [0.4, 0.6, 0.5],  // Neighbor 3 - Transport function
      4: [0.8, 0.4, 0.2],  // Neighbor 4 - Signal function
      5: [0.3, 0.7, 0.6]   // Neighbor 5 - Structural function
    };
    
    const nodeFunctions = {
      0: 'Unknown',
      1: 'Immune Response',
      2: 'Metabolism',
      3: 'Transport',
      4: 'Signaling',
      5: 'Structure'
    };

    function updateStatusPanel(step, details) {
      const statusText = document.getElementById('msg-status-text');
      const formulaDiv = document.getElementById('msg-formula');
      
      if (!statusText) {
        console.warn('Status text element not found');
        return;
      }
      
      const steps = {
        0: {
          title: 'Step 0: Initial State',
          content: `
            <p><strong>Each protein has initial features:</strong></p>
            <ul style="font-size: 0.9rem; margin: 0.5rem 0;">
              <li><strong>Protein 0 (Central):</strong> Features: [${nodeFeatures[0].map(f => f.toFixed(2)).join(', ')}]</li>
              <li><strong>Neighbor 1:</strong> Features: [${nodeFeatures[1].map(f => f.toFixed(2)).join(', ')}] - ${nodeFunctions[1]}</li>
              <li><strong>Neighbor 2:</strong> Features: [${nodeFeatures[2].map(f => f.toFixed(2)).join(', ')}] - ${nodeFunctions[2]}</li>
              <li><strong>Neighbor 3:</strong> Features: [${nodeFeatures[3].map(f => f.toFixed(2)).join(', ')}] - ${nodeFunctions[3]}</li>
              <li><strong>Neighbor 4:</strong> Features: [${nodeFeatures[4].map(f => f.toFixed(2)).join(', ')}] - ${nodeFunctions[4]}</li>
              <li><strong>Neighbor 5:</strong> Features: [${nodeFeatures[5].map(f => f.toFixed(2)).join(', ')}] - ${nodeFunctions[5]}</li>
            </ul>
            <p style="margin-top: 1rem;">Use "Next Step" to manually progress through each step of the message passing process.</p>
          `
        },
        1: {
          title: 'Step 1: Identify Neighbors',
          content: `
            <p><strong>Neighbors are identified:</strong></p>
            <p>The central protein (Protein 0) has <strong>5 neighboring proteins</strong> connected by edges in the interaction network.</p>
            <p style="color: #ff9800;">✨ Neighbors are highlighted in orange.</p>
          `
        },
        2: {
          title: 'Step 2: Compute Messages',
          content: `
            <p><strong>Each neighbor prepares a message:</strong></p>
            <p>Messages are computed using the neighbor's features multiplied by learned weights:</p>
            <div style="background: rgba(255, 255, 255, 0.6); padding: 0.75rem; border-radius: 5px; margin: 0.5rem 0; font-size: 0.85rem;">
              message<sub>j</sub> = W · h<sub>j</sub>
            </div>
            <p style="color: #667eea;">💫 Messages flow along edges (highlighted in blue).</p>
          `
        },
        3: {
          title: 'Step 3: Normalize Messages',
          content: `
            <p><strong>Apply symmetric normalization:</strong></p>
            <p>To prevent nodes with many neighbors from dominating, we normalize by degrees:</p>
            <div style="background: rgba(255, 255, 255, 0.6); padding: 0.75rem; border-radius: 5px; margin: 0.5rem 0; font-size: 0.85rem;">
              normalized_msg = message / √(d<sub>central</sub> · d<sub>neighbor</sub>)
            </div>
            <p>Central node degree: 5, Neighbor degree: 1</p>
          `
        },
        4: {
          title: 'Step 4: Aggregate Messages',
          content: `
            <p><strong>Sum all normalized messages:</strong></p>
            <p>The central node aggregates (sums) all incoming messages:</p>
            <div style="background: rgba(255, 255, 255, 0.6); padding: 0.75rem; border-radius: 5px; margin: 0.5rem 0; font-size: 0.85rem;">
              h'<sub>0</sub> = Σ<sub>j=1 to 5</sub> normalized_msg<sub>j</sub>
            </div>
            <p style="color: #4caf50;">✅ Aggregation complete!</p>
          `
        },
        5: {
          title: 'Step 5: Apply Activation',
          content: `
            <p><strong>Apply ReLU activation function:</strong></p>
            <div style="background: rgba(255, 255, 255, 0.6); padding: 0.75rem; border-radius: 5px; margin: 0.5rem 0; font-size: 0.85rem;">
              h'<sub>final</sub> = ReLU(h'<sub>0</sub>)
            </div>
            <p>The central protein now has an <strong>updated representation</strong> that incorporates information from all its neighbors!</p>
            <p style="color: #4caf50; font-weight: bold;">🎉 Message passing complete!</p>
          `
        }
      };
      
      if (steps[step]) {
        if (statusText) {
          statusText.innerHTML = `<p><strong>${steps[step].title}</strong></p>${steps[step].content}`;
        }
        if (formulaDiv) {
          if (step >= 2) {
            formulaDiv.style.display = 'block';
          } else {
            formulaDiv.style.display = 'none';
          }
        }
      }
    }

    function initMessagePassing() {
      const container = document.getElementById('message-passing-demo');
      if (!container || typeof vis === 'undefined') return;

      // Create nodes with feature labels
      msgNodes = new vis.DataSet([
        { 
          id: 0, 
          label: 'P0\n[0.50, 0.30, 0.80]', 
          title: 'Central Protein\nFeatures: [0.50, 0.30, 0.80]',
          color: { background: '#667eea', border: '#4c51bf' }, 
          font: { color: '#000000', size: 12 },
          size: 30
        },
        { 
          id: 1, 
          label: 'P1\n[0.70, 0.20, 0.10]', 
          title: 'Neighbor 1 - Immune Response\nFeatures: [0.70, 0.20, 0.10]',
          color: { background: '#f093fb', border: '#ea580c' }, 
          font: { color: '#000000', size: 11 },
          size: 24
        },
        { 
          id: 2, 
          label: 'P2\n[0.20, 0.90, 0.30]', 
          title: 'Neighbor 2 - Metabolism\nFeatures: [0.20, 0.90, 0.30]',
          color: { background: '#f093fb', border: '#ea580c' }, 
          font: { color: '#000000', size: 11 },
          size: 24
        },
        { 
          id: 3, 
          label: 'P3\n[0.40, 0.60, 0.50]', 
          title: 'Neighbor 3 - Transport\nFeatures: [0.40, 0.60, 0.50]',
          color: { background: '#f093fb', border: '#ea580c' }, 
          font: { color: '#000000', size: 11 },
          size: 24
        },
        { 
          id: 4, 
          label: 'P4\n[0.80, 0.40, 0.20]', 
          title: 'Neighbor 4 - Signaling\nFeatures: [0.80, 0.40, 0.20]',
          color: { background: '#f093fb', border: '#ea580c' }, 
          font: { color: '#000000', size: 11 },
          size: 24
        },
        { 
          id: 5, 
          label: 'P5\n[0.30, 0.70, 0.60]', 
          title: 'Neighbor 5 - Structure\nFeatures: [0.30, 0.70, 0.60]',
          color: { background: '#f093fb', border: '#ea580c' }, 
          font: { color: '#000000', size: 11 },
          size: 24
        }
      ]);

      // Create edges with labels
      msgEdges = new vis.DataSet([
        { id: 'e1', from: 1, to: 0, color: { color: '#888' }, width: 2, smooth: { type: 'continuous' }, label: '', arrows: { to: { enabled: false } } },
        { id: 'e2', from: 2, to: 0, color: { color: '#888' }, width: 2, smooth: { type: 'continuous' }, label: '', arrows: { to: { enabled: false } } },
        { id: 'e3', from: 3, to: 0, color: { color: '#888' }, width: 2, smooth: { type: 'continuous' }, label: '', arrows: { to: { enabled: false } } },
        { id: 'e4', from: 4, to: 0, color: { color: '#888' }, width: 2, smooth: { type: 'continuous' }, label: '', arrows: { to: { enabled: false } } },
        { id: 'e5', from: 5, to: 0, color: { color: '#888' }, width: 2, smooth: { type: 'continuous' }, label: '', arrows: { to: { enabled: false } } }
      ]);

      const data = { nodes: msgNodes, edges: msgEdges };
      const options = {
        physics: {
          stabilization: { iterations: 200 },
          barnesHut: { gravitationalConstant: -30000 }
        },
        interaction: {
          zoomView: false,
          dragView: true,
          dragNodes: true,
          tooltipDelay: 100
        },
        nodes: {
          shape: 'dot',
          font: { color: '#000000' },
          borderWidth: 2
        },
        edges: {
          smooth: { type: 'continuous', roundness: 0.5 },
          font: { color: '#333', size: 10, align: 'middle' },
          labelHighlightBold: false
        }
      };

      msgNetwork = new vis.Network(container, data, options);
      
      // Button handlers
      document.getElementById('reset-msg')?.addEventListener('click', resetMessagePassing);
      document.getElementById('step-msg')?.addEventListener('click', stepMessagePassing);
      
      updateStatusPanel(0);
    }

    function startFullAnimation() {
      if (isAnimating) return;
      isAnimating = true;
      currentStep = 0;
      
      // Step 1: Highlight neighbors
      updateStatusPanel(1);
      [1, 2, 3, 4, 5].forEach((id, idx) => {
        setTimeout(() => {
          msgNodes.update({ id: id, color: { background: '#ff9800', border: '#e65100' } });
        }, idx * 150);
      });
      
      setTimeout(() => {
        // Step 2: Messages flow
        updateStatusPanel(2);
        document.getElementById('msg-formula').style.display = 'block';
        // Update edges by ID directly
        ['e1', 'e2', 'e3', 'e4', 'e5'].forEach((edgeId, idx) => {
          setTimeout(() => {
            msgEdges.update({ 
              id: edgeId, 
              color: { color: '#667eea', highlight: '#667eea' }, 
              width: 4,
              label: 'msg',
              arrows: { to: { enabled: true, scaleFactor: 0.8, type: 'arrow' } }
            });
          }, idx * 200);
        });
        
        setTimeout(() => {
          // Step 3: Normalize
          updateStatusPanel(3);
          
          setTimeout(() => {
            // Step 4: Aggregate
            updateStatusPanel(4);
            msgNodes.update({ 
              id: 0, 
              color: { background: '#9c27b0', border: '#6a1b9a' },
              label: 'P0\nAggregating...',
              font: { color: '#ffffff' }
            });
            
            setTimeout(() => {
              // Step 5: Activation
              updateStatusPanel(5);
              msgNodes.update({ 
                id: 0, 
                color: { background: '#4caf50', border: '#2e7d32' },
                label: 'P0\n[0.48, 0.52, 0.44]',
                title: 'Updated Features\n[0.48, 0.52, 0.44]',
                font: { color: '#000000' }
              });
              
              setTimeout(() => {
                isAnimating = false;
              }, 2000);
            }, 1000);
          }, 800);
        }, 1200);
      }, 1000);
    }

    function stepMessagePassing() {
      if (isAnimating) {
        return;
      }
      
      // Execute current step and advance
      const stepActions = {
        0: () => {
          // Step 1: Highlight neighbors
          updateStatusPanel(1);
          [1, 2, 3, 4, 5].forEach(id => {
            if (msgNodes) {
              msgNodes.update({ id: id, color: { background: '#ff9800', border: '#e65100' } });
            }
          });
          const stepBtn = document.getElementById('step-msg');
          if (stepBtn) stepBtn.textContent = '⏭ Step 2: Messages';
        },
        1: () => {
          // Step 2: Messages flow
          updateStatusPanel(2);
          const formulaDiv = document.getElementById('msg-formula');
          if (formulaDiv) formulaDiv.style.display = 'block';
          
          // Update edges by ID
          if (msgEdges) {
            ['e1', 'e2', 'e3', 'e4', 'e5'].forEach(edgeId => {
              msgEdges.update({ 
                id: edgeId, 
                color: { color: '#667eea' }, 
                width: 4,
                label: 'msg',
                arrows: { to: { enabled: true, scaleFactor: 0.8 } }
              });
            });
          }
          
          const stepBtn = document.getElementById('step-msg');
          if (stepBtn) stepBtn.textContent = '⏭ Step 3: Normalize';
        },
        2: () => {
          // Step 3: Normalize
          updateStatusPanel(3);
          const stepBtn = document.getElementById('step-msg');
          if (stepBtn) stepBtn.textContent = '⏭ Step 4: Aggregate';
        },
        3: () => {
          // Step 4: Aggregate
          updateStatusPanel(4);
          if (msgNodes) {
            msgNodes.update({ 
              id: 0, 
              color: { background: '#9c27b0', border: '#6a1b9a' },
              label: 'P0\nAggregating...',
              font: { color: '#ffffff' }
            });
          }
          const stepBtn = document.getElementById('step-msg');
          if (stepBtn) stepBtn.textContent = '⏭ Step 5: Activate';
        },
        4: () => {
          // Step 5: Activation
          updateStatusPanel(5);
          if (msgNodes) {
            msgNodes.update({ 
              id: 0, 
              color: { background: '#4caf50', border: '#2e7d32' },
              label: 'P0\n[0.48, 0.52, 0.44]',
              title: 'Updated Features\n[0.48, 0.52, 0.44]',
              font: { color: '#000000' }
            });
          }
          const stepBtn = document.getElementById('step-msg');
          if (stepBtn) stepBtn.textContent = '🔄 Reset';
        }
      };
      
      // Execute current step and increment
      if (currentStep < 5 && stepActions[currentStep]) {
        try {
          stepActions[currentStep]();
          currentStep++;
        } catch (error) {
          console.error('Error executing step', currentStep, error);
        }
      } else if (currentStep >= 5) {
        // Reset after all steps
        resetMessagePassing();
      }
    }

    function resetMessagePassing() {
      isAnimating = false;
      currentStep = 0;
      msgNodes.update([
        { id: 0, label: 'P0\n[0.50, 0.30, 0.80]', color: { background: '#667eea', border: '#4c51bf' }, font: { color: '#000000' } },
        { id: 1, label: 'P1\n[0.70, 0.20, 0.10]', color: { background: '#f093fb', border: '#ea580c' }, font: { color: '#000000' } },
        { id: 2, label: 'P2\n[0.20, 0.90, 0.30]', color: { background: '#f093fb', border: '#ea580c' }, font: { color: '#000000' } },
        { id: 3, label: 'P3\n[0.40, 0.60, 0.50]', color: { background: '#f093fb', border: '#ea580c' }, font: { color: '#000000' } },
        { id: 4, label: 'P4\n[0.80, 0.40, 0.20]', color: { background: '#f093fb', border: '#ea580c' }, font: { color: '#000000' } },
        { id: 5, label: 'P5\n[0.30, 0.70, 0.60]', color: { background: '#f093fb', border: '#ea580c' }, font: { color: '#000000' } }
      ]);
      // Reset edges by ID
      ['e1', 'e2', 'e3', 'e4', 'e5'].forEach(edgeId => {
        msgEdges.update({ 
          id: edgeId, 
          color: { color: '#888' }, 
          width: 2,
          label: '',
          arrows: { to: { enabled: false } }
        });
      });
      document.getElementById('step-msg').textContent = '⏭ Next Step';
      document.getElementById('msg-formula').style.display = 'none';
      updateStatusPanel(0);
    }

    // Initialize when page loads
    function tryInit() {
      if (typeof vis !== 'undefined') {
        if (document.readyState === 'loading') {
          document.addEventListener('DOMContentLoaded', initMessagePassing);
        } else {
          initMessagePassing();
        }
      } else {
        setTimeout(tryInit, 100);
      }
    }
    tryInit();
  }
</script>

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

Each layer expands the **receptive field** of a node, allowing it to gather information from progressively distant neighbors:

<div class="layer-hops-visualization" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
  <div class="layer-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 1.5rem; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); color: white;">
    <div style="text-align: center; margin-bottom: 1rem;">
      <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">1</div>
      <div style="font-size: 1.1rem; font-weight: 600; opacity: 0.95;">Layer 1</div>
    </div>
    <div id="layer1-graph" style="height: 180px; background: rgba(255, 255, 255, 0.15); border-radius: 10px; margin-bottom: 1rem;"></div>
    <div style="text-align: center; font-size: 0.95rem; opacity: 0.9;">
      <strong>1-hop:</strong> Immediate neighbors
    </div>
  </div>
  
  <div class="layer-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; padding: 1.5rem; box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3); color: white;">
    <div style="text-align: center; margin-bottom: 1rem;">
      <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">2</div>
      <div style="font-size: 1.1rem; font-weight: 600; opacity: 0.95;">Layer 2</div>
    </div>
    <div id="layer2-graph" style="height: 180px; background: rgba(255, 255, 255, 0.15); border-radius: 10px; margin-bottom: 1rem;"></div>
    <div style="text-align: center; font-size: 0.95rem; opacity: 0.9;">
      <strong>2-hop:</strong> Neighbors of neighbors
    </div>
  </div>
  
  <div class="layer-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; padding: 1.5rem; box-shadow: 0 8px 20px rgba(79, 172, 254, 0.3); color: white;">
    <div style="text-align: center; margin-bottom: 1rem;">
      <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">3</div>
      <div style="font-size: 1.1rem; font-weight: 600; opacity: 0.95;">Layer 3</div>
    </div>
    <div id="layer3-graph" style="height: 180px; background: rgba(255, 255, 255, 0.15); border-radius: 10px; margin-bottom: 1rem;"></div>
    <div style="text-align: center; font-size: 0.95rem; opacity: 0.9;">
      <strong>3-hop:</strong> Three steps away
    </div>
  </div>
  
  <div class="layer-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); border-radius: 15px; padding: 1.5rem; box-shadow: 0 8px 20px rgba(250, 112, 154, 0.3); color: white;">
    <div style="text-align: center; margin-bottom: 1rem;">
      <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">4</div>
      <div style="font-size: 1.1rem; font-weight: 600; opacity: 0.95;">Layer 4</div>
    </div>
    <div id="layer4-graph" style="height: 180px; background: rgba(255, 255, 255, 0.15); border-radius: 10px; margin-bottom: 1rem;"></div>
    <div style="text-align: center; font-size: 0.95rem; opacity: 0.9;">
      <strong>4-hop:</strong> Four steps away
    </div>
  </div>
</div>

<script type="text/javascript">
  if (typeof window !== 'undefined') {
    function createHopGraph(containerId, hops) {
      if (typeof vis === 'undefined') return;
      const container = document.getElementById(containerId);
      if (!container) return;

      // Central node
      const nodes = [{ 
        id: 0, 
        label: 'Center', 
        x: 0, 
        y: 0, 
        fixed: true,
        color: { background: '#ffffff', border: '#333333' },
        font: { color: '#000000', size: 12 },
        size: 20
      }];

      const edges = [];
      let nodeId = 1;
      const radius = 50;
      const angles = {
        1: [0, Math.PI * 2 / 3, Math.PI * 4 / 3], // 3 immediate neighbors
        2: [0, Math.PI * 2 / 3, Math.PI * 4 / 3, Math.PI / 6, Math.PI * 5 / 6, Math.PI * 3 / 2], // 6 nodes
        3: [], // Will be auto-generated
        4: [] // Will be auto-generated
      };

      // Layer 1: 3 immediate neighbors
      if (hops === 1) {
        const layer1Nodes = 3;
        for (let i = 0; i < layer1Nodes; i++) {
          const angle = angles[1][i];
          nodes.push({
            id: nodeId++,
            label: `N${i+1}`,
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
            fixed: true,
            color: { background: 'rgba(255, 255, 255, 0.9)', border: '#4c51bf' },
            font: { color: '#000000', size: 11 },
            size: 16
          });
          edges.push({ from: 0, to: nodeId - 1, color: { color: 'rgba(255, 255, 255, 0.8)' }, width: 2 });
        }
      }
      // Layer 2: 2-hop neighbors (6 nodes total)
      else if (hops === 2) {
        // First layer neighbors
        const layer1Nodes = 3;
        for (let i = 0; i < layer1Nodes; i++) {
          const angle = angles[1][i];
          nodes.push({
            id: nodeId++,
            label: `N${i+1}`,
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
            fixed: true,
            color: { background: 'rgba(255, 255, 255, 0.9)', border: '#4c51bf' },
            font: { color: '#000000', size: 11 },
            size: 16
          });
          edges.push({ from: 0, to: nodeId - 1, color: { color: 'rgba(255, 255, 255, 0.8)' }, width: 2 });
        }
        // Second layer neighbors
        for (let i = 0; i < layer1Nodes; i++) {
          const angle = angles[1][i];
          const layer2NodeId = nodeId++;
          nodes.push({
            id: layer2NodeId,
            label: `N${i+4}`,
            x: Math.cos(angle) * (radius * 1.8),
            y: Math.sin(angle) * (radius * 1.8),
            fixed: true,
            color: { background: 'rgba(255, 255, 255, 0.7)', border: '#764ba2' },
            font: { color: '#000000', size: 10 },
            size: 14
          });
          // Connect to first layer neighbor
          edges.push({ from: i + 1, to: layer2NodeId, color: { color: 'rgba(255, 255, 255, 0.6)' }, width: 1.5 });
        }
      }
      // Layer 3: 3-hop neighbors
      else if (hops === 3) {
        // Build a tree structure
        const layers = [
          [{ angle: 0, dist: radius }],
          [{ angle: Math.PI / 6, dist: radius * 1.5 }, { angle: -Math.PI / 6, dist: radius * 1.5 }],
          [{ angle: Math.PI / 3, dist: radius * 2.2 }, { angle: -Math.PI / 3, dist: radius * 2.2 }, { angle: Math.PI, dist: radius * 2.2 }]
        ];
        
        let prevLayerIds = [0];
        layers.forEach((layer, layerIdx) => {
          const currentLayerIds = [];
          layer.forEach((nodeSpec) => {
            const nodeIdNew = nodeId++;
            nodes.push({
              id: nodeIdNew,
              label: `L${layerIdx+1}`,
              x: Math.cos(nodeSpec.angle) * nodeSpec.dist,
              y: Math.sin(nodeSpec.angle) * nodeSpec.dist,
              fixed: true,
              color: { background: `rgba(255, 255, 255, ${0.9 - layerIdx * 0.15})`, border: layerIdx === 0 ? '#4c51bf' : '#00f2fe' },
              font: { color: '#000000', size: 10 },
              size: 16 - layerIdx * 2
            });
            // Connect to a node from previous layer
            const parentId = prevLayerIds[Math.floor(Math.random() * prevLayerIds.length)];
            edges.push({ from: parentId, to: nodeIdNew, color: { color: `rgba(255, 255, 255, ${0.8 - layerIdx * 0.2})` }, width: 2 - layerIdx * 0.3 });
            currentLayerIds.push(nodeIdNew);
          });
          prevLayerIds = [...prevLayerIds, ...currentLayerIds];
        });
      }
      // Layer 4: 4-hop neighbors
      else if (hops === 4) {
        // Build a deeper tree structure
        const layers = [
          [{ angle: 0, dist: radius }],
          [{ angle: Math.PI / 4, dist: radius * 1.4 }, { angle: -Math.PI / 4, dist: radius * 1.4 }],
          [{ angle: Math.PI / 3, dist: radius * 2.0 }, { angle: -Math.PI / 3, dist: radius * 2.0 }],
          [{ angle: Math.PI / 2, dist: radius * 2.6 }, { angle: -Math.PI / 2, dist: radius * 2.6 }, { angle: Math.PI, dist: radius * 2.6 }]
        ];
        
        let prevLayerIds = [0];
        layers.forEach((layer, layerIdx) => {
          const currentLayerIds = [];
          layer.forEach((nodeSpec) => {
            const nodeIdNew = nodeId++;
            nodes.push({
              id: nodeIdNew,
              label: `L${layerIdx+1}`,
              x: Math.cos(nodeSpec.angle) * nodeSpec.dist,
              y: Math.sin(nodeSpec.angle) * nodeSpec.dist,
              fixed: true,
              color: { background: `rgba(255, 255, 255, ${0.9 - layerIdx * 0.15})`, border: layerIdx === 0 ? '#4c51bf' : '#fee140' },
              font: { color: '#000000', size: 9 },
              size: 16 - layerIdx * 2
            });
            const parentId = prevLayerIds[Math.floor(Math.random() * prevLayerIds.length)];
            edges.push({ from: parentId, to: nodeIdNew, color: { color: `rgba(255, 255, 255, ${0.8 - layerIdx * 0.15})` }, width: 2 - layerIdx * 0.25 });
            currentLayerIds.push(nodeIdNew);
          });
          prevLayerIds = [...prevLayerIds, ...currentLayerIds];
        });
      }

      const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
      const options = {
        physics: false,
        interaction: { zoomView: false, dragView: false, dragNodes: false },
        nodes: {
          shape: 'dot',
          font: { color: '#000000', size: 11 }
        },
        edges: {
          smooth: false,
          arrows: { to: false }
        },
        layout: {
          hierarchical: false
        }
      };

      new vis.Network(container, data, options);
    }

    function initLayerHops() {
      const tryInit = () => {
        if (typeof vis !== 'undefined') {
          createHopGraph('layer1-graph', 1);
          createHopGraph('layer2-graph', 2);
          createHopGraph('layer3-graph', 3);
          createHopGraph('layer4-graph', 4);
        } else {
          setTimeout(tryInit, 100);
        }
      };
      
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', tryInit);
      } else {
        tryInit();
      }
    }

    initLayerHops();
  }
</script>

<div class="info-box warning">
  <strong>⚠️ Over-smoothing Problem:</strong> Deeper networks allow information to propagate farther, but too many layers can lead to **over-smoothing** (all nodes become similar). We'll experiment with different depths to find the sweet spot.
</div>

---

## Building Our GCN Architecture

### The Model Definition

### Architecture Diagram

Our GCN architecture follows this structure:

<div id="architecture-diagram" class="mermaid" style="background: rgba(255, 255, 255, 0.9); padding: 1.5rem; border-radius: 10px; box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1); margin: 2rem 0;">
graph LR
    A["Input Features<br/>50 dimensions"] -->|"GCN Layer 1<br/>+ ReLU + Dropout"| B["Hidden Layer<br/>256 dimensions"]
    B -->|"GCN Layer 2<br/>+ ReLU + Dropout"| C["Hidden Layer<br/>256 dimensions"]
    C -->|"GCN Layer 3<br/>(No activation)"| D["Output Logits<br/>121 classes"]
    D -->|"Sigmoid"| E["Multi-label<br/>Predictions"]
    
    style A fill:#667eea,stroke:#4c51bf,color:#fff
    style B fill:#f093fb,stroke:#ea580c,color:#000
    style C fill:#f093fb,stroke:#ea580c,color:#000
    style D fill:#4facfe,stroke:#00f2fe,color:#000
    style E fill:#4caf50,stroke:#2e7d32,color:#fff
</div>

**Key points:**
- **Input:** 50-dimensional node features (biological descriptors)
- **Hidden layers:** 256-dimensional embeddings (configurable number of layers)
- **Output:** 121 logits (one per function class)
- **Activation:** ReLU between layers, no activation on final layer (sigmoid applied in loss)

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

Now that we have our architecture defined, let's train it and see how it performs!

---

## Training and Evaluation

### The Loss Function: Binary Cross Entropy

Since this is multi-label classification, we need a loss function that handles multiple independent binary predictions. We use **Binary Cross Entropy with Logits Loss**:

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

When evaluating multi-label classification, choosing the right metric is crucial:

For multi-label classification, we use **micro-averaged F1** because:
- It treats each label prediction as an individual binary classification
- It's more robust to class imbalance than macro-averaged F1
- It's the standard metric for PPI function prediction tasks

---

## Results and Insights

### Training Curves

Let's examine how different architectures converge during training:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
  <div style="background: rgba(255, 255, 255, 0.9); padding: 1.5rem; border-radius: 10px; box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);">
    <h4 style="margin-top: 0; color: #667eea; text-align: center;">Training Loss</h4>
    <img src="{{ site.baseurl }}/assets/loss_curve.png" alt="Loss Curve" style="width: 100%; height: auto; border-radius: 5px;">
  </div>
  <div style="background: rgba(255, 255, 255, 0.9); padding: 1.5rem; border-radius: 10px; box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);">
    <h4 style="margin-top: 0; color: #667eea; text-align: center;">Validation F1 Score</h4>
    <img src="{{ site.baseurl }}/assets/f1_curve.png" alt="F1 Curve" style="width: 100%; height: auto; border-radius: 5px;">
  </div>
</div>

**Key observations:**
- **2-Layer GCN:** Converges smoothly and achieves the lowest training loss with stable F1 scores
- **3-Layer GCN:** Shows more variance in F1 scores during training, but can achieve good performance
- **4-Layer GCN:** Struggles more—deeper networks can suffer from over-smoothing, leading to lower F1 scores

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
      <td><strong>0.564</strong></td>
      <td><span class="badge badge-success">Best</span></td>
    </tr>
    <tr>
      <td><strong>3-Layer GCN</strong></td>
      <td>0.534</td>
      <td><span class="badge badge-info">Good</span></td>
    </tr>
    <tr>
      <td><strong>4-Layer GCN</strong></td>
      <td>0.462</td>
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

<div style="text-align: center; margin: 2.5rem 0;">
  <a href="https://colab.research.google.com/drive/1ldaz7PJEzqI_cbFVz23U94Wngvsyll_H?usp=sharing" target="_blank" class="colab-button">
    <span style="font-size: 1.5rem; line-height: 1;">🚀</span>
    <span>Open in Google Colab</span>
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="flex-shrink: 0;">
      <path d="M7 17L17 7M17 7H7M17 7V17" stroke-linecap="round"/>
    </svg>
  </a>
</div>

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

<!-- Mermaid.js for architecture diagrams -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  if (typeof window !== 'undefined') {
    function initMermaid() {
      if (typeof mermaid !== 'undefined') {
        mermaid.initialize({ 
          startOnLoad: true,
          theme: 'default',
          flowchart: { 
            useMaxWidth: true, 
            htmlLabels: true,
            curve: 'basis'
          },
          securityLevel: 'loose'
        });
        
        // Explicitly render if already loaded
        if (document.readyState === 'complete' || document.readyState === 'interactive') {
          mermaid.run();
        }
      }
    }
    
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initMermaid);
    } else {
      initMermaid();
    }
  }
</script>
