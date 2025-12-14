import torch
import torch.nn.functional as F
import pandas as pd
import time
from src import config, models, utils
from src.data import DatasetFactory
from src.methods.method1_materialize import _materialize_graph
from src.methods.method2_kmv_sample import _run_kmv_propagation, _build_graph_from_sketches

# --- Config ---
EPOCHS = 50  # Quick training
METAPATH_LENGTH = config.METAPATH_LENGTH
NK = 3

def train_model(model, graph, features, labels, masks):
    """
    Quickly trains the model on the provided graph structure.
    """
    print(f"   Training GCN on {graph.num_edges} edges for {EPOCHS} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    
    features = features.to(config.DEVICE)
    graph = graph.to(config.DEVICE)
    labels = labels.to(config.DEVICE)
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        out = model(features, graph.edge_index)
        loss = F.cross_entropy(out[masks['train']], labels[masks['train']])
        loss.backward()
        optimizer.step()
        
    return model

def evaluate(model, graph, features, labels, mask):
    """
    Tests the model using the provided graph structure.
    """
    model.eval()
    features = features.to(config.DEVICE)
    graph = graph.to(config.DEVICE)
    labels = labels.to(config.DEVICE)
    
    with torch.no_grad():
        logits = model(features, graph.edge_index)
        pred = logits.argmax(dim=1)
        correct = (pred[mask] == labels[mask]).sum()
        acc = int(correct) / int(mask.sum())
    return acc

def benchmark():
    print(f"{'='*60}")
    print(f"ACCURACY BENCHMARK: {config.DATASET_NAME}")
    print(f"{'='*60}\n")

    # 1. Load Data
    g_hetero, info = DatasetFactory.get_data(
        config.CURRENT_CONFIG['source'],
        config.CURRENT_CONFIG['dataset_name'],
        config.TARGET_NODE_TYPE
    )
    g_hetero = g_hetero.to(config.DEVICE)
    
    # 2. Metapath
    metapath = utils.generate_random_metapath(g_hetero, config.TARGET_NODE_TYPE, METAPATH_LENGTH)
    print(f"   Metapath: {' -> '.join([t[1] for t in metapath])}")

    # 3. Generate GROUND TRUTH (Exact Graph)
    print("\n1. Materializing Exact Graph (Ground Truth)...")
    g_exact, _ = _materialize_graph(g_hetero, info, metapath, config.TARGET_NODE_TYPE)
    
    # 4. Train Model on Ground Truth
    print("\n2. Training Reference Model on Exact Graph...")
    model = models.get_model("GCN", info['in_dim'], info['out_dim'], config.HIDDEN_DIM, config.GAT_HEADS).to(config.DEVICE)
    model = train_model(model, g_exact, info['features'], info['labels'], info['masks'])
    
    # Baseline Accuracy
    base_acc = evaluate(model, g_exact, info['features'], info['labels'], info['masks']['test'])
    print(f"   >>> BASELINE EXACT ACCURACY: {base_acc:.4f}")

    results = []
    results.append({
        "Method": "Exact", "K": "N/A", 
        "Edges": g_exact.num_edges, 
        "Test_Acc": base_acc,
        "Rel_Acc": 1.0
    })

    # 5. Test KMV Approximations
    print("\n3. Testing KMV Approximations...")
    
    # We use the SAME trained model, but swap the edge_index
    # This tests "Inference Robustness"
    
    for k in config.K_VALUES:
        # A. Generate KMV Graph
        sketches, hashes, rev_map, _ = _run_kmv_propagation(g_hetero, metapath, k, NK, config.DEVICE)
        g_kmv, _ = _build_graph_from_sketches(sketches, hashes, rev_map, g_hetero[config.TARGET_NODE_TYPE].num_nodes, config.DEVICE)
        
        # B. Evaluate (No Retraining!)
        acc = evaluate(model, g_kmv, info['features'], info['labels'], info['masks']['test'])
        
        print(f"   [K={k}] Edges: {g_kmv.num_edges} | Acc: {acc:.4f}")
        
        results.append({
            "Method": "KMV", "K": k,
            "Edges": g_kmv.num_edges, 
            "Test_Acc": acc,
            "Rel_Acc": acc / base_acc # Relative to exact
        })
        
        del g_kmv, sketches, hashes, rev_map
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(pd.DataFrame(results).to_markdown(index=False))

if __name__ == "__main__":
    benchmark()