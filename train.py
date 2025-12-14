# ---------- File: ./train_main.py ----------

import sys
import os
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader


import src.config as config  # Import from root
from src import data as dataset # Renamed for clarity
from src import models
from src import utils
from src.lit_model import LitGNN
from src.methods.method1_materialize import _materialize_graph

# List of models to train
MODELS_TO_TRAIN = ["GCN", "GAT", "SAGE"]

def run_training():
    """
    Trains all models on the training metapath using PyTorch Lightning.
    """
    print("="*40)
    print("      PyTorch Lightning Training Pipeline")
    print("="*40)
    
    # --- 1. Load Heterogeneous Data ---
    print("\n[Step 1/3] Loading data...")
    g_hetero, data_info = dataset.load_dblp_data(config.TARGET_NODE_TYPE)
    
    # --- 2. Materialize Homogeneous Graph for Training ---
    print("\n[Step 2/3] Materializing graph for training...")
    
    # --- MODIFIED CALL (added data_info) ---
    g_homo_train, prep_time = _materialize_graph(
        g_hetero,
        data_info, # <-- PASS THE MASKS
        config.TRAIN_METAPATH,
        config.TARGET_NODE_TYPE
    )
    print(f"Materialized graph '{utils.get_metapath_suffix(config.TRAIN_METAPATH)}' in {prep_time:.4f}s")
    
    # --- 3. Wrap in DataLoader ---
    train_loader = DataLoader([g_homo_train], batch_size=1, shuffle=False)
    
    # ... (rest of the file is unchanged) ...
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    print("\n[Step 3/3] Training models...")
    for model_name in MODELS_TO_TRAIN:
        print(f"\n--- Training: {model_name} ---")
        
        checkpoint_path = utils.get_model_checkpoint_path(
            config.MODEL_DIR, 
            model_name, 
            config.TRAIN_METAPATH
        )
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint found at {checkpoint_path}. Skipping training.")
            continue
            
        encoder = models.get_model(
            model_name, 
            data_info['in_dim'], 
            data_info['out_dim'],
            config.HIDDEN_DIM,
            config.GAT_HEADS
        )
        
        lit_model = LitGNN(
            encoder=encoder,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        trainer = pl.Trainer(
            max_epochs=config.EPOCHS,
            accelerator="auto", 
            devices=1,
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=False 
        )
        
        trainer.fit(lit_model, train_loader, train_loader)
        
        trainer.save_checkpoint(checkpoint_path)
        print(f"Model checkpoint saved to: {checkpoint_path}")
        
    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    run_training()