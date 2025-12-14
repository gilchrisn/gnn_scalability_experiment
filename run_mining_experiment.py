from src.data import DatasetFactory
from src.anyburl_utils import AnyBURLRunner

# Config
DATASET = 'HNE_DBLP'
ANYBURL_JAR = './tools/AnyBURL-23-1.jar'

def main():
    # 1. Load HNE Data
    print("--- 1. Loading HNE Data ---")
    g, _ = DatasetFactory.get_data("HNE", "DBLP", "author")

    # 2. Export to AnyBURL
    print("--- 2. Exporting to AnyBURL ---")
    miner = AnyBURLRunner("./data/HNE_DBLP", ANYBURL_JAR)
    miner.export_graph(g)

    # 3. Mine Rules
    print("--- 3. Mining Rules ---")
    miner.run_mining(timeout=60) # Run for 60 seconds

    # 4. Parse Results
    print("--- 4. Best Metapath ---")
    # We look for rules that predict 'author_to_paper' or similar relations
    best_rels = miner.parse_best_metapath("author", "author")
    
    if best_rels:
        print(f"Found Best Metapath: {' -> '.join(best_rels)}")
        print("Use this sequence in your src/config.py as TRAIN_METAPATH!")
    else:
        print("No valid rules found.")

if __name__ == "__main__":
    main()