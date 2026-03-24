# Sketch Edge Overcount Analysis

**Date:** 2026-03-25
**Status:** Hypothesis — needs verification before code changes

## Observation

On OAG_CS PAP metapath (`rev_AP_write_first,AP_write_first`) at 20% (125K papers):

| Method | Edges | Avg Degree |
|--------|-------|------------|
| Exact (`materialize`) | 371,242 | 3.67 |
| Sketch (`sketch k=32`) | 1,303,508 | 12.88 |

72,956 / 101,196 nodes (72%) have **sketch degree > exact degree**, which should be impossible since KMV sketch is a subset approximation.

On HGB_ACM PSP metapath (`paper_to_subject,subject_to_paper`):

| Method | Edges | Avg Degree |
|--------|-------|------------|
| Exact | 2,217,089 | 733.9 |
| Sketch (k=32) | 91,151 | 31.1 |

**0 violators** — sketch degree <= exact degree for every node. Constraint holds perfectly.

## Spot Check

Node 565987 on OAG_CS:
- Exact neighbors: {565987} (just self-loop, degree 1)
- Sketch neighbors: {565987, 606103, 583967, 583666, ...} (21 neighbors)
- Extra (false positives): 20
- Missing: 0
- All real neighbors present, plus 20 that don't belong

## Hypothesis: Edge-Type Leakage in hidden_edges

### How exact materialization works (`run_materialization` → `hidden_graph_construction` in hg.cpp)

For each peer paper P:
1. `MidHop(P, ...)` — follows ONLY the specified edge types (checks `et == pattern->ETypes[pattern_pos]` in peer.cpp:47) to find intermediate nodes
2. `rMidHop(f, ...)` — follows ONLY the reverse edge types back to find PAP neighbors

**Key:** MidHop/rMidHop filter by edge type. Only edges matching the metapath are traversed.

### How sketch materialization works (`run_sketch_sampling` → `gnn_synopses` in join_synopse.cpp)

1. Build hidden_edges (ReadAnyBURLRules.cpp:1466-1483):
```cpp
for (unsigned int nbr: *(g.EL[p])) {           // ALL outgoing edges
    if (ractive->at(l + 1)->at(nbr) > 0) {     // only checks: is target active?
        hidden_edges->push_back(he);            // edge type NOT checked
    }
}
```

2. `gnn_synopses` propagates hashes along ALL hidden_edges (forward pass), then backward.

**Key:** `g.EL[p]` (defined in hin.cpp) contains ALL outgoing edges from node p regardless of edge type. The `ractive` filter checks if the neighbor is active at the next layer, but does NOT check if the edge type matches the metapath.

### Why HGB_ACM works but OAG_CS doesn't

**HGB_ACM:** Between papers and subjects, there is only ONE edge type (`paper_to_subject` / `subject_to_paper`). So `g.EL[p]` for a paper towards subjects only contains edges of the correct type. No leakage.

**OAG_CS:** Between papers and authors, there are MULTIPLE edge types:
- `AP_write_first` (first author)
- `AP_write_other` (other authors)
- `AP_write_last` (last author)
- Possibly more

The metapath specifies `rev_AP_write_first,AP_write_first` — only first-author connections. But `g.EL[p]` includes edges of ALL types to authors. If author A is active (reachable via `AP_write_first`), then ALL edges to A (including `AP_write_other`) pass the `ractive` check and enter hidden_edges.

This causes hash values to propagate along non-metapath edges, inflating the sketch neighborhood.

### Scope of the issue

- **Affects sketch reconstruction:** False positive edges in the output adjacency file
- **Affects GloD/GloH degree estimation:** Same hidden_edges construction is used in `effectiveness.cpp` (COD_prop_global_cross_f1, lines 788-804). Degree estimates include edges from wrong types.
- **Does NOT affect exact materialization:** `hidden_graph_construction` uses `MidHop`/`rMidHop` which filter by edge type
- **Only manifests when** multiple edge types exist between the same pair of node types

### Datasets affected

| Dataset | Edge types between same node-type pairs | Affected? |
|---------|----------------------------------------|-----------|
| HGB_ACM | 1 per pair | No |
| HGB_DBLP | 1 per pair | No |
| HGB_IMDB | 1 per pair | No |
| OGB_MAG | Multiple (writes, affiliated_with share author type) | Possibly |
| OAG_CS | Multiple (AP_write_first, AP_write_other, AP_write_last) | Yes |

## Verification Steps (before any code changes)

1. **Confirm edge types in OAG_CS:**
   ```python
   for et in g.edge_types:
       src, rel, dst = et
       print(f"{src} --[{rel}]--> {dst}: {g[et].edge_index.size(1)} edges")
   ```
   Check: are there multiple edge types between paper↔author?

2. **Count hidden_edges by type:** Add a counter in C++ to log how many hidden_edges come from each edge type. Compare against the metapath-specified types.

3. **Check OGB_MAG PAP:** `rev_writes,writes` — is `writes` the only edge type between paper↔author in OGB_MAG? If yes, OGB_MAG PAP should have 0 violators.

4. **Check GloD accuracy:** If GloD uses the same inflated hidden_edges, its degree estimates should also be inflated. Compare GloD degree estimates against exact degrees for a few nodes on OAG_CS.

## Potential Fixes

### Option A: Filter hidden_edges by edge type in C++ (proper fix)
- Requires `g.EL[p]` to be replaced with type-specific edge lists
- Or: tag each edge in EL with its type and filter during hidden_edges construction
- Risk: changes core C++ data structure used by all commands

### Option B: Build type-filtered hidden_edges in Python, pass to C++
- Python knows the edge types. Could write a filtered edge list file.
- C++ sketch reads from file instead of building from `g.EL`
- Risk: new I/O pathway, potential format mismatch

### Option C: Post-process sketch output in Python
- After loading sketch adjacency, intersect with exact PAP edges computed in Python
- Defeats the purpose of sketch (need to compute exact to filter)

### Option D: Accept over-approximation, document limitation
- CKA is still reasonable (0.83-0.87)
- The extra edges add noise but the GNN is robust
- Report sketch edges honestly, note the edge-type leakage
- Focus the paper's scalability argument on materialization TIME, not edge count

## Related: Layer 0 vs Layer meta_layer fix (already applied)

A separate issue was found and fixed: the sketch was reading from layer 0 (after backward pass = 2*L hop neighborhoods) instead of layer meta_layer (end of forward pass = L hop neighborhoods). This reduced sketch edges from 1.9M to 1.3M on OAG_CS. The remaining 1.3M > 371K is due to the edge-type leakage described above.

On HGB_ACM, the layer fix + no edge-type leakage = 0 violators (correct behavior).
