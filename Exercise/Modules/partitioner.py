import numpy as np
import os
import tempfile
from typing import List, Set, Dict, Tuple


def graph_to_csr(
    adjacency: List[Set[int]],
    edge_data: Dict[Tuple[int, int], dict],
    n_nodes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert weighted undirected graph to CSR format for KaHIP."""
    vwgt = np.ones(n_nodes, dtype=np.int32)
    
    # Build adjacency and weight arrays
    adjncy_list = []
    adjcwgt_list = []
    xadj = np.zeros(n_nodes + 1, dtype=np.int32)
    
    for node_id in range(n_nodes):
        # Start index for this node
        xadj[node_id] = len(adjncy_list)
        
        # Get sorted neighbors for reproducibility
        neighbors = sorted(adjacency[node_id])
        
        for neighbor in neighbors:
            # Get edge weight from edge_data
            edge_key = (min(node_id, neighbor), max(node_id, neighbor))
            weight = edge_data[edge_key]['weight']
            
            adjncy_list.append(neighbor)
            adjcwgt_list.append(weight)
    
    # Final index = total number of edges (×2 for undirected)
    xadj[n_nodes] = len(adjncy_list)
    
    # Convert to numpy arrays
    adjncy = np.array(adjncy_list, dtype=np.int32)
    adjcwgt = np.array(adjcwgt_list, dtype=np.int32)
    
    return vwgt, xadj, adjncy, adjcwgt


def write_kahip_graph(
    output_path: str,
    vwgt: np.ndarray,
    xadj: np.ndarray,
    adjncy: np.ndarray,
    adjcwgt: np.ndarray
) -> None:
    """
    Write graph in KaHIP's METIS format.
    
    METIS format specification:
    - Line 1: n_nodes n_edges format_flags
      - format_flags = 11: vertex weights (1), edge weights (1), no vertex size (1)
    - Lines 2 to n+1: For each node (1-indexed in file):
      - [vertex_weight] neighbor1 edge_weight1 neighbor2 edge_weight2 ...
    
    Note: METIS uses 1-based indexing for nodes, so we add 1 to all node indices.
    """
    n_nodes = len(vwgt)
    # Number of undirected edges (each edge counted once)
    n_edges = len(adjncy) // 2
    
    with open(output_path, 'w') as f:
        # Header line: n_nodes n_edges format_flags
        # Format 11: vertex weights=1, edge weights=1, no vertex size=1
        f.write(f"{n_nodes} {n_edges} 11\n")
        
        # Write each node's adjacency list
        for node_id in range(n_nodes):
            # Vertex weight
            f.write(f"{vwgt[node_id]}")
            
            # Get neighbors and edge weights for this node
            start_idx = xadj[node_id]
            end_idx = xadj[node_id + 1]
            
            for i in range(start_idx, end_idx):
                neighbor = adjncy[i]
                edge_weight = adjcwgt[i]
                
                # METIS uses 1-based indexing, so add 1 to neighbor
                f.write(f" {neighbor + 1} {edge_weight}")
            
            f.write("\n")


def validate_csr_format(
    vwgt: np.ndarray,
    xadj: np.ndarray,
    adjncy: np.ndarray,
    adjcwgt: np.ndarray,
    n_nodes: int
) -> bool:
    """Validate CSR format arrays for consistency."""
    # Check shapes
    if len(vwgt) != n_nodes:
        raise ValueError(f"vwgt length {len(vwgt)} != n_nodes {n_nodes}")
    
    if len(xadj) != n_nodes + 1:
        raise ValueError(f"xadj length {len(xadj)} != n_nodes + 1 = {n_nodes + 1}")
    
    if len(adjncy) != len(adjcwgt):
        raise ValueError(f"adjncy length {len(adjncy)} != adjcwgt length {len(adjcwgt)}")
    
    # Check xadj is monotonically increasing
    if not np.all(xadj[1:] >= xadj[:-1]):
        raise ValueError("xadj is not monotonically increasing")
    
    # Check xadj[0] = 0 and xadj[n] = len(adjncy)
    if xadj[0] != 0:
        raise ValueError(f"xadj[0] = {xadj[0]}, expected 0")
    
    if xadj[n_nodes] != len(adjncy):
        raise ValueError(f"xadj[{n_nodes}] = {xadj[n_nodes]}, expected {len(adjncy)}")
    
    # Check all node indices are valid
    if len(adjncy) > 0:
        if np.min(adjncy) < 0 or np.max(adjncy) >= n_nodes:
            raise ValueError(f"Invalid node indices in adjncy: min={np.min(adjncy)}, max={np.max(adjncy)}")
    
    # Check edge weights are positive
    if len(adjcwgt) > 0:
        if np.min(adjcwgt) <= 0:
            raise ValueError(f"Edge weights must be positive, got min={np.min(adjcwgt)}")
    
    return True


def partition_graph(
    adjacency: List[Set[int]],
    edge_data: Dict[Tuple[int, int], dict],
    n_parts: int,
    imbalance: float = 0.03,
    mode: int = 2,
    seed: int = 1
) -> np.ndarray:
    """Partition graph using KaHIP's kaffpa executable."""
    import subprocess
    import shutil
    import os
    
    # Expand ~ in PATH search
    expanded_paths = []
    for path in os.environ.get('PATH', '').split(':'):
        expanded_paths.append(os.path.expanduser(path))
    
    # Check common KaHIP installation locations
    kaffpa_locations = [
        shutil.which('kaffpa'),  # Standard PATH
        os.path.expanduser('~/KaHIP/build/kaffpa'),
        os.path.expanduser('~/KaHIP/deploy/kaffpa'),
        '/usr/local/bin/kaffpa',
        '/usr/bin/kaffpa'
    ]
    
    kaffpa_path = None
    for loc in kaffpa_locations:
        if loc and os.path.isfile(loc) and os.access(loc, os.X_OK):
            kaffpa_path = loc
            break
    
    if kaffpa_path is None:
        raise FileNotFoundError(
            "KaHIP's 'kaffpa' executable not found in PATH or common locations. "
            "Please install KaHIP before using graph partitioning. "
            "Searched locations: ~/KaHIP/build/, ~/KaHIP/deploy/, /usr/local/bin/, /usr/bin/. "
            "See: https://github.com/KaHIP/KaHIP"
        )
    
    n_nodes = len(adjacency)
    
    # Convert graph to CSR format
    vwgt, xadj, adjncy, adjcwgt = graph_to_csr(adjacency, edge_data, n_nodes)
    validate_csr_format(vwgt, xadj, adjncy, adjcwgt, n_nodes)
    
    # Map mode to KaHIP preconfiguration
    mode_map = {
        0: 'fast',
        1: 'eco',
        2: 'strong'
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid mode {mode}, must be 0 (FAST), 1 (ECO), or 2 (STRONG)")
    mode_str = mode_map[mode]
    
    # Use temporary directory for graph and partition files
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(tmpdir, 'graph.metis')
        partition_file = os.path.join(tmpdir, 'graph.metis.part')
        
        # Write graph in METIS format
        write_kahip_graph(graph_file, vwgt, xadj, adjncy, adjcwgt)
        
        # Build kaffpa command
        # Note: imbalance is specified as percentage (3 for 3%)
        imbalance_pct = int(imbalance * 100)
        
        cmd = [
            kaffpa_path,  # Use full path instead of 'kaffpa'
            graph_file,
            f'--k={n_parts}',
            f'--imbalance={imbalance_pct}',
            f'--preconfiguration={mode_str}',
            f'--seed={seed}',
            f'--output_filename={partition_file}'
        ]
        
        # Invoke KaHIP
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=72000  # 20 hours timeout for STRONG mode on large graphs (1M points)
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"KaHIP (kaffpa) failed with exit code {e.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"KaHIP (kaffpa) timed out after 72000 seconds (20 hours)")
        
        # Parse partition file
        if not os.path.exists(partition_file):
            raise RuntimeError(
                f"KaHIP did not produce partition file: {partition_file}\n"
                f"Stdout: {result.stdout}\n"
                f"Stderr: {result.stderr}"
            )
        
        # Read partition labels (one per line, 0-indexed)
        with open(partition_file, 'r') as f:
            labels_list = []
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    labels_list.append(int(line))
        
        # Convert to numpy array
        labels = np.array(labels_list, dtype=np.int32)
        
        # Validate partition
        if len(labels) != n_nodes:
            raise RuntimeError(
                f"Partition file has {len(labels)} labels but graph has {n_nodes} nodes"
            )
        
        if labels.min() < 0 or labels.max() >= n_parts:
            raise RuntimeError(
                f"Invalid partition labels: min={labels.min()}, max={labels.max()}, "
                f"expected range [0, {n_parts-1}]"
            )
        
        # Check that all partitions are used
        unique_labels = np.unique(labels)
        if len(unique_labels) != n_parts:
            import warnings
            warnings.warn(
                f"Partition has {len(unique_labels)} non-empty parts but requested {n_parts} parts"
            )
    
    return labels


def partition_knn_graph(
    indices: np.ndarray,
    distances: np.ndarray,
    n_parts: int,
    imbalance: float = 0.03,
    mode: int = 2,
    seed: int = 1
) -> np.ndarray:
    """Partition k-NN graph end-to-end using KaHIP."""
    from Exercise.Modules.graph_utils import build_weighted_graph
    
    # Build weighted undirected graph
    adjacency, edge_data = build_weighted_graph(indices, distances)
    
    # Partition using KaHIP
    labels = partition_graph(adjacency, edge_data, n_parts, imbalance, mode, seed)
    
    return labels
