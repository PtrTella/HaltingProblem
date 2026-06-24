import re
import torch
import networkx as nx
from torch_geometric.data import Data

INSTRUCTION_VOCAB = [
    'alloca', 'store', 'load', 'add', 'sub', 'mul', 'sdiv', 'udiv', 'urem', 'srem',
    'icmp', 'fcmp', 'br', 'ret', 'phi', 'select', 'call', 'shl', 'lshr', 'ashr',
    'and', 'or', 'xor', 'getelementptr', 'extractvalue', 'insertvalue', 'unreachable'
]
VOCAB_MAP = {inst: i for i, inst in enumerate(INSTRUCTION_VOCAB)}

def parse_llvm_ir(llvm_ir_str: str) -> Data:
    """
    Parses a string of LLVM IR and returns a PyTorch Geometric Data object.
    
    Nodes represent basic blocks, and directed edges represent control flow jumps.
    Node features represent the occurrences of instruction types in the basic block
    (28-dimensional bag-of-instructions frequency vector).
    """
    lines = llvm_ir_str.splitlines()
    functions = {}
    current_func = None
    current_block = None
    
    # First pass: parse functions, basic blocks, and their instruction contents
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Check for function definition start
        # e.g., "define i32 @main(i32 %0) {"
        func_match = re.match(r'^define\s+\S+\s+@([a-zA-Z0-9._]+)\s*\(', stripped)
        if func_match:
            current_func = func_match.group(1)
            functions[current_func] = {}
            current_block = "entry"  # default starting block
            functions[current_func][current_block] = []
            continue
            
        # Check for function definition end
        if stripped == "}":
            current_func = None
            current_block = None
            continue
            
        if current_func is None:
            continue
            
        # Check for label comments like '; <label>:4' or '; <label>:loop'
        label_comment_match = re.match(r'^;\s*<label>:([a-zA-Z0-9._]+)', stripped)
        if label_comment_match:
            current_block = label_comment_match.group(1)
            if current_block not in functions[current_func]:
                functions[current_func][current_block] = []
            continue
            
        # Check for standard label declarations like 'entry:' or '4: ; preds = %0'
        label_match = re.match(r'^([a-zA-Z0-9._]+):', stripped)
        if label_match:
            current_block = label_match.group(1)
            if current_block not in functions[current_func]:
                functions[current_func][current_block] = []
            continue
            
        # Strip trailing comments for instruction parsing
        clean_line = re.sub(r';.*', '', stripped).strip()
        if not clean_line:
            continue
            
        # Extract opcode
        inst_part = clean_line
        if '=' in clean_line:
            parts = clean_line.split('=', 1)
            inst_part = parts[1].strip()
            
        # First word of the instruction part is the opcode (e.g. alloca, br, ret)
        inst_match = re.match(r'^([a-z0-9]+)', inst_part)
        if inst_match:
            opcode = inst_match.group(1)
            functions[current_func][current_block].append({
                'opcode': opcode,
                'line': clean_line
            })

    # Second pass: construct directed graph and extract edges
    g = nx.DiGraph()
    node_features = {}
    
    for func_name, blocks in functions.items():
        for block_name, insts in blocks.items():
            # Namespace node IDs by prefixing function name to prevent clashes
            node_id = f"{func_name}:{block_name}"
            g.add_node(node_id)
            
            # Construct feature vector: counts of opcodes + 1 for 'other'
            feature_vector = [0.0] * (len(INSTRUCTION_VOCAB) + 1)
            for inst in insts:
                op = inst['opcode']
                if op in VOCAB_MAP:
                    feature_vector[VOCAB_MAP[op]] += 1.0
                else:
                    feature_vector[-1] += 1.0  # 'other'
            node_features[node_id] = feature_vector
            
            # Extract edges from terminator instruction
            if insts:
                last_inst = insts[-1]
                # Branch instruction: br label %X or br i1 %cond, label %Y, label %Z
                if last_inst['opcode'] == 'br':
                    targets = re.findall(r'label\s+%([a-zA-Z0-9._]+)', last_inst['line'])
                    for target in targets:
                        target_id = f"{func_name}:{target}"
                        g.add_edge(node_id, target_id)
                # Switch instruction
                elif last_inst['opcode'] == 'switch':
                    targets = re.findall(r'label\s+%([a-zA-Z0-9._]+)', last_inst['line'])
                    for target in targets:
                        target_id = f"{func_name}:{target}"
                        g.add_edge(node_id, target_id)
                # Note: ret, unreachable, etc. have no successors
                
    # Handle edge case: empty or invalid graph (ensure downstream layers don't crash)
    if len(g.nodes) == 0:
        x = torch.zeros((1, len(INSTRUCTION_VOCAB) + 1), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
        
    # Map node strings to indices for PyTorch Geometric
    node_to_idx = {node: i for i, node in enumerate(g.nodes)}
    
    # Construct tensors
    x_list = [node_features[node] for node in g.nodes]
    x = torch.tensor(x_list, dtype=torch.float)
    
    edges = list(g.edges)
    if edges:
        edge_index_list = [[node_to_idx[u], node_to_idx[v]] for u, v in edges]
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        
    return Data(x=x, edge_index=edge_index)
