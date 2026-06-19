"""
Interpreter and vocabulary mappings for the minimalist Turing-complete toy language.
"""

# Vocabulary Constants
PAD_TOKEN = 0
HLT_TOKEN = 1
INC_TOKEN_BASE = 2
DEC_TOKEN_BASE = 4
JNZ_TOKEN_BASE = 6

def inst_to_token(inst):
    """
    Converts an instruction tuple to a token ID.
    Example:
        ('PAD',) -> 0
        ('HLT',) -> 1
        ('INC', 0) -> 2
        ('INC', 1) -> 3
        ('DEC', 0) -> 4
        ('DEC', 1) -> 5
        ('JNZ', 0, L) -> 6 + L
        ('JNZ', 1, L) -> 14 + L
    """
    op = inst[0]
    if op == 'PAD':
        return PAD_TOKEN
    elif op == 'HLT':
        return HLT_TOKEN
    elif op == 'INC':
        reg = inst[1]
        return INC_TOKEN_BASE + reg
    elif op == 'DEC':
        reg = inst[1]
        return DEC_TOKEN_BASE + reg
    elif op == 'JNZ':
        reg = inst[1]
        target = inst[2]
        # target must be within 0..7
        if not (0 <= target <= 7):
            raise ValueError(f"Jump target {target} out of range [0, 7]")
        return JNZ_TOKEN_BASE + reg * 8 + target
    raise ValueError(f"Unknown instruction operation: {inst}")

def token_to_inst(token):
    """Converts a token ID back to its instruction tuple representation."""
    if token == PAD_TOKEN:
        return ('PAD',)
    elif token == HLT_TOKEN:
        return ('HLT',)
    elif token < DEC_TOKEN_BASE:
        return ('INC', token - INC_TOKEN_BASE)
    elif token < JNZ_TOKEN_BASE:
        return ('DEC', token - DEC_TOKEN_BASE)
    elif token < 22:
        val = token - JNZ_TOKEN_BASE
        reg = val // 8
        target = val % 8
        return ('JNZ', reg, target)
    raise ValueError(f"Unknown token ID: {token}")

def execute_program(program, max_steps=500):
    """
    Simulates execution of a program represented as a list of instruction tuples.
    Registers R0, R1 are initialized to 0.
    
    Returns:
        halted (int): 1 if program hit HLT or jumped out-of-bounds, 0 if it timed out/looped.
        steps_taken (int): number of execution steps.
        trace (list of tuples): sequence of execution states (pc, R0, R1).
    """
    pc = 0
    R = [0, 0]
    n = len(program)
    trace = []
    visited = set()
    
    for step in range(max_steps):
        # Out-of-bounds PC halts
        if pc < 0 or pc >= n:
            return 1, step, trace
            
        state = (pc, R[0], R[1])
        if state in visited:
            # Infinite loop detected
            return 0, step, trace
        visited.add(state)
        trace.append(state)
        
        inst = program[pc]
        op = inst[0]
        if op == 'HLT':
            return 1, step + 1, trace
        elif op == 'INC':
            R[inst[1]] += 1
            pc += 1
        elif op == 'DEC':
            R[inst[1]] -= 1
            pc += 1
        elif op == 'JNZ':
            if R[inst[1]] != 0:
                pc = inst[2]
            else:
                pc += 1
                
    return 0, max_steps, trace
