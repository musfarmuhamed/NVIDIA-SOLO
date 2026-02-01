import pytest
import numpy as np
import os
import math
from unittest.mock import patch, MagicMock

# Import the specific functions from your file
# Ensure LABS_Quantum_GPU.py is in the Python path
from LABS_Quantum_GPU import (
    get_interactions,
    compute_topology_overlaps,
    compute_theta,
    energy_function,
    calculate_merit_factor,
    run_labs_sweep
)

# --- 1. Classical Logic Tests (Deterministic) ---

def test_energy_function_known_cases():
    """
    Test the LABS energy function against known sequences.
    Energy E(S) = sum_{k=1}^{N-1} (C_k)^2
    """
    # Case 1: Barker Sequence N=3 (+ + -) -> 1 1 0 (or similar)
    # Spins: 1, 1, -1
    # C1 = 1(1) + 1(-1) = 0
    # C2 = 1(-1) = -1
    # E = 0^2 + (-1)^2 = 1
    assert energy_function("110") == 1
    # Check if input mapping (0/1) consistency doesn't break logic
    # If 1->1, 0->-1: 1 1 -1 result is same.
    
    # Case 2: N=4 All Ones (1 1 1 1) -> "1111"
    # Spins: 1 1 1 1
    # C1=3, C2=2, C3=1 -> E = 9 + 4 + 1 = 14
    assert energy_function("1111") == 14

def test_merit_factor_calculation():
    """
    Test Merit Factor Formula: F = N^2 / (2 * E)
    """
    # N=3, E=1 -> MF = 9 / 2 = 4.5
    assert calculate_merit_factor(3, 1) == 4.5
    
    # N=4, E=14 -> MF = 16 / 28 = 0.5714...
    mf = calculate_merit_factor(4, 14)
    assert math.isclose(mf, 16/28, rel_tol=1e-5)
    
    # Zero Energy Case (Perfect sequence)
    assert calculate_merit_factor(10, 0) == float('inf')

def test_interaction_generation_n5():
    """
    Verify G2 and G4 generation for a small N=5 case.
    Eq 2 implies specific loop bounds.
    """
    N = 5
    G2, G4 = get_interactions(N)
    
    # G2 elements are pairs [i, j].
    # Bounds Check: indices must be between 0 and N-1
    for pair in G2:
        assert len(pair) == 2
        assert max(pair) < N
        assert min(pair) >= 0
        
    # G4 elements are quads [i, j, k, l]
    for quad in G4:
        assert len(quad) == 4
        assert max(quad) < N
        
    # Check types
    assert isinstance(G2, list)
    assert isinstance(G4, list)

def test_topology_overlaps():
    """
    Test the counting logic for I_22, I_44.
    """
    # Create artificial sets
    mock_G2 = [[0, 1], [1, 2], [0, 1]] # [0,1] repeats
    mock_G4 = [[0, 1, 2, 3]]
    
    # Note: The function in source logic counts matches of list_a in list_b.
    # If G2 is passed as both args, it counts self-overlaps.
    
    # However, standard set logic usually implies unique sets, 
    # but your code: 'count_matches' iterates list_a and checks if in set_b.
    overlaps = compute_topology_overlaps(mock_G2, mock_G4)
    
    # I_22: How many items in mock_G2 exist in set(mock_G2)?
    # set(mock_G2) has {[0,1], [1,2]}.
    # Iterating mock_G2:
    # [0,1] -> exists -> +1
    # [1,2] -> exists -> +1
    # [0,1] -> exists -> +1
    # Total 3.
    assert overlaps['22'] == 3
    assert overlaps['44'] == 1
    assert overlaps['24'] == 0 

def test_compute_theta_basic():
    """
    Ensure compute_theta returns a float and handles t=0 correctly.
    """
    N = 5
    G2, G4 = get_interactions(N)
    dt = 0.1
    T = 1.0
    
    # At t=0
    theta_0 = compute_theta(0, dt, T, N, G2, G4)
    # sin(0) = 0 -> lambda=0 -> lam_dot=0 -> returns 0
    assert theta_0 == 0.0
    
    # At t=0.5
    theta_mid = compute_theta(0.5, dt, T, N, G2, G4)
    assert isinstance(theta_mid, float)
    # Should not be NaN
    assert not math.isnan(theta_mid)

# --- 2. Mocked Integration Tests (No Quantum Hardware needed) ---

@patch('LABS_Quantum_GPU.cudaq')
@patch('builtins.print') # Suppress print output
@patch('os.path.isfile') # Mock file existence check
@patch('builtins.open')  # Mock file writing
def test_run_labs_sweep_mocked(mock_open, mock_isfile, mock_print, mock_cudaq):
    """
    Test the full loop logic without running actual quantum kernels.
    """
    # 1. Setup Mock for cudaq.sample
    # It returns a dictionary of {bitstring: count}
    mock_counts = {
        "11111": 100, # High Energy
        "10101": 50   # Lower Energy
    }
    mock_cudaq.sample.return_value = mock_counts
    
    # Mock target name
    mock_cudaq.get_target().name = "mock-qpp-cpu"

    # 2. Run Sweep for small range
    # Run only for N=5
    run_labs_sweep(min_N=5, max_N=5, filename="test_results.csv")
    
    # 3. Assertions
    # Ensure sample was called
    assert mock_cudaq.sample.called
    
    # Verify file operations
    # Should check if file exists
    assert mock_isfile.called
    # Should open file for writing results
    assert mock_open.called

# --- 3. Quantum Kernel Declaration Test (Structure Only) ---

def test_kernel_structures():
    """
    We cannot execute kernels easily without the runtime, 
    but we can verify they are decorated functions in the imported module.
    """
    from LABS_Quantum_GPU import rzz, two_qubit_rotation_block, four_qubit_rotation_block
    
    # In a standard python environment, these appear as cudaq Kernel objects 
    # or compiled objects depending on installation.
    # We just check they exist.
    assert rzz is not None
    assert two_qubit_rotation_block is not None
    assert four_qubit_rotation_block is not None