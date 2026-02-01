import pytest
import torch
import os
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the class from your specific filename
# Ensure LABS_Classical_MTS_GPU.py is in the same folder or python path
from LABS_Classical_MTS_GPU import MTS_Solver_Torch, run_experiment

# --- Fixtures ---

@pytest.fixture
def cpu_device():
    return 'cpu'

@pytest.fixture
def solver_n5(cpu_device):
    """Returns a solver instance with N=5 for basic logic testing."""
    torch.manual_seed(42) # Fix seed for reproducibility
    return MTS_Solver_Torch(N=5, pop_size=10, device=cpu_device)

# --- Mathematical Correctness Tests ---

def test_energy_calculation_barker_n3(cpu_device):
    """
    Test energy calculation against a known Barker sequence.
    Barker N=3 sequence: ++- (1, 1, 0)
    Spins: 1, 1, -1
    Autocorrelations:
    k=1: (1)(1) + (1)(-1) = 0
    k=2: (1)(-1) = -1
    Energy = 0^2 + (-1)^2 = 1
    """
    solver = MTS_Solver_Torch(N=3, device=cpu_device)
    
    # Input tensor shape (Batch, N)
    # Using float because code casts to float in init
    test_seq = torch.tensor([[1.0, 1.0, 0.0]], device=cpu_device) 
    
    energy = solver.calculate_energy_batch(test_seq)
    
    assert energy.item() == 1.0, f"Expected Energy 1.0 for Barker N=3, got {energy.item()}"

def test_energy_calculation_n4(cpu_device):
    """
    Test N=4 sequence: 1 1 1 1 (all +1 spins)
    Spins: 1 1 1 1
    C1 = 1+1+1 = 3
    C2 = 1+1 = 2
    C3 = 1
    Energy = 3^2 + 2^2 + 1^2 = 9 + 4 + 1 = 14
    """
    solver = MTS_Solver_Torch(N=4, device=cpu_device)
    test_seq = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=cpu_device)
    
    energy = solver.calculate_energy_batch(test_seq)
    
    assert energy.item() == 14.0, f"Expected Energy 14.0 for all ones N=4, got {energy.item()}"

def test_energy_batch_processing(cpu_device):
    """Ensure the vectorized batch calculation works for multiple rows."""
    solver = MTS_Solver_Torch(N=3, device=cpu_device)
    # Batch of 2 sequences: [++-] (E=1) and [+++] (E=3 for N=3: C1=2, C2=1 -> 4+1=5)
    batch = torch.tensor([
        [1.0, 1.0, 0.0], # E=1
        [1.0, 1.0, 1.0]  # E=5
    ], device=cpu_device)
    
    energies = solver.calculate_energy_batch(batch)
    
    assert torch.allclose(energies, torch.tensor([1.0, 5.0], device=cpu_device))

# --- Genetic Operator Tests ---

def test_initialization_shapes(solver_n5):
    """Check population shape and value range."""
    assert solver_n5.population.shape == (10, 5)
    # Check values are only 0 or 1
    unique_vals = torch.unique(solver_n5.population)
    assert torch.all( (unique_vals == 0) | (unique_vals == 1) )

def test_crossover_logic(solver_n5):
    """Test that combine produces a child of correct length from parents."""
    p1 = torch.zeros(5)
    p2 = torch.ones(5)
    
    child = solver_n5.combine(p1, p2)
    
    assert child.shape == (5,)
    # Child should be a mix of 0s and 1s (unless cut point is 0 or N, but code restricts to 1..N-1)
    assert torch.sum(child) > 0 and torch.sum(child) < 5

def test_mutation_logic(solver_n5):
    """Test that mutation flips bits."""
    # Force mutation probability to 1.0
    solver_n5.p_mut = 1.0
    original = torch.zeros(5) # [0,0,0,0,0]
    
    mutated = solver_n5.mutate(original)
    
    # If p_mut=1, all 0s should become 1s
    assert torch.equal(mutated, torch.ones(5))

# --- Tabu Search Logic Tests ---

def test_tabu_search_improvement(solver_n5):
    """Ensure Tabu Search returns a result <= input energy."""
    # Create a random sequence
    input_s = torch.randint(0, 2, (5,)).float()
    input_e = solver_n5.calculate_energy_batch(input_s.unsqueeze(0)).item()
    
    best_s, best_e = solver_n5.tabu_search(input_s)
    
    assert best_e <= input_e, "Tabu search should not make the solution worse"
    assert best_s.shape == (5,)

# --- Integration / Workflow Tests ---

def test_full_solve_run(solver_n5):
    """Run a short full solve loop to ensure no runtime errors."""
    solver_n5.max_iters = 5 # Short run
    solver_n5.ts_max_iters = 5
    
    best_s, best_e = solver_n5.solve()
    
    assert best_s.shape == (5,)
    assert isinstance(best_e, float)

@patch('pandas.DataFrame.to_csv')
@patch('builtins.print') # Suppress print output
def test_experiment_runner(mock_print, mock_to_csv):
    """
    Test the high-level run_experiment function.
    Mocks file writing to avoid creating CSV files during tests.
    """
    # Run a tiny experiment
    df = run_experiment(n_start=5, n_end=6, devices_to_test=['cpu'])
    
    assert 'N' in df.columns
    assert 'Best_Energy' in df.columns
    assert len(df) == 2 # N=5 and N=6
    
    # Verify we attempted to save results
    assert mock_to_csv.called

# --- GPU Availability Test (Conditional) ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_computation():
    """Only runs if GPU is detected."""
    N = 10
    solver = MTS_Solver_Torch(N=N, device='cuda')
    assert str(solver.device).startswith('cuda')
    
    seq = torch.randint(0, 2, (1, N), device='cuda').float()
    energy = solver.calculate_energy_batch(seq)
    
    assert energy.is_cuda