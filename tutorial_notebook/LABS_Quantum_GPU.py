# Low Autocorrelation Binary Sequences (LABS) problem
import cudaq
import numpy as np
from math import floor, sin, cos, pi
import time
import csv
import os


# --- GPU Configuration ---
# specific target for NVIDIA GPUs. 
# If you run into issues, verify your installation with 'cudaq.get_target().name'
try:
    cudaq.set_target("nvidia")
    print(f"Target set to NVIDIA GPU.")
except:
    print("NVIDIA target not found. Falling back to CPU simulation (qpp-cpu).")
    cudaq.set_target("qpp-cpu")

# 1. Basic Rzz kernel used as a building block for multi-qubit rotations

@cudaq.kernel
def rzz(theta: float, q1: cudaq.qubit, q2: cudaq.qubit):
    cx(q1, q2)
    rz(theta, q2)
    cx(q1, q2)



# 2. Decomposition of the block of two-qubit rotations R_YZ(theta)R_ZY(theta)
# Based on Figure 3, requiring 2 entangling Rzz gates and 4 single-qubit gates
@cudaq.kernel
def two_qubit_rotation_block(theta: float, q0: cudaq.qubit, q1: cudaq.qubit):
    # Basis changes for YZ and ZY rotations
    rx(np.pi / 2, q1)
    rzz(theta, q0, q1)
    rx(np.pi / 2, q0)
    rx(-np.pi / 2, q1)
    rzz(theta, q0, q1)
    rx(-np.pi / 2, q0)



# 3. Decomposition of the four-qubit rotation block
# Based on Figure 4, requiring 10 entangling Rzz gates
@cudaq.kernel
def four_qubit_rotation_block(theta: float, q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit):
    # This kernel implements the sequence of Rzz and basis changes
    # for the 4-local terms in the LABS Hamiltonian
    # Layer 1: Basis changes
    rx(-np.pi/2, q0)
    ry(np.pi/2, q1)
    ry(-np.pi/2, q2)

    rzz(-np.pi/2,q0,q1)
    rzz(-np.pi/2,q2,q3)

    rx(np.pi/2, q0)
    ry(-np.pi/2, q1)
    ry(np.pi/2, q2)
    rx(-np.pi/2, q3)

    rx(-np.pi/2, q1)
    rx(-np.pi/2, q2)
   
    rzz(theta,q1,q2)

    rx(np.pi/2, q1)
    rx(np.pi  , q2)

    ry(np.pi/2, q1)

    rzz(np.pi/2,q0,q1)

    rx(np.pi/2, q0)
    ry(-np.pi/2, q1)

    rzz(-theta,q1,q2)

    rx(np.pi/2, q1)
    rx(-np.pi  , q2)

    rzz(-theta,q1,q2)

    rx(-np.pi, q1)
    ry(np.pi/2, q2)

    rzz(-np.pi/2,q2,q3)

    ry(-np.pi/2, q2)
    rx(-np.pi/2, q3)

    rx(-np.pi/2, q2)

    rzz(theta,q1,q2)

    rx(np.pi/2, q1)
    rx(np.pi/2, q2)
    
    ry(-np.pi/2, q1)
    ry(np.pi/2, q2)

    rzz(np.pi/2,q0,q1)
    rzz(np.pi/2,q2,q3)
   
    ry(np.pi/2, q1)
    ry(-np.pi/2, q2)
    rx(np.pi/2, q0)


@cudaq.kernel
def trotterized_circuit(N: int, G2: list[list[int]], G4: list[list[int]], steps: int, dt: float, T: float, thetas: list[float]):
    reg = cudaq.qvector(N)
    h(reg) # Initialize to ground state of Hi

    # Iterate through the Trotter steps
    for step_idx in range(steps):
        # Fetch precomputed theta for this time step
        current_theta = thetas[step_idx]

        # Apply the two-body terms defined in G2
        for pair in G2:
            two_qubit_rotation_block(4*current_theta*(step_idx*dt), reg[pair[0]], reg[pair[1]])
          
        # Apply the four-body terms defined in G4, taking h^x_i terms are all 1
        for quad in G4:
            four_qubit_rotation_block(8*current_theta*(step_idx*dt), reg[quad[0]], reg[quad[1]], reg[quad[2]], reg[quad[3]])
            four_qubit_rotation_block(8*current_theta*(step_idx*dt), reg[quad[3]], reg[quad[0]], reg[quad[1]], reg[quad[2]])
            four_qubit_rotation_block(8*current_theta*(step_idx*dt), reg[quad[2]], reg[quad[3]], reg[quad[0]], reg[quad[1]])
            four_qubit_rotation_block(8*current_theta*(step_idx*dt), reg[quad[1]], reg[quad[2]], reg[quad[3]], reg[quad[0]])


def get_interactions(N):
    """
    Generates the interaction sets G2 and G4 based on the loop limits in Eq. 2.
    Returns standard 0-based indices as lists of lists of ints.   
    Args:
        N (int): Sequence length.
    Returns:
        G2: List of lists containing two body term indices
        G4: List of lists containing four body term indices
    """
    G2 = []
    G4 = []

    # Two-body interactions: 2 * sum_{i=1}^{N-2} sum_{k=1}^{floor((N-i)/2)}
    for i in range(1, N - 1): # i = 1 to N-2
        limit_k = (N - i) // 2
        for k in range(1, limit_k + 1):
            # Adjust to 0-based indexing: i-1, i+k-1
            G2.append([i - 1, i + k - 1])

    # Four-body interactions: 4 * sum_{i=1}^{N-3} sum_{t=1}^{floor((N-i-1)/2)} sum_{k=t+1}^{N-i-t}
    for i in range(1, N - 2): # i = 1 to N-3
        limit_t = (N - i - 1) // 2
        for t in range(1, limit_t + 1):
            for k in range(t + 1, N - i - t + 1):
                # Indices in terms of i, t, k are: i, i+t, i+k, i+k+t
                # Adjust to 0-based indexing: i-1, i+t-1, i+k-1, i+k+t-1
                G4.append([i - 1, i + t - 1, i + k - 1, i + k + t - 1])

    return G2, G4





def compute_topology_overlaps(G2, G4):
    """
    Computes the topological invariants I_22, I_24, I_44 based on set overlaps.
    I_alpha_beta counts how many sets share IDENTICAL elements.
    """
    # Helper to count identical sets
    def count_matches(list_a, list_b):
        matches = 0
        # Convert to sorted tuples to ensure order doesn't affect equality
        set_b = set(tuple(sorted(x)) for x in list_b)
        for item in list_a:
            if tuple(sorted(item)) in set_b:
                matches += 1
        return matches

    # For standard LABS/Ising chains, these overlaps are often 0 or specific integers
    # We implement the general counting logic here.
    I_22 = count_matches(G2, G2) # Self overlap is just len(G2)
    I_44 = count_matches(G4, G4) # Self overlap is just len(G4)
    I_24 = 0 # 2-body set vs 4-body set overlap usually 0 as sizes differ
    
    return {'22': I_22, '44': I_44, '24': I_24}


def compute_theta(t, dt, total_time, N, G2, G4):
    """
    Computes theta(t) using the analytical solutions for Gamma1 and Gamma2.
    """
    
    # ---  Better Schedule (Trigonometric) ---
    # lambda(t) = sin^2(pi * t / 2T)
    # lambda_dot(t) = (pi / 2T) * sin(pi * t / T)
    
    if total_time == 0:
        return 0.0

    # Argument for the trig functions
    arg = (pi * t) / (2.0 * total_time)
    
    lam = sin(arg)**2
    # Derivative: (pi/2T) * sin(2 * arg) -> sin(pi * t / T)
    lam_dot = (pi / (2.0 * total_time)) * sin((pi * t) / total_time)
    
    
    # ---  Calculate Gamma Terms (LABS assumptions: h^x=1, h^b=0) ---
    # For G2 (size 2): S_x = 2
    # For G4 (size 4): S_x = 4
    
    # Gamma 1 (Eq 16)
    # Gamma1 = 16 * Sum_G2(S_x) + 64 * Sum_G4(S_x)
    term_g1_2 = 16 * len(G2) * 2
    term_g1_4 = 64 * len(G4) * 4
    Gamma1 = term_g1_2 + term_g1_4
    
    # Gamma 2 (Eq 17)
    # G2 term: Sum (lambda^2 * S_x)
    # S_x = 2
    sum_G2 = len(G2) * (lam**2 * 2)
    
    # G4 term: 4 * Sum (4*lambda^2 * S_x + (1-lambda)^2 * 8)
    # S_x = 4
    # Inner = 16*lam^2 + 8*(1-lam)^2
    sum_G4 = 4 * len(G4) * (16 * (lam**2) + 8 * ((1 - lam)**2))
    
    # Topology part
    I_vals = compute_topology_overlaps(G2, G4)
    term_topology = 4 * (lam**2) * (4 * I_vals['24'] + I_vals['22']) + 64 * (lam**2) * I_vals['44']
    
    # Combine Gamma 2
    Gamma2 = -256 * (term_topology + sum_G2 + sum_G4)

    # ---  Alpha & Theta ---
    if abs(Gamma2) < 1e-12:
        alpha = 0.0
    else:
        alpha = - Gamma1 / Gamma2
        
    return dt * alpha * lam_dot



def energy_function(bitstring):
    """LABS Energy: Sum of squared autocorrelations (excluding peak at 0)."""
    # Convert '0'/'1' string to -1/+1 spins
    spin_s = [1 if bit == '1' else -1 for bit in bitstring]
    # Note: cudaq usually returns '10...' strings. 
    # Standard mapping 1->-1, 0->1 or vice versa doesn't change Energy magnitude 
    # but let's stick to standard: 0 -> 1, 1 -> -1 (or similar)
    
    N = len(spin_s)
    energy = 0
    for k in range(1, N):
        c_k = 0
        for i in range(N - k):
            c_k += spin_s[i] * spin_s[i+k]
        energy += c_k**2
    return energy


def calculate_merit_factor(N, energy):
    if energy == 0:
        return float('inf') # Perfect code, rare/impossible for large N
    return (N**2) / (2 * energy)


# --- Main Execution Loop ---

def run_labs_sweep(min_N, max_N, filename="labs_quantum_results.csv"):
    # Simulation Parameters
    T = 1.0            
    n_steps = 20      # Increased steps for better accuracy
    dt = T / n_steps
    shots = 2000      # Number of shots per N

    # Prepare CSV file
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["N", "Computation_Time_s", "Best_Energy", "Merit_Factor", "Best_Sequence"])

    print(f"Starting LABS sweep for N=[{min_N} ... {max_N}] on {cudaq.get_target().name}...")
    print("-" * 60)

    for N in range(min_N, max_N + 1):
        try:
            # 1. Setup Problem
            G2, G4 = get_interactions(N)
            
            # 2. Compute Schedule
            thetas = []
            for step in range(1, n_steps + 1):
                t = step * dt
                theta_val = compute_theta(t, dt, T, N, G2, G4)
                thetas.append(theta_val)

            # 3. Run Quantum Circuit (with timing)
            start_time = time.time()
            
            # Execute Sample
            # Note: G2 and G4 are lists of lists, cudaq handles this serialization
            counts = cudaq.sample(trotterized_circuit, N, G2, G4, n_steps, dt, T, thetas, shots_count=shots)
            
            end_time = time.time()
            comp_time = end_time - start_time

            # 4. Process Results
            best_energy = float('inf')
            best_sequence = ""

            # Iterate over observed bitstrings
            for bitstring, count in counts.items():
                current_E = energy_function(bitstring)
                if current_E < best_energy:
                    best_energy = current_E
                    best_sequence = bitstring
            
            merit_factor = calculate_merit_factor(N, best_energy)

            # 5. Save and Print
            with open(filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([N, f"{comp_time:.4f}", best_energy, f"{merit_factor:.4f}", best_sequence])

            print(f"N={N:02d} | Time: {comp_time:.3f}s | Energy: {best_energy:4.0f} | MF: {merit_factor:6.2f} | Seq: {best_sequence}")

        except Exception as e:
            print(f"Error processing N={N}: {e}")

# --- Run ---
if __name__ == "__main__":
    # Example range: N=5 to N=15
    # Be cautious with large N (>25) as simulation time grows exponentially
    run_labs_sweep(min_N=5, max_N=15)
