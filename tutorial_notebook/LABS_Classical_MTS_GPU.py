# Low Autocorrelation Binary Sequences (LABS) problem
# Solution  using classical optimization technique is Memetic Tabu search (MTS)
# Need to find the minimum energy case

import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

# --- Configuration ---
RESULTS_FILE = "labs_mts_results_classical.csv"
RESULTS_FILE_late = "labs_mts_results_classical_latest3.csv"

class MTS_Solver_Torch:
    def __init__(self, N, pop_size=20, p_mut=0.05, max_iters=500, ts_max_iters=100, device='cpu'):
        self.N = N
        self.pop_size = pop_size
        self.p_mut = p_mut
        self.max_iters = max_iters
        self.ts_max_iters = ts_max_iters
        self.device = torch.device(device)
        
        # Initialize population: Shape (pop_size, N)
        # We use float for easier matrix math later (0.0 and 1.0)
        self.population = torch.randint(0, 2, (self.pop_size, self.N), device=self.device).float()
        self.pop_energies = self.calculate_energy_batch(self.population)
        
        # Track global best
        best_idx = torch.argmin(self.pop_energies)
        self.best_s = self.population[best_idx].clone()
        self.best_e = self.pop_energies[best_idx].item()

    def calculate_energy_batch(self, sequences):
        """
        Vectorized energy calculation.
        Input: sequences tensor of shape (Batch_Size, N) containing 0s and 1s.
        Output: energies tensor of shape (Batch_Size,)
        """
        # Convert {0, 1} to {-1, 1}
        spins = 2 * sequences - 1 
        
        batch_size, n = spins.shape
        energies = torch.zeros(batch_size, device=self.device)
        
        # Calculate autocorrelations for all k (1 to N-1) in parallel for the batch
        for k in range(1, n):
            # Element-wise multiplication of shifted arrays
            # spins[:, :n-k] corresponds to s[i]
            # spins[:, k:]   corresponds to s[i+k]
            # Sum along dimension 1 (the sequence length) to get C_k
            c_k = (spins[:, :n-k] * spins[:, k:]).sum(dim=1)
            energies += c_k ** 2
            
        return energies

    def combine(self, p1, p2):
        k = torch.randint(1, self.N - 1, (1,)).item()
        return  torch.cat((p1[:k], p2[k:]))
        

    def mutate(self, s):
        # Generate a mask of bits to flip based on probability p_mut
        mask = (torch.rand(self.N, device=self.device) < self.p_mut).float()
        # XOR operation logic: (s + mask) % 2 handles the flip (0->1, 1->0)
        # Using abs() to handle potential -1s if we were using spins, but here we use 0/1
        return torch.abs(s - mask)

    def tabu_search(self, s_init):
        curr_s = s_init.clone()
        curr_e = self.calculate_energy_batch(curr_s.unsqueeze(0)).item()
        
        best_local_s = curr_s.clone()
        best_local_e = curr_e
        
        tabu_list = torch.zeros(self.N, device=self.device)
        min_tabu = self.N // 10
        extra_tabu = self.N // 50

        for k in range(self.ts_max_iters):
            # --- Vectorized Neighbor Generation ---
            # Create a batch of neighbors where each row has 1 bit flipped
            # Identity matrix represents the bit to flip for each neighbor
            flip_mask = torch.eye(self.N, device=self.device)
            
            # Expand current sequence to shape (N, N) and apply flips
            # shape: (N_neighbors, N_length)
            neighbors = torch.abs(curr_s.unsqueeze(0).repeat(self.N, 1) - flip_mask)
            
            # Calculate energy of ALL neighbors at once
            neighbor_energies = self.calculate_energy_batch(neighbors)
            
            # --- Filter Admissible Moves ---
            # Check Tabu status
            is_tabu = tabu_list > k
            
            # Aspiration criterion: Allow tabu if it beats best local score
            aspiration = neighbor_energies < best_local_e
            
            # Valid moves are (Not Tabu) OR (Aspiration met)
            is_valid = (~is_tabu) | aspiration
            
            # We want to minimize energy. Set invalid moves to infinity so they aren't picked
            masked_energies = neighbor_energies.clone()
            masked_energies[~is_valid] = float('inf')
            
            # Find best neighbor
            best_neighbor_e, best_neighbor_idx = torch.min(masked_energies, dim=0)
            
            if best_neighbor_e == float('inf'):
                break # No moves possible
            
            # Move
            best_neighbor_idx = best_neighbor_idx.item()
            curr_s = neighbors[best_neighbor_idx].clone()
            curr_e = best_neighbor_e.item()
            
            # Update Tabu List
            tenure = min_tabu + torch.randint(0, max(1, extra_tabu), (1,)).item()
            tabu_list[best_neighbor_idx] = k + tenure
            
            # Update Local Best
            if curr_e < best_local_e:
                best_local_e = curr_e
                best_local_s = curr_s.clone()
                
        return best_local_s, best_local_e

    def solve(self):
        iteration = 0
        while iteration < self.max_iters:
            # 1. Crossover / Selection
            if torch.rand(1).item() < 0.5:
                # Random selection
                idx = torch.randint(0, self.pop_size, (1,)).item()
                child = self.population[idx].clone()
            else:
                # Crossover
                idxs = torch.randperm(self.pop_size)[:2]
                child = self.combine(self.population[idxs[0]], self.population[idxs[1]])
            
            # 2. Mutation
            child = self.mutate(child)
            
            # 3. Tabu Search
            refined_s, refined_e = self.tabu_search(child)
            
            # 4. Update Global Best
            if refined_e < self.best_e:
                self.best_e = refined_e
                self.best_s = refined_s.clone()
            
            # 5. Update Population (Simple Replacement)
            replace_idx = torch.randint(0, self.pop_size, (1,)).item()
            if refined_e < self.pop_energies[replace_idx]:
                self.population[replace_idx] = refined_s
                self.pop_energies[replace_idx] = refined_e
            
            iteration += 1
            
        return self.best_s, self.best_e

# --- Experiment Runner ---

def run_experiment(n_start=7, n_end=30, devices_to_test=['cpu']):
    results = []
    
    # Check if CUDA is actually available if requested
    if 'cuda' in devices_to_test and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Skipping GPU tests.")
        devices_to_test.remove('cuda')

    if not os.path.exists(RESULTS_FILE):
        header = pd.DataFrame(columns=['N', 'Device', 'Time_Seconds', 'Best_Energy', 'Merit_Factor', 'Best_Sequence'])
        header.to_csv(RESULTS_FILE, index=False)

    print(f"Starting Experiment: N={n_start} to {n_end}")
    print(f"Testing on: {devices_to_test}")
    print("-" * 60)

    for N in range(n_start, n_end + 1):
        for device in devices_to_test:
            print(f"Running N={N} on {device}...", end="\r")
            
            start_time = time.time()
            
            # Instantiate and Solve
            # Note: For strict benchmarking, we might want fixed seeds, 
            # but for optimization search, randomness is key.
            solver = MTS_Solver_Torch(N=N, pop_size=30, max_iters=200, device=device)
            best_seq_tensor, best_energy = solver.solve()
            
            elapsed_time = time.time() - start_time
            
            # Convert sequence to string
            best_seq_str = ''.join(str(int(x)) for x in best_seq_tensor.cpu().numpy())
            merit_factor = (N**2) / (2 * best_energy) if best_energy > 0 else 0
            
            # --- SAVE IMMEDIATELY AFTER THIS N ---
            row = {
                'N': N,
                'Device': device,
                'Time_Seconds': round(elapsed_time, 4),
                'Best_Energy': int(best_energy),
                'Merit_Factor': round(merit_factor, 4),
                'Best_Sequence': best_seq_str
            }
            
            row_df = pd.DataFrame([row])
            
            # Append to file without writing header
            row_df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
            print(f"Done! Saved to {RESULTS_FILE}")
            results.append(row)
            
        print(f"Finished N={N}. Best E (last run): {best_energy}")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE_late, index=False)
    print("-" * 60)
    print(f"Experiment Complete. Results saved to {RESULTS_FILE_late}")
    return df

# --- Execution ---

if __name__ == "__main__":
    # Define devices: ['cpu'] or ['cpu', 'cuda']
    # If you have a GPU, change this to: devices = ['cpu', 'cuda']
    devices = ['cpu']#, 'cuda']
    if torch.cuda.is_available():
        devices.append('cuda')
        print("NVIDIA GPU detected. Enabling CUDA tests.")
        
    df_results = run_experiment(n_start=51, n_end=60, devices_to_test=devices)

    # Simple Plotting of Time vs N
    plt.figure(figsize=(10, 6))
    for device in df_results['Device'].unique():
        subset = df_results[df_results['Device'] == device]
        plt.plot(subset['N'], subset['Time_Seconds'], marker='o', label=f'Device: {device}')
    
    plt.title('Computation Time vs Sequence Length (N)')
    plt.xlabel('Sequence Length N')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig("LABS_Classical_MTS_time_vs_N.jpeg")
    #plt.show()

    # Simple Plotting of Merit Factor vs N (for one device)
    plt.figure(figsize=(10, 6))
    #subset = df_results[df_results['Device'] == df_results['Device'].unique()[0]]
    #plt.plot(subset['N'], subset['Merit_Factor'], marker='x', color='green', label='Merit Factor')

    for device in df_results['Device'].unique():
        subset = df_results[df_results['Device'] == device]
        plt.plot(subset['N'], subset['Merit_Factor'], marker='x', label=f'Device: {device}')
    plt.title('Best Merit Factor Found vs Sequence Length (N)')
    plt.xlabel('Sequence Length N')
    plt.ylabel('Merit Factor (N^2 / 2E)')
    plt.legend()
    plt.grid(True)
    plt.savefig("LABS_Classical_MTS_merit_vs_N.jpeg")
    plt.show()
