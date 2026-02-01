# Technical Report: Hybrid Quantum-Enhanced Optimization for the LABS Problem (NVIDIA iQuHACK 2026)
Team: SOLO
Member: Musfar Muhamed Kozhikkal
1. The Challenge: Optimizing Radar and Telecommunication Signals

The Low Autocorrelation Binary Sequences (LABS) problem is a critical optimization bottleneck in radar and telecommunications engineering. Strategically, this challenge arises from a necessary "balancing act" in signal design: long pulses are required to maximize the signal-to-noise ratio (SNR) and detection range, yet short pulses are essential to maintain high range resolution. Pulse compression resolves this by phase-encoding a long signal—typically using binary shifts of 0^\circ or 180^\circ—which requires finding sequences with minimal autocorrelation sidelobes to prevent false detections and signal interference.

Mathematically, we seek a binary sequence s = (s_1, \dots, s_N) \in \{-1, 1\}^N that minimizes the total energy function E(s), defined as the sum of squared autocorrelations:

E(s) = \sum_{k=1}^{N-1} C_k^2

where the autocorrelation coefficient C_k for a lag k is defined as:

C_k = \sum_{i=1}^{N-k} s_i s_{i+k}

The computational complexity of LABS is notoriously difficult. With a configuration space of 2^N and a landscape fraught with underlying symmetries and degeneracies, local search methods are easily trapped in sub-optimal states. While the classical state-of-the-art method is currently Memetic Tabu Search (MTS), its scaling remains an intractable barrier for large-scale signal optimization.

2. Classical Benchmark: Memetic Tabu Search (MTS) on CPU and GPU

Memetic Tabu Search (MTS) is a hybrid metaheuristic that bridges global population-based evolution with intensive local refinement. By maintaining a population of bitstrings, the algorithm utilizes crossover and mutation operators to explore the configuration space, subsequently applying a Tabu Search to each individual. This local search employs a "tabu list" to avoid cycles and encourage the exploration of new regions in the degenerate LABS landscape.

Benchmarking the MTS performance across CPU and NVIDIA CUDA environments reveals a performance paradox at smaller sequence lengths. As shown in the table below, CUDA execution times are consistently higher than CPU times for the tested range.

Sequence Length (N)	Device	Execution Time (s)	Best Energy Found	Merit Factor (F)
10	CPU	8.7932	13	3.8462
10	CUDA	17.1938	13	3.8462
30	CPU	19.1520	59	7.6271
30	CUDA	39.0802	59	7.6271
60	CPU	33.6951	282	6.3830
60	CUDA	67.5678	314	5.7325

HPC Note: The slower performance on CUDA for these values of N is attributed to kernel launch overhead and low GPU occupancy. For these relatively small problem sizes, the overhead of managing device memory and launching parallel kernels outweighs the architectural benefits of GPU acceleration. However, the Merit Factor (F = N^2 / 2E) serves as our primary success metric. Analysis of F reveals high volatility for small N, notably spiking at N=13 (the Barker sequence) which represents a global maximum for F in the small-N regime. As N approaches 60, F stabilizes, underscoring the O(1.34^N) scaling limitation of classical MTS.

3. Quantum Strategy: Counteradiabatic (CD) Optimization with CUDA-Q

To break the classical scaling limit, we transition to a hybrid quantum-enhanced workflow. Here, the quantum subroutine is not intended to find the final global minimum, but rather to act as a "seeding" mechanism. By sampling from a quantum-optimized state, we can initialize the MTS population with high-quality candidates that are statistically superior to random initialization.

This implementation utilizes a Counteradiabatic (CD) strategy in CUDA-Q. By introducing an adiabatic gauge potential A_\lambda, we suppress diabatic transitions that typically plague fast adiabatic evolution. In the "impulse regime," where evolution is rapid, the CD term H_{CD} dominates the adiabatic Hamiltonian. Our implementation utilizes a first-order approximation of A_\lambda to evolve the system toward the ground state of the LABS Hamiltonian H_f:

H_f = 2 \sum_{i=1}^{N-2} \sigma_i^z \sum_{k=1}^{\lfloor \frac{N-i}{2} \rfloor} \sigma_{i+k}^z + 4 \sum_{i=1}^{N-3} \sigma_i^z \sum_{t=1}^{\lfloor \frac{N-i-1}{2} \rfloor} \sum_{k=t+1}^{N-i-t} \sigma_{i+t}^z \sigma_{i+k}^z \sigma_{i+k+t}^z

This CD approach is significantly more efficient than standard QAOA. For N=67, QAOA requires approximately 1.4 million entangling gates, whereas the CD logic reduces this to only 236,000 gates. The implementation relies on high-performance quantum kernels:

* rzz: The core entangling gate, implementing a two-qubit rotation.
* two_qubit_rotation_block: Implements R_{YZ} and R_{ZY} rotations through basis changes and rzz gates.
* four_qubit_rotation_block: A sophisticated kernel implementing 4-local terms via ten entangling rzz gates and RX/RY basis changes.

4. Phase 2 Comparative Analysis: CPU vs. GPU vs. CUDA-Q Simulation

Phase 2 testing was conducted on NVIDIA A100 GPUs via the Brev platform to evaluate the acceleration of both the quantum simulation and the classical search. While the A100 provides immense throughput, the simulation of quantum logic encounters a steep "Memory Wall."

As N increases, the simulation time jumps exponentially, roughly doubling with every increment of N. For instance, simulation time rose from 233.1s at N=29 to 507.5s at N=30. This follows the O(2^N) scaling of the state-vector representation. At N=31, the system encountered a failure ("requested size is too big"), as the 2^{31} complex entries required for the state vector exceeded the memory capacity and kernel overhead limits of a single 80GB A100.

Despite simulation costs, the quality of "Quantum Seeds" is demonstrably superior. For N=10, the energy distribution of random seeds is broad and centered at high energy values. In contrast, the Quantum-Seeded (DCQO) distribution, as seen in output_20_1.png, peaks significantly lower in the 25–50 energy range. This shift provides the classical MTS with a statistically advantaged starting point, effectively placing the search much closer to the global minimum from the outset.

5. Software Rigor: Validation and Testing Framework

In an AI-assisted "Vibe Coding" environment, human-led verification is paramount. We implemented a rigorous pytest suite to ensure the mathematical and logical integrity of our hybrid workflow.

The validation framework included:

* Mathematical Correctness: We validated energy calculations against known Barker sequences. The system correctly returned E=1.0 for N=3 (++-) and E=14.0 for an all-ones N=4 sequence.
* Genetic Operators: Unit tests verified that crossover and mutation logic maintained sequence length and correctly flipped bits.
* Quantum Logic: Mocked integration tests ensured cudaq sampling and interaction generation were logically sound before deployment on GPU hardware.

A critical "Bug Fix" phase occurred when the test suite identified a 0-based indexing error in the Hamiltonian interaction generation. The logic originally used i=1 \dots N bounds, which caused misalignment in the CUDA-Q kernels; the suite facilitated a shift to i=0 \dots N-1, leading to the "test approved" status required for Phase 2 deployment.

6. Conclusion and Strategic Outlook

The NVIDIA iQuHACK 2026 challenge successfully validated the hybrid Quantum-Enhanced Memetic Tabu Search (QE-MTS) as a viable path for solving the LABS problem. Our findings demonstrate that while classical MTS remains faster for small-scale instances, QE-MTS offers a superior scaling trajectory.

The QE-MTS approach is projected to achieve O(1.24^N) scaling, which significantly outperforms the O(1.46^N) scaling of standard QAOA and the O(1.34^N) limit of classical MTS. As sequence lengths grow beyond N=47—the identified Quantum Advantage Horizon—QE-MTS is expected to surpass classical methods by providing high-quality initial populations that are unreachable through classical heuristics. While current state-vector simulation is constrained by the Memory Wall, this architectural framework is fully ready for deployment on future QPU hardware to solve the next generation of telecommunications challenges.
