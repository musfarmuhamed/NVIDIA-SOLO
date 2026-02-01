# Product Requirements Document (PRD)

**Project Name:** SOLO-Nv
**Team Name:** SOLO
**GitHub Repository:** https://github.com/musfarmuhamed/NVIDIA-SOLO

---

> **Note to Students:** > The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have a solid plan, not just to fill in boxes.

---

## 1. Team Roles & Responsibilities 

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **All in All**  | [Musfar Muhamed Kozhikkal] | [@musfarmuhamed] | [@original_wumpu] |

I planned the participate this hackathan with 3 of my friends. Since the holidays started, all of them made other plans and went to other places to visit. Though they said they will join me. I was waiting for them to join online, meanwhile I started this on my own. They havent responded to me after the hackathon began. I didnt asked around the discord for other members since I thought my teammates will join. Hence I am doing all this as a SOLO project.
---

## 2. The Architecture

### Choice of Quantum Algorithm
* **Algorithm:** Variational Quantum Eigensolver (VQE) like solver with `cudaq.sample` .

* **Motivation:** I chose this in this case since this the easiest to write. When I thinks in the kernel, I am getting some unexpected errors. Solving those errors takes time, which I dont have. Hence I went with the simplest and efficient error free solution. Some other simple solutions I tried wasnt that much efficient. 
   

### Literature Review
Havent done much Liternature review now, since I have good knowledge about this from my regular reading for articles. I spend most of my time reading the documentation of `cudaq`, since this is my first with it. I had used QAOA and other algorthims before, hence didnt spend much more time today on it.

---

## 3. The Acceleration Strategy

### Quantum Acceleration (CUDA-Q)
* **Strategy:** For Now I have only used v8 CPU for the computation of the first part. Hopefull will look at the GPU side on the second phase.

### Classical Acceleration (MTS)
* **Strategy:** For Now I have only used v8 CPU for the computation of the first part. Hopefull will look at the GPU side on the second phase.
### Hardware Targets
* **Dev Environment:** Qbraid (CPU) for logic
* **Production Environment:** v8  CPU

---

## 4. The Verification Plan

### Unit Testing Strategy
* **Framework:** havent doen any of this yet.
* **AI Hallucination Guardrails:** Manuallly checked teh code line by line.

### Core Correctness Checks
* **Check 1 (Symmetry):** [Describe a specific physics check]
    * *Example:* "LABS sequence $S$ and its negation $-S$ must have identical energies. We will assert `energy(S) == energy(-S)`."
* **Check 2 (Ground Truth):**
    "For $N=7$, the known optimal energy is 3.0. Thats what I used for check.

---

## 5. Execution Strategy & Success Metrics

### Agentic Workflow
* **Plan:** [How will you orchestrate your tools?]
    * *Example:* "We are using Cursor as the IDE. We have created a `skills.md` file containing the CUDA-Q documentation so the agent doesn't hallucinate API calls. The QA Lead runs the tests, and if they fail, pastes the error log back into the Agent to refactor."

### Success Metrics
* **Metric 1 (Approximation):** [e.g., Target Ratio > 0.9 for N=30]
* **Metric 2 (Speedup):** [e.g., 10x speedup over the CPU-only Tutorial baseline]
* **Metric 3 (Scale):** [e.g., Successfully run a simulation for N=40]

### Visualization Plan
* **Plot 1:** [e.g., "Time-to-Solution vs. Problem Size (N)" comparing CPU vs. GPU]
* **Plot 2:** [e.g., "Convergence Rate" (Energy vs. Iteration count) for the Quantum Seed vs. Random Seed]

---

## 6. Resource Management Plan

* **Plan:** [How will you avoid burning all your credits?]
    * *Example:* "We will develop entirely on Qbraid (CPU) until the unit tests pass. We will then spin up a cheap L4 instance on Brev for porting. We will only spin up the expensive A100 instance for the final 2 hours of benchmarking."
    * *Example:* "The GPU Acceleration PIC is responsible for manually shutting down the Brev instance whenever the team takes a meal break."
