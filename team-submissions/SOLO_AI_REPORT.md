> **Note to Students:** > The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have employed AI agents in a thoughtful way.

**Required Sections:**

1. **The Workflow:** How did you organize your AI agents? (e.g., "We used a Cursor agent for coding and a separate ChatGPT instance for documentation").

I used Gemini for coding and it gives a bit of documentation

3. **Verification Strategy:** How did you validate code created by AI?

I am traditionally coder by training. Hence I already had a view on how the code should look like. The AI had gave almost all the code as I thought, except for one (which use `class`, I thought `class` is not needed there). 

I had looked at each step, one by one and checked whether it makes sense. So steps where wrong, some steps where missing, and some like `four_qubit_rotation_block` where AI said gave first few and said so on. hahaha. 

Hence I believe I have corrected all the mistakes of Gemini.
   

3. **The "Vibe" Log:**
* *Win:* It reduced the time and found functions in cudaq that I needed (First time in cudaq). It gave classical MTS really well.
* *Learn:* AI gave a beatiful code. And since its my first time coding cudaq, it helped how to code in cudaq. Although I checked each library function it gave in the cudaq documentation to confirm thats the correct and uptodate verison.
* *Fail:* Lots of place AI had failed like i said before AI gave  first few and said like this for other gates for `four_qubit_rotation_block`. It didnt give the cyclic permutation needed for four qubit gate index. It failed initize the list  properly, though that a basic thing. 
* *Context Dump:* Share any prompts, `skills.md` files, MCP etc. that demonstrate thoughtful prompting.


