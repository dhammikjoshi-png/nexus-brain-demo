# NEXUS Brain demo

A tiny character-level Transformer from the NEXUS research project, running entirely in your browser (GitHub Pages).
Nothing is sent to a server: the weights load from this site and every token is computed on your device.

This is a **research model demo, not the NEXUS coding agent**. The models have about 31,000 parameters
(2 layers, 32 dimensions) and will not write real programs. They show the behaviour measured in
Experiments 1 to 4 (identifier copying and generalisation). Both models are single runs (seed 1) from Experiment 4.

## Files

- `index.html`, `nexus-model.js`: the page and a JavaScript port of the NumPy forward pass.
- `models/code_model.json`, `models/copy_model.json`: the exported weights (float32, base64 in JSON).

The JavaScript port was checked against the original NumPy model: maximum logit difference about 2e-5
and identical greedy output on 15 test prompts. The tests and export tool live with the (private) research code.
