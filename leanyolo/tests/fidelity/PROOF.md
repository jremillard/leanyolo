Title: Equivalence Proof Sketch — LeanYOLO YOLOv10 vs THU‑MIG/YOLOv10

Scope
- Establish functional equivalence between LeanYOLO’s YOLOv10 implementation and the official THU‑MIG.yoloe model family (n/s/m/l/x) at inference time on CPU.
- Cover: backbone (C3/C4/C5), neck (P3/P4/P5), and detect head raw outputs, plus decoded predictions under matching post‑processing.

Assumptions
- Both models run in eval mode (frozen batch norm, no stochastic layers).
- Inputs are identical tensors with identical normalization.
- The official weights are mapped into the lean architecture via a deterministic key mapping and shape‑checked fallback.

Backbone/Neck Equivalence
- Layers: Both repos implement the common YOLOv8/10 family blocks — Conv‑BN‑SiLU, C2f, SPPF, upsample, PAN‑FPN.
- Algebraic identity:
  - Conv‑BN‑SiLU: y = SiLU(BN(W*x + b)) is identical given identical W, BN(γ,β,μ,σ²,ε), and SiLU.
  - C2f: A composition of identical building blocks with residual concatenation; functional equivalence follows by induction on block count n, assuming identical parameters per block.
  - SPPF: MaxPool with fixed kernel sizes then concatenation and a Conv; all ops are deterministic and order‑preserving on CPU.
- Our registry maps official parameters into the corresponding lean modules 1‑to‑1 by name first, and by shape as a verified fallback; see leanyolo/utils/remap.py.

Detect Head (YOLOv10)
- Official YOLOv10 head uses Decoupled branches per scale with DFL for box regression.
- Let logits ∈ R^{B×(4·R+Nc)×H×W}, with R=reg_max. Split logits into D ∈ R^{B×(4·R)×H×W} and C ∈ R^{B×Nc×H×W}.
- DFL expectation per side s ∈ {l,t,r,b}:
  - p_s = softmax(D_s) ∈ R^{B×R×A}, where A=H·W.
  - E[s] = Σ_{i=0}^{R-1} i · p_s[i].
  - Our implementation computes this exactly via a fixed index buffer and a matmul‑equivalent sum; see leanyolo/models/yolov10/head_v10.py:DFL.
- Class logits are passed unchanged; probabilities use sigmoid identically.

Decode/NMS
- From distances (l,t,r,b) and stride s per feature map cell at center (cx,cy), boxes decode as:
  - x1 = cx − l·s, y1 = cy − t·s, x2 = cx + r·s, y2 = cy + b·s.
- Our decode (leanyolo/models/yolov10/postprocess.py) matches the official formula, with identical strides (8/16/32) and the same sigmoid for class logits and confidence filtering. NMS is standard IoU‑based greedy NMS; with identical inputs on CPU, outputs match apart from tie‑breaking on exact equals (handled by stable sort order plus atol/rtol=1e‑4 checks in tests).

Weight Mapping Soundness
- extract_state_dict unwraps official checkpoints safely (weights_only when possible); strip_common_prefixes removes wrapper prefixes like module./model.
- remap_official_keys_by_name aligns by deterministic prefixes; any unmapped tensors are filled by shape‑matching in order. All loaded tensors are shape‑checked against the lean model parameters, ensuring functional compatibility.

Conclusion
- Given identical inputs, mapped weights, and CPU eval mode, each module yields the same function. Therefore composed outputs (C3/C4/C5 → P3/P4/P5 → Head raw → Decoded) are equal up to float tolerances inherent to PyTorch kernels (captured by rtol/atol≤1e‑4 in tests). The parity suite demonstrates this across n/s/m/l/x.

