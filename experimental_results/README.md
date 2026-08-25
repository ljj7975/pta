# Experimental Results

This directory contains the consolidated experimental findings from the patch-level fusion investigation.

## Documents

### [PTA_Limitations_and_Patch_Signal_Analysis.md](./PTA_Limitations_and_Patch_Signal_Analysis.md)

A self-contained report covering:

1. **Background** — How PTA works, image-level vs. patch-level prototypes, write-time vs. inference-time behavior
2. **PTA's Prototype Drift Problem** — Why image-level prototypes fail when CLIP is confidently wrong
3. **Patch-Level Prototype Investigation** — Why patch fusion doesn't fix PTA (bank quality, separability, structural asymmetry)
4. **Patch Agreement as Quality Signal** — The validated meta-signal and why it doesn't convert to accuracy wins
5. **Fusion Mechanisms** — How different fusion mechanisms perform and why always-on fusion collapses
6. **Summary and Supplementary Data**

### [METHODOLOGY.md](./METHODOLOGY.md)

Detailed description of how each finding was derived, including:
- Exact script used (under `scripts/` folder)
- Input data and write-time settings
- Methodology and computation details
- Output file containing the numbers

---

## Key Takeaways

1. **PTA has a structural limitation**: when CLIP and the image-level prototype disagree, the prototype is wrong 75.3% of the time — but no confidence-based filter can catch this.

2. **Patch-level evidence provides a genuinely independent signal**: whether CLIP's CLS-level guess and the patch-level vote agree is a strong indicator of prediction correctness (30+ pp purity gap).

3. **This signal does not convert to accuracy wins**: across 30+ experiments, patch content fused into predictions never reliably beats plain PTA.

4. **The fundamental issue is structural**: patch banks are built from CLIP's own guesses (no ground truth), class clusters overlap at the patch level (negative separability margins everywhere), and the signal quality is fixed by the underlying CLIP features.
