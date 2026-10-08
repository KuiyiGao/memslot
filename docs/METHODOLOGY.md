# Pipeline and evaluation

## Data flow

MemSlot attention, learned over frozen RoBERTa features of the contract text, decides which words are selected and how they are rendered. DeepSeek-OCR decodes the rendered pages into text, and that text goes to a separate text QA reader.

`MemSlotSaliency.train_on_contracts` uses reconstruction and slot-diversity losses with no question or answer labels. The supervised checkpoint in `notebooks/MemSlot_Report.ipynb` starts from the unsupervised one and adds a binary cross-entropy loss against gold answer spans. The notebooks split CUAD by contract; record the exact split, data, checkpoint, and outputs for each run. The earlier `run.py` uses question-conditioned saliency, which is a different setting.

The text branches prune words or request a summary before the same QA-reader stage. Word budgets, tokenizer counts, rendered-page counts, OCR visual-token estimates, and decoded-text token counts are different quantities; record each one separately. The shared tokenizer is used only for counting.

## Local span-QA diagnostics

`deepseek_pipeline/cuad_metrics.py` provides `cuad_evaluate(predictions, golds, confidences=None, jaccard_threshold=0.5)` for notebook callers. Its metric definitions are local; use the official CUAD evaluation code for numbers comparable to published results.

- Each item contains a prediction and a list of acceptable nonempty gold spans. An empty gold list means unanswerable; a blank prediction is an abstention.
- EM and token F1 are averaged over all items. An abstention on an unanswerable item receives one for each.
- A nonblank prediction is a true positive when a gold span exists and normalized word-set Jaccard reaches the threshold. Other attempted answers, including answers to unanswerable items, remain in the precision denominator.
- Recall divides by all answerable items, including incorrect answers and abstentions. Each complete confidence tie group defines one threshold; missing confidence gives one tied operating point.
- Area uses trapezoidal integration from `(recall=0, precision=1)` to the observed operating points, with no extension beyond achieved recall. `precision_at_80_recall` is the maximum observed precision at recall ≥ 0.8, or zero if unreachable.
- With no answerable items, precision-recall statistics are undefined (`NaN`). An empty batch also has undefined EM/F1.

Reader confidence is the reader's self-reported score from 1 to 5, divided by 5, or 0.5 when the reply cannot be parsed as JSON.

## Implementation notes

The offline tests check metric edge cases, deterministic word pruning, and the OCR wrapper's contract using fixtures. Gaussian smoothing takes a centered slice of the full convolution, so positions and output length stay correct when the word sequence is shorter than the kernel. The OCR wrapper loads the model files as published, without patching them or installing placeholder Transformers classes, and raises an error if the model returns no decoded text.

Attention rejects sequences with no valid tokens. Its padding sentinel follows the input dtype, and half-precision saliency is normalized in float32 so constant scores stay finite. Empty training input raises an error. Word scores use character offsets and take the maximum across overlapping windows.

Rendering measures the full wrapped layout before allocating the image. A word wider than the line is drawn at a smaller font size so it stays whole; a header wider than the line raises an error. The fixed-page renderer `render_tsvr_page` warns when the budgeted words do not all fit on the page.

`n_vision_tokens` comes from the mode table in `metrics.py`. For the cropping `gundam` mode the wrapper counts one crop; pass the real crop count to `vision_token_count_for_mode` when you need an exact figure.
