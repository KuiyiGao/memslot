# Multipage OCR experiment plan

This is the planned comparison behind `notebooks/DeepSeek_OCR_CUAD_Multipage_Colab.ipynb`. Where the notebook's descriptions differ from the [pipeline notes](docs/METHODOLOGY.md), follow the notes.

## Question and branches

Does saliency-guided text selection and rendering preserve useful contract information after OCR, at a constrained text budget?

The planned branches include full text, saliency pruning, API summarization, lead/random selection, and OCR of selected or summarized text. A randomly initialized MemSlot is the control for slot training. Its frozen RoBERTa features still carry structure, so it differs from uniform random selection.

Each OCR branch renders pages, decodes them with DeepSeek-OCR, concatenates the decoded text, and passes that text to a separate QA reader. The OCR model's internal visual tokens stay inside the OCR model.

## Controls

- Split at the contract level and keep document IDs, sampled question IDs, and random seeds. Train saliency only on training-contract text.
- Use the same QA model, version, prompt, and response handling for every branch. Keep questions out of compression and saliency selection.
- Report the actual decoded-text length under one counting tokenizer, along with word budgets, page counts, and estimated OCR visual tokens. Report achieved ratios, not target ratios.
- Separate the effects of word selection, font emphasis, page boundaries, and OCR errors, since different rendered inputs differ in more than the OCR settings.
- Record failed calls separately from valid empty answers. Keep attempt counts, outputs, and costs.
- Record exact dependencies, model revisions, checkpoint identities, and data hashes. Use explicit seeds, because Python's built-in string hash changes between processes.

## Evaluation

Scoring will use the local EM/F1 and precision-recall diagnostics described in the [pipeline notes](docs/METHODOLOGY.md). The comparison will report full output records, achieved lengths, and uncertainty at the contract level on a held-out split. Before a paid run, check model compatibility, provider context limits, prices, and quotas.
