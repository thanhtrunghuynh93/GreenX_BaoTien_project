# Style sheet — Green-X water paper

Derived from the author's accepted papers (voice) + the collected water-journal papers (venue) + anti-slop rules. Every section is drafted and reviewed against this file.

## Voice (from the author's accepted papers)

- Intro build: definition + citation-stuffed application list → task definition → enumerated deficiencies ("Firstly, … Secondly, … Thirdly, …") → "To address the above challenges, …" → "In essence, our idea is to …" → objectives.
- Sentences: 22–30-word active declaratives; ~75–80% active voice; "we/our". Passive only for setup facts ("All models are implemented in scikit-learn…").
- Sentence-initial connectives: "However," "Moreover," "More specifically," "Intuitively," "It is worth noting that," "A key finding is that".
- Paragraphs may open with **bold run-in labels** ("**Evaluation protocol.**").
- Methods: plain-English gloss → notation → intuition sentence. Requirement/task labels: (T1), (T2). Forward pointers: "Details are in Text S1."
- Results: open with a numbered RQ list; each subsection answers one ("Table 3 answers (RQ1) …"). Claims as ranges, not decimals ("2–3× larger", "roughly half"). Every claim is followed by a causal explanation ("This is because …"). Hedges: "This could be explained by …". Negative results stated flatly and reframed as supporting the thesis.
- Figures/tables: ultra-short noun-phrase captions without a terminal period ("Fig. 2: Evaluation workflow"); the only long caption allowed is the motivating figure.
- Conclusion: one paragraph restating the contribution, then "In future work, we plan to …".

## Venue (STOTEN / CEJ / ACS ES&T Water conventions)

- Four numbered sections: 1. Introduction / 2. Materials and methods / 3. Results and discussion / 4. Conclusions. Environmental implications = closing subsection 3.x, not its own section.
- Abstract: one unstructured paragraph, 300–340 words, opens with the environmental problem, carries quantified results, closes with a practice-relevant sentence. 5 Highlights bullets (≤85 chars each). Graphical abstract required at submission.
- Introduction: 5–8 paragraphs, ends with "this study aims to (1) … (2) … (3) …".
- ML methods: one prose paragraph per algorithm citing its original source; ≤6 numbered equations in main text; hyperparameters and architecture detail → Supplementary (Table S1, Text S1…). One workflow schematic figure.
- Performance: metric tables + a parity plot; the split protocol is explicitly justified.
- Limitations: informal closing paragraph inside Results and discussion ("First, … The second concern is …").

## Banned (out of character / AI slop)

- "In today's rapidly evolving landscape", "game-changer", "cutting-edge", "revolutionary", "remarkably", "strikingly", "seamless", "paradigm".
- Rhetorical questions outside the RQ list; exclamation marks; contractions; "you"; first-person singular.
- Over-precise bragging ("a 3.7% improvement") — round to ranges.
- Standalone Limitations section; fake vulnerability arcs; a closing question.
- Mixing British/American spelling — **American English throughout**.
- Any citation not verified against a real source: use `\citep{TODO-<topic>}` and log it in `main.bib` under the UNVERIFIED block. Never invent author/year/venue.

## Evidence discipline

Every number in the manuscript must trace to a CSV in `Model/paper_figures/` or `Model/<experiment>/model_scores.csv`. When drafting ahead of a finished run, insert `XX` and list the pending value in `CLAIMS.md`; no `XX` may survive to a submitted draft.
