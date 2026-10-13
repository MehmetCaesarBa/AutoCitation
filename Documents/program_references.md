# References — program-guided verification (`Backend/Pipeline/program.py`)

Sources behind the comparison-frame design in `program.py`. For each one: what it
contributes, and where that idea appears in the code.

---

## 1. Fact-checking with reasoning programs and decomposition

**[1] ProgramFC.** Liangming Pan, Xiaobao Wu, Xinyuan Lu, Anh Tuan Luu, William Yang
Wang, Min-Yen Kan, Preslav Nakov. *Fact-Checking Complex Claims with Program-Guided
Reasoning.* ACL 2023. arXiv:2305.12744.
- An LLM writes a reasoning program (`Question`, `Verify`, `Predict`) for each claim,
  and the program is run step by step.
- In code: the `Step`/`Program` representation, the `Question` handler and the trace
  attached to every rationale. `program.py` departs from the paper on purpose:
  programs are built from a typed frame instead of being written freely by a model
  (see the module docstring).

**[2] QACheck.** Liangming Pan, Xinyuan Lu, Min-Yen Kan, Preslav Nakov. *QACheck: A
Demonstration System for Question-Guided Multi-Hop Fact-Checking.* EMNLP 2023 System
Demonstrations. arXiv:2310.07609.
- Verifies a claim through a series of generated questions. A *QA validator* checks
  each answer before the reasoner uses it.
- In code: the Question → Match flow for ranking claims. The validator idea matches
  `_answer_is_grounded` and the NLI check in `_answer_is_right_kind`.

**[3] FOLK.** Haoran Wang, Kai Shu. *Explainable Claim Verification via
Knowledge-Grounded Reasoning with Large Language Models.* Findings of EMNLP 2023.
arXiv:2310.05253.
- Translates a claim into first-order-logic predicates, one per sub-claim, and
  answers each with a grounded question.
- In code: the `ComparisonFrame`. FOLK uses free-form predicates; the frame uses a
  fixed set of typed slots that the model fills (`FRAME_PARSER = "llm"`), and each
  slot is checked against the claim.

**[4] QuanTemp.** V. Venktesh, Abhijit Anand, Avishek Anand, Vinay Setty. *QuanTemp:
A Real-World Open-Domain Benchmark for Fact-Checking Numerical Claims.* SIGIR 2024,
pp. 650–660. arXiv:2403.17169.
- 15,514 real numerical claims from 45 fact-checking organisations. It reports that
  claim decomposition and NLI-pretrained models help on numerical claims, and has
  comparison claims as their own category.
- In code: supports both doing the arithmetic in Python (`_handle_compare`) and
  using an MNLI/FEVER-trained cross-encoder for the kind check (`NLI_MODEL`). A
  candidate benchmark for the numeric route.

**[5] QuanTemp++ (follow-up).** *A Benchmark for Open-Domain Numerical Fact-Checking
Enhanced by Claim Decomposition.* arXiv:2510.22055.
- Builds numerical claims by decomposing them the way human fact-checkers do.
- In code: background for the decomposition route. Not used directly.

## 2. One operator set for comparisons and superlatives

**[6] Lambda DCS.** Percy Liang. *Lambda Dependency-Based Compositional Semantics.*
Technical report, 2013. arXiv:1309.4408.
- Writes superlatives as `argmax(rank, count, set, relation)`. Rank is a parameter,
  so "the second-highest" is the same operation as "the highest" with rank 2.
- In code: `ComparisonFrame.rank` and `ner.superlative_rank`. This is the main
  reason ranking and pairwise claims share one frame instead of being two claim types.

**[7] TabFact / Latent Program Algorithm.** Wenhu Chen, Hongmin Wang, Jianshu Chen,
Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, William Yang Wang. *TabFact: A
Large-scale Dataset for Table-based Fact Verification.* ICLR 2020. arXiv:1909.02164.
- Treats fact verification as program synthesis over about 50 composable functions
  (`argmax`, `argmin`, `max`, `min`, `count`, `filter`, …) run by an interpreter.
- In code: a small fixed operator set (`Question`, `Match`, `Quantity`, `Compare`)
  combined from a frame, with a deterministic executor. Also the `TableRank` lookup
  (`program._ranked_rows`): Wikipedia tables are read as rows
  (`retriever.table_sentences`), and a row is selected by the value in its rank
  cell, not by similarity. Measured: similarity ranking chose the rows ranked 11,
  12, 57 and 60 for "second highest peak Earth".

**[8] KQA Pro / KoPL.** Shulin Cao, Jiaxin Shi, Liangming Pan, Lunyiu Nie, Yutong
Xiang, Lei Hou, Juanzi Li, Bin He, Hanwang Zhang. *KQA Pro: A Dataset with Explicit
Compositional Programs for Complex Question Answering over Knowledge Base.*
ACL 2022. arXiv:2007.03875.
- KoPL writes complex questions as multi-step programs, with separate operators for
  comparing attributes between two entities and among a set.
- In code: the split between a pairwise frame (`reference` set) and a ranking frame
  (`rank` set) over the same `Quantity` lookup. *Check the exact KoPL operator names
  in the paper before quoting them in the thesis.*

## 3. Structured and unstructured evidence

**[9] FEVEROUS.** Rami Aly, Zhijiang Guo, Michael Schlichtkrull, James Thorne, Andreas
Vlachos, Christos Christodoulopoulos, Oana Cocarascu, Arpit Mittal. *FEVEROUS: Fact
Extraction and VERification Over Unstructured and Structured information.* NeurIPS
2021 Datasets and Benchmarks. arXiv:2106.05707.
- Wikipedia fact-checking where the evidence can be sentences, tables or both.
- In code: background for the **structured route** (`Pipeline/wikidata.py`). When
  a frame's category and dimension map to a Wikidata class and property (mountain
  elevation P2044, river length P2043, area P2046, population P1082, building
  height P2048), the program reads both entities' unit-normalised values and
  compares them. It does not read a sorted ranking: measured, ranking all mountains
  took 19.8 s and timed out once a second class was added, while a lookup by
  English Wikipedia sitelink takes 0.4–2 s. Matching on sitelinks also removes
  label ambiguity ("K2" matched two mountains by label).

**[15] Wikidata.** Denny Vrandečić, Markus Krötzsch. *Wikidata: a free collaborative
knowledgebase.* Communications of the ACM 57(10), 2014. doi:10.1145/2629489. Query
service: https://query.wikidata.org/ (SPARQL).
- The values compared by `wikidata.value`, read through `psn:` (normalised to SI
  units) for quantities with units, and `wdt:` for unitless counts (population).

## 4. Tools

**[10] Penn Treebank tag set.** Mitchell P. Marcus, Beatrice Santorini, Mary Ann
Marcinkiewicz. *Building a Large Annotated Corpus of English: The Penn Treebank.*
Computational Linguistics 19(2), 1993. Tagging guidelines: Beatrice Santorini, *Part-
of-Speech Tagging Guidelines for the Penn Treebank Project*, 1990.
- The fine-grained tags spaCy's `token.tag_` uses: `JJR`/`JJS` (comparative and
  superlative adjectives), `RBR`/`RBS` (comparative and superlative adverbs), `RB`,
  `CD`, `NNP`, etc. `_COMPARATIVE_TAGS` and `ner._SUPERLATIVE_TAGS` are defined in
  these tags.

**[11] spaCy.** Matthew Honnibal, Ines Montani, Sofie Van Landeghem, Adriane Boyd.
*spaCy: Industrial-strength Natural Language Processing in Python.* 2020.
doi:10.5281/zenodo.1212303. Model: `en_core_web_sm`.
- Dependency labels (`amod`, `acomp`, `advmod`, `attr`, `pobj`) in `_dimension_of`,
  `_find_comparative`, `_is_universal`, `_scope_of`.

**[12] Pint.** Hernan E. Grecco et al. *Pint: a Python package to define, operate and
manipulate physical quantities.* https://github.com/hgrecco/pint. Pinned to
`>=0.24,<0.25` (0.24.4 is the newest release that installs on the project's Python
3.10 venv).
- Replaced the hand-written `_UNIT_TO_BASE` table in `_to_base`. Units are checked
  for dimensionality (`[length]`, `[length]**2`, `[mass]`), so a length can never be
  compared with a mass.

**[13] DeBERTa-v3 NLI cross-encoder.** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
(Hugging Face), trained on MNLI, FEVER and ANLI. Base model: Pengcheng He, Jianfeng
Gao, Weizhu Chen. *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with
Gradient-Disentangled Embedding Sharing.* ICLR 2023. arXiv:2111.09543.
- The kind check in `_answer_is_right_kind`.

**[14] Ollama structured outputs.** The `format` field of `/api/generate` takes a JSON
schema and constrains decoding to it. https://github.com/ollama/ollama/blob/main/docs/api.md
- `_FRAME_SCHEMA`, sent through `llm.generate(..., format=...)` when
  `FRAME_PARSER = "llm"`.
