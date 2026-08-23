# Literature Review: Linear-Time Long-Context Retrieval with Intra-Document Cross-Page Consistency Verification

**Prepared for:** MS AI Research Project  
**Target Venue:** EMNLP Findings / NAACL / AAAI Applied Track  
**Date:** June 2026

---

## ⚠️ NOVELTY FLAG — READ FIRST

After a thorough literature scan across SSM-based retrieval, long-document retrieval, cross-document verification, and RAG hallucination mitigation, **I found no prior work that combines (a) a linear-time selective state-space model encoder for long-document retrieval with (b) an intra-document cross-page consistency verification/reranking step applied at retrieval time.**

The two closest papers are:

- **Zhang et al., "Mamba Retriever" (2408.08066, CIKM 2024):** Uses a Mamba bi-encoder for dense retrieval. Validated on MS MARCO, BEIR, and LoCoV0. Does **not** include any cross-page consistency module. Our work explicitly builds on top of this.

- **Liu et al., "Fast and Faithful" (2603.23508, March 2026):** Adds a verification component to a long-document RAG pipeline, but it operates **post-generation** (checking whether the generated answer faithfully reflects retrieved passages) and does **not** use an SSM encoder. Our work operates **pre-generation**, at the retrieval stage, checking whether the retrieved evidence is internally consistent across pages before anything reaches the generator.

**Conclusion: The combination and specific setting are novel. Proceed to Part 2.**

---

## 1. Narrative Synthesis

### 1.1 SSM Architecture Foundations

The Mamba model (Gu & Dao, 2023; arXiv:2312.00752) introduced selective state-space models (selective SSMs or S6), achieving linear-time sequence processing by making the SSM parameters input-dependent. Unlike fixed-recurrence SSMs, the selection mechanism lets the model choose which information to propagate, giving it expressiveness comparable to attention while scaling as O(T) rather than O(T²). This was a meaningful practical advance for long sequences: a 2,048-token document costs roughly 4× less to process than under full attention, and at 8,192 tokens the gap is ~16×.

Mamba-2 (Dao & Gu, 2024; "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality") formalised a duality between SSMs and linear attention via structured matrices, simplifying the recurrence to a scalar-times-identity update (SSD framework). This enabled tensor- and sequence-parallelism tooling originally developed for Transformers to port directly to SSMs, and Mamba-2 outperformed Mamba-1 in both throughput and downstream quality. The OpenReview forum (ICLR 2025) lists the accepted paper.

Mamba-3 (arXiv:2603.15569, ICLR 2026) is the current state of the art. It introduces exponential-trapezoidal discretization, complex-valued state updates (making state tracking more expressive), and a MIMO (multi-input multi-output) formulation. At 1.5B parameters, Mamba-3 improves average downstream accuracy by 1.8 points over Mamba-2 while using half its predecessor's state size.

**Implication for our work:** Our retriever fine-tunes from a Mamba-2 checkpoint (available on HuggingFace as `state-spaces/mamba2`). Mamba-3's released weights could serve as a future upgrade and stronger baseline in the Discussion section.

### 1.2 Mamba Applied to Retrieval

Zhang et al.'s "Mamba Retriever" (arXiv:2408.08066, CIKM 2024) is the paper our work directly builds on and must be clearly cited as such. They trained a Mamba bi-encoder retriever via in-batch negatives on MS MARCO, achieving competitive MRR@10 against BERT-based dense retrievers while showing markedly lower latency on long-text passages (their LoCoV0 experiments). They evaluated on the standard BEIR zero-shot benchmark suite and on LoCoV0 for long-context settings. They demonstrated that the linear-time recurrence transfers well to the retrieval setting.

**Gap:** Zhang et al. treat long-document retrieval as a pure encoding problem. They do not address what happens when retrieved passages from different pages of the same document contradict or partially overlap — that is, the consistency of the retrieved evidence set is entirely ignored. This is our entry point.

Xu et al., "State Space Models are Strong Text Rerankers" (arXiv:2412.14354, RepL4NLP 2025) evaluated Mamba-1 and Mamba-2 as pointwise text rerankers (replacing BERT in a standard cross-encoder reranking setup). Mamba-2 matched transformer rerankers of similar size while being faster at inference. However, their reranker scores individual query-document pairs independently for relevance — it does not model pairwise interactions between candidates from different pages of the same document.

**Gap:** No cross-page awareness; the reranker treats each (query, passage) pair atomically.

### 1.3 Foundational Retrieval and RAG Papers

Karpukhin et al.'s DPR (arXiv:2004.04906, EMNLP 2020) established the modern bi-encoder dense retrieval paradigm: a dual-encoder trained with in-batch negatives, indexed with FAISS. Our Mamba-2 bi-encoder follows exactly this architecture with the Transformer replaced by an SSM.

Lewis et al.'s RAG (arXiv:2005.11401, NeurIPS 2020) established the retrieve-then-generate pipeline we work within. They observed that retrieval quality directly bounds generation faithfulness — a result that motivates our cross-page consistency module: garbage-in (contradictory retrieved passages) → garbage-out (hallucinated answers).

### 1.4 Long-Document Retrieval Benchmarks

**LoCoV0 / LoCo** (Chang et al., arXiv:2402.07440, 2024) is a 12-task benchmark for long-context retrieval, constructed to measure settings where chunking is not possible or loses critical information. Tasks include CourtListener legal filings, scientific abstracts, and StackOverflow posts. LoCoV0 is the open preview version used in Zhang et al.'s evaluation; LoCoV1 is the current stable release. We adopt LoCoV1 as our efficiency benchmark.

**LegalBench-RAG** (Pipitone et al., arXiv:2408.10343, 2024) is a retrieval benchmark for the legal domain, constructed from CUAD (general contracts), MAUD (merger agreements), ContractNLI (NDAs), and PrivacyQA (privacy policies). It uses character-level Precision@k and Recall@k rather than document-level nDCG, reflecting the need for precise span extraction. Legal documents are ideal for our cross-page consistency task: a contract signed on page 3 may be amended on page 17, and a naive chunked retriever that fetches page 3 without page 17 will return stale, misleading evidence.

**TREC DL (2019–2023)** uses the MS MARCO document collection and provides standard NDCG@10 / MRR comparisons that reviewers expect for any dense retrieval paper. We use it to establish credibility against the BM25 and Transformer baselines.

### 1.5 Cross-Document Verification and Graph-Based Reranking

**GraphRAG** (Edge et al., Microsoft, 2024) builds a knowledge graph from a document corpus via LLM entity extraction, then generates hierarchical community summaries for global query answering. It targets queries that span many documents (sensemaking over a whole corpus). GraphRAG is not designed for — and does not address — the case where a single multi-page document contains internally contradictory passages that a naive retriever might simultaneously surface.

**HippoRAG / HippoRAG 2** (2024/2025; arXiv:2502.14802) uses Personalized PageRank over a dual-node (passage + phrase) knowledge graph to answer multi-hop questions that require chaining evidence *across separate documents* in a corpus. The graph is built over the full corpus offline. Again, the problem setting is cross-document multi-hop, not intra-document cross-page.

**IndexRAG** (Bao & Shi, arXiv:2603.16415, March 2026) shifts cross-document reasoning to index time by identifying "bridge entities" shared across documents and pre-generating bridging facts as standalone retrievable units. Improves multi-hop QA by 4.6 F1 points over naive RAG. Does not use an SSM encoder; the multi-hop setting is across separate documents, not within one document.

**Cross-Granularity Hypergraph RAG** (arXiv:2508.11247, 2025) extends RAG with a hypergraph over chunks to capture multi-hop evidence across documents for QA. Same cross-document scope; no intra-document consistency modelling.

**What all of these share:** They model relationships *between documents* (or between chunks from different documents). None of them model whether two passages retrieved from *page 3* and *page 17 of the same contract* are mutually consistent or contradictory.

### 1.6 RAG Faithfulness and Hallucination Mitigation

**Fast and Faithful** (Liu et al., arXiv:2603.23508, March 2026) is the most recent and most relevant hallucination-mitigation paper. It integrates a real-time verifier into a production RAG pipeline to check that the *generated answer* faithfully reflects the *retrieved passages*, processing documents up to 32K tokens with adaptive inference. The verification is post-generation (output grounding check). Our work is pre-generation: we verify that the *retrieved evidence itself* is internally consistent before the generator sees it. These are complementary, not competing.

**ReDeEP** (OpenReview, 2024) detects hallucination via mechanistic interpretability of the generator, also post-generation.

**RAC** (arXiv:2410.15667, 2024) corrects factuality post-generation via retrieval augmentation of the output. Post-generation.

**Rowen** (SIGIR-AP 2025) adaptively retrieves additional context when a low-confidence generation is detected. Iterative retrieval, not consistency-based reranking.

**Pattern:** The hallucination mitigation literature is almost entirely post-generation. Pre-generation, retrieval-side consistency checking — the angle we take — is an underexplored intervention point.

---

## 2. Related-Work Table

Use this table directly in the paper's Related Work section (as a supporting `\table{}`).

| Paper | Year | Method | Dataset(s) | Gap vs. Our Work |
|---|---|---|---|---|
| Gu & Dao — Mamba | 2023 | Selective SSM, linear-time seq modeling via input-dependent parameters | Language modeling (Pile, Wikitext) | Architecture backbone only; no retrieval application |
| Dao & Gu — Mamba-2 (SSD) | 2024 | State space duality; scalar-times-identity recurrence; tensor parallelism | Language modeling | Architecture improvement; enables our encoder; no retrieval |
| Gu et al. — Mamba-3 | 2026 | Complex-valued states, exponential-trapezoidal discretization, MIMO | Language modeling (ICLR 2026) | Latest SSM; potential future encoder upgrade |
| Karpukhin et al. — DPR | 2020 | Bi-encoder dense retrieval; in-batch negatives; FAISS index | NQ, TriviaQA, WebQ, TREC | Foundational; Transformer-based; fixed-length chunks; no long-doc |
| Lewis et al. — RAG | 2020 | Retriever + seq2seq generator; non-parametric Wikipedia index | NQ, TriviaQA, WebQ | Foundational pipeline; short passages; no cross-page |
| Zhang et al. — Mamba Retriever | 2024 | Mamba bi-encoder; contrastive fine-tuning; FAISS | MS MARCO, BEIR, LoCoV0 | Direct predecessor; no cross-page consistency module |
| Xu et al. — SSM Rerankers | 2024 | Mamba-1/2 as pointwise text reranker | TREC DL, standard BEIR | Reranks by query-doc relevance only; no pairwise cross-page reasoning |
| Edge et al. — GraphRAG | 2024 | LLM-extracted KG + community summaries; global query answering | MS internal corpus | Cross-document/corpus; no intra-document consistency |
| Gutierrez et al. — HippoRAG 2 | 2025 | Dual-node KG + Personalized PageRank | BEIR, MuSiQue, HotpotQA | Cross-document multi-hop; not intra-document; no SSM |
| Bao & Shi — IndexRAG | 2026 | Bridge entities at index time; flat retrieval + single LLM call | HotpotQA, 2WikiMHQA, MuSiQue | Cross-document; index-time; not SSM; not intra-doc |
| Pipitone et al. — LegalBench-RAG | 2024 | Benchmark: CUAD, MAUD, ContractNLI, PrivacyQA; char-level P/R@k | Legal documents | Evaluation benchmark only; no retrieval model proposed |
| Chang et al. — LoCo / LoCoV0 | 2024 | Long-context retrieval benchmark; M2-BERT encoder | 12 tasks (legal, sci, forum) | Benchmark; no cross-page consistency modelling |
| Liu et al. — Fast and Faithful | 2026 | Post-generation faithfulness verifier; 32K tokens; adaptive inference | Enterprise RAG (production) | **Post-generation** check; no SSM encoder; not retrieval-side |
| Xu et al. — Hypergraph RAG | 2025 | Hypergraph over chunks; multi-hop QA | HotpotQA, 2WikiMHQA | Cross-document; not intra-document; no SSM |

---

## 3. Confirmed Novelty Statement

The following combination and setting has **not** been addressed by any paper found in this review:

> A **Mamba-2 (or Mamba-3) bi-encoder** used for long-document retrieval **without chunking**, paired with a **cross-page consistency verification module** that inspects pairwise agreement/contradiction among retrieved passages from *different pages of the same document* and reranks/filters them before passing the evidence set to the generator — evaluated specifically on **long single-document multi-page grounding** tasks (legal contracts, TREC DL long documents).

This is distinct from:
- "First to use Mamba for retrieval" → Zhang et al. (2024) already did that ✗
- "First to do graph reranking" → GraphRAG, HippoRAG, IndexRAG family already did that ✗
- "Post-generation faithfulness verification" → Fast and Faithful (2026) already did that ✗

The contribution is the **combination + intra-document setting**. Frame it precisely.

---

## 4. BibTeX Entries (for use in paper)

```bibtex
@article{gu2023mamba,
  title   = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author  = {Gu, Albert and Dao, Tri},
  journal = {arXiv preprint arXiv:2312.00752},
  year    = {2023}
}

@inproceedings{dao2024mamba2,
  title     = {Transformers are {SSM}s: Generalized Models and Efficient Algorithms
               Through Structured State Space Duality},
  author    = {Dao, Tri and Gu, Albert},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning
               (ICML)},
  year      = {2024}
}

@inproceedings{gu2026mamba3,
  title     = {Mamba-3: Improved Sequence Modeling using State Space Principles},
  author    = {Gu, Albert and others},
  booktitle = {The Fourteenth International Conference on Learning Representations
               (ICLR)},
  year      = {2026},
  note      = {arXiv:2603.15569}
}

@inproceedings{zhang2024mambaretriever,
  title     = {Mamba Retriever: Utilizing Mamba for Effective and Efficient Dense
               Retrieval},
  author    = {Zhang, Hanqi and others},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information
               and Knowledge Management (CIKM)},
  year      = {2024},
  note      = {arXiv:2408.08066}
}

@inproceedings{xu2024ssmrerankers,
  title     = {State Space Models are Strong Text Rerankers},
  author    = {Xu, Zhichao and Yan, Jinghua and Gupta, Ashim and Srikumar, Vivek},
  booktitle = {Proceedings of the 5th Workshop on Representation Learning for NLP
               (RepL4NLP 2025)},
  year      = {2025},
  note      = {arXiv:2412.14354}
}

@inproceedings{karpukhin2020dpr,
  title     = {Dense Passage Retrieval for Open-Domain Question Answering},
  author    = {Karpukhin, Vladimir and others},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural
               Language Processing (EMNLP)},
  year      = {2020},
  note      = {arXiv:2004.04906}
}

@inproceedings{lewis2020rag,
  title     = {Retrieval-Augmented Generation for Knowledge-Intensive {NLP} Tasks},
  author    = {Lewis, Patrick and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020},
  note      = {arXiv:2005.11401}
}

@inproceedings{pipitone2024legalbenchrag,
  title     = {{LegalBench-RAG}: A Benchmark for Retrieval-Augmented Generation in
               the Legal Domain},
  author    = {Pipitone, Domenic and others},
  journal   = {arXiv preprint arXiv:2408.10343},
  year      = {2024}
}

@inproceedings{chang2024loco,
  title     = {Benchmarking and Building Long-Context Retrieval Models with {LoCo}
               and {M2-BERT}},
  author    = {Chang, Jon Saad-Falcon and others},
  journal   = {arXiv preprint arXiv:2402.07440},
  year      = {2024}
}

@article{edge2024graphrag,
  title   = {From Local to Global: A Graph {RAG} Approach to Query-Focused
             Summarization},
  author  = {Edge, Darren and others},
  journal = {arXiv preprint arXiv:2404.16130},
  year    = {2024}
}

@article{gutierrez2025hipporag2,
  title   = {From {RAG} to Memory: Non-Parametric Continual Learning for Large
             Language Models},
  author  = {Gutierrez, Bernal Jimenez and others},
  journal = {arXiv preprint arXiv:2502.14802},
  year    = {2025}
}

@article{bao2026indexrag,
  title   = {{IndexRAG}: Bridging Facts for Cross-Document Reasoning at Index Time},
  author  = {Bao, Zhenghua and Shi, Yi},
  journal = {arXiv preprint arXiv:2603.16415},
  year    = {2026}
}

@article{liu2026fastandfaithful,
  title   = {Fast and Faithful: Real-Time Verification for Long-Document
             Retrieval-Augmented Generation Systems},
  author  = {Liu, Xunzhuo and He, Bowei and Liu, Xue and Zhang, Haichen and
             Chen, Huamin},
  journal = {arXiv preprint arXiv:2603.23508},
  year    = {2026}
}
```

---

## 5. Gaps Confirmed / Things to Watch

- **"State Space Models are Strong Text Rerankers" (2412.14354)** should be cited in Related Work under reranking. Their finding that Mamba-2 matches Transformer rerankers actually strengthens our case: it shows SSMs are viable for ranking tasks, and we extend that to consistency-based reranking.
- **LoCo/LoCoV0:** The original Mamba Retriever paper uses LoCoV0. We should use LoCoV1 (the current stable release) and note the version difference.
- **LegalBench-RAG character-level metric:** Reviewers familiar with this benchmark will expect character-level precision/recall. Implement it properly (see Section 2.1 of the codebox spec).
- **"Fast and Faithful" (2603.23508):** Cite it positively in Related Work — "concurrent work addresses post-generation faithfulness; our work is complementary, addressing retrieval-side consistency before the generator is invoked."
- **MEGA-RAG:** Not found as a distinct published paper with this exact title. Likely a term used loosely in blog posts. Do not cite it. Use GraphRAG + HippoRAG + IndexRAG as the relevant graph-RAG family instead.
- **MIRAGE:** The found "MiRAGE" (arXiv:2601.15487) is a multimodal QA dataset generation framework, not a retrieval verification method. Not relevant to our work; skip it.
