"""
Synthetic cross-page consistency benchmark generator.

Takes long documents from LegalBench-RAG (specifically CUAD contracts)
and programmatically generates query-evidence pairs where the correct
answer requires reconciling information stated on two different pages:
one page states a fact, a later page amends / contradicts / refines it.

Contradiction types injected
-----------------------------
1. NUMERIC_AMENDMENT    — a numeric value (payment, penalty, term length) is changed
2. DATE_AMENDMENT       — an effective date or deadline is changed
3. PARTY_SUBSTITUTION   — a named party is replaced in a later clause
4. OBLIGATION_ADDITION  — a new obligation is imposed that supersedes a permissive clause
5. TERMINATION_CHANGE   — a notice period or termination right is altered

Targeting 2 000+ pairs
-----------------------
Default parameters: 400 source documents × 5 contradictions_per_doc = 2 000 pairs.
The generator retries up to 3× per slot, so actual yield depends on document richness.

Quality validation
------------------
A random sample of `validation_sample` (default 100) items is written to
validation_sample.jsonl for manual review.  An automated sanity check verifies
that each injected span is exactly recoverable from the modified document.
Coverage statistics (injection type breakdown, section-gap distribution) are
written to benchmark_stats.json.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Dollar amounts and unit-bearing numbers
NUMERIC_PATTERN = re.compile(
    r"\$[\d,]+(?:\.\d{2})?"
    r"|\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*"
    r"(?:days?|months?|years?|weeks?|hours?|percent|%)",
    re.IGNORECASE,
)

# Calendar dates in common legal formats
DATE_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}",
    re.IGNORECASE,
)

# Named parties (very rough heuristic — capitalised N-grams after "between" / "and")
PARTY_PATTERN = re.compile(
    r"(?:between|and)\s+([A-Z][A-Za-z\s,\.]+(?:Inc\.|LLC|Corp\.|Ltd\.|LP|LLP)?)",
)

# Permissive phrases that can be negated
PERMISSIVE_PHRASES = [
    ("may terminate", "may not terminate"),
    ("shall have the right to assign", "shall not have the right to assign"),
    ("is permitted to sublicense", "is not permitted to sublicense"),
    ("has the option to renew", "shall have no option to renew"),
    ("may disclose", "shall not disclose"),
]

AMENDMENT_TEMPLATES_NUMERIC = [
    "Notwithstanding Section {orig_section}, the {field} shall be {new_value} "
    "from the Effective Date of this Amendment.",
    "Section {orig_section} is hereby amended and restated in its entirety. "
    "The {field} is revised to {new_value}.",
    "For the avoidance of doubt, the parties agree that {field} referenced in "
    "Section {orig_section} shall henceforth be {new_value}.",
    "This Amendment supersedes Section {orig_section}: the {field} is "
    "changed from its original value to {new_value} with immediate effect.",
]

AMENDMENT_TEMPLATES_DATE = [
    "Notwithstanding Section {orig_section}, the effective date is hereby "
    "amended to {new_value}.",
    "Section {orig_section} is amended: all references to {orig_value} are "
    "replaced with {new_value}.",
    "By mutual agreement, the deadline set out in Section {orig_section} "
    "is extended to {new_value}.",
]

AMENDMENT_TEMPLATES_PARTY = [
    "The parties hereby acknowledge that all rights and obligations of "
    "{orig_value} under Section {orig_section} are assigned to and assumed "
    "by {new_value} as of the date of this Amendment.",
    "Notwithstanding Section {orig_section}, {new_value} shall be substituted "
    "for {orig_value} throughout this Agreement.",
]

AMENDMENT_TEMPLATES_OBLIGATION = [
    "Section {orig_section} is supplemented by the following: {new_value}.",
    "In addition to Section {orig_section}, the parties further agree: {new_value}.",
    "This clause supersedes and replaces Section {orig_section}: {new_value}.",
]

# New notice periods / termination alternatives
NOTICE_PERIODS = ["30 days", "60 days", "90 days", "14 days", "45 days"]


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _split_sections(text: str) -> List[Tuple[int, str]]:
    """
    Split document text into (section_index, section_text) pairs.
    Boundaries: blank lines OR §/Article/SECTION/Exhibit/Schedule headers.
    """
    section_re = re.compile(
        r"(?:\n{2,})|(?=§\s*\d)|(?=Article\s+\d)|(?=SECTION\s+\d)"
        r"|(?=Exhibit\s+[A-Z])|(?=Schedule\s+[A-Z0-9])",
        re.IGNORECASE,
    )
    parts = section_re.split(text)
    return [(i, p.strip()) for i, p in enumerate(parts) if p.strip()]


# ---------------------------------------------------------------------------
# Injection strategies
# ---------------------------------------------------------------------------

def _inject_numeric_amendment(
    sections: List[Tuple[int, str]], rng: random.Random
) -> Optional[Dict]:
    """Find a section with a numeric value; inject a later amendment."""
    for idx, (sec_idx, sec_text) in enumerate(sections[:-2]):
        matches = NUMERIC_PATTERN.findall(sec_text)
        if not matches:
            continue
        orig_value = rng.choice(matches)
        num_match = re.search(r"[\d,]+(?:\.\d+)?", orig_value)
        if not num_match:
            continue
        orig_num = float(num_match.group().replace(",", ""))
        if orig_num == 0:
            continue
        multiplier = rng.uniform(1.1, 1.5) if rng.random() > 0.5 else rng.uniform(0.5, 0.9)
        new_num = round(orig_num * multiplier, 2)
        # Keep the unit suffix
        suffix = orig_value[num_match.end():]
        new_value = f"{new_num:,.2f}{suffix}"
        is_dollar = orig_value.startswith("$")
        if is_dollar:
            new_value = f"${new_num:,.2f}"
        field = "payment amount" if is_dollar else "term"
        insert_idx = min(idx + rng.randint(2, 4), len(sections) - 1)
        template = rng.choice(AMENDMENT_TEMPLATES_NUMERIC)
        amendment_text = template.format(
            orig_section=sec_idx + 1, field=field, new_value=new_value
        )
        return {
            "type": "NUMERIC_AMENDMENT",
            "original_section_idx": sec_idx,
            "original_value": orig_value,
            "new_value": new_value,
            "amendment_text": amendment_text,
            "insert_after_section_idx": insert_idx,
            "query": f"What is the {field} specified in this agreement?",
            "gold_answer": new_value,
            "distractor_answer": orig_value,
            "gold_span": amendment_text,
            "distractor_span": orig_value,
        }
    return None


def _inject_date_amendment(
    sections: List[Tuple[int, str]], rng: random.Random
) -> Optional[Dict]:
    """Find a section with a date; inject a later date amendment."""
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    for idx, (sec_idx, sec_text) in enumerate(sections[:-2]):
        matches = DATE_PATTERN.findall(sec_text)
        if not matches:
            continue
        orig_value = rng.choice(matches)
        # Generate a date ±1-3 years away
        year_match = re.search(r"\d{4}", orig_value)
        if not year_match:
            continue
        orig_year = int(year_match.group())
        delta = rng.randint(1, 3) * (1 if rng.random() > 0.3 else -1)
        new_year = orig_year + delta
        new_month = rng.choice(months)
        new_day = rng.randint(1, 28)
        new_value = f"{new_month} {new_day}, {new_year}"
        insert_idx = min(idx + rng.randint(2, 4), len(sections) - 1)
        template = rng.choice(AMENDMENT_TEMPLATES_DATE)
        amendment_text = template.format(
            orig_section=sec_idx + 1,
            orig_value=orig_value,
            new_value=new_value,
        )
        return {
            "type": "DATE_AMENDMENT",
            "original_section_idx": sec_idx,
            "original_value": orig_value,
            "new_value": new_value,
            "amendment_text": amendment_text,
            "insert_after_section_idx": insert_idx,
            "query": "What is the effective date specified in this agreement?",
            "gold_answer": new_value,
            "distractor_answer": orig_value,
            "gold_span": amendment_text,
            "distractor_span": orig_value,
        }
    return None


def _inject_party_substitution(
    sections: List[Tuple[int, str]], rng: random.Random
) -> Optional[Dict]:
    """Find a named party; substitute with a successor entity."""
    for idx, (sec_idx, sec_text) in enumerate(sections[:-2]):
        matches = PARTY_PATTERN.findall(sec_text)
        if not matches:
            continue
        orig_party = rng.choice(matches).strip().rstrip(",")
        if len(orig_party) < 5:
            continue
        # Create a successor name
        suffixes = ["Holdings", "Successor", "Acquisition Corp.", "Merged Entity Inc."]
        new_party = f"{orig_party.split()[0]} {rng.choice(suffixes)}"
        insert_idx = min(idx + rng.randint(2, 5), len(sections) - 1)
        template = rng.choice(AMENDMENT_TEMPLATES_PARTY)
        amendment_text = template.format(
            orig_section=sec_idx + 1,
            orig_value=orig_party,
            new_value=new_party,
        )
        return {
            "type": "PARTY_SUBSTITUTION",
            "original_section_idx": sec_idx,
            "original_value": orig_party,
            "new_value": new_party,
            "amendment_text": amendment_text,
            "insert_after_section_idx": insert_idx,
            "query": f"Which entity holds the obligations of {orig_party} under this agreement?",
            "gold_answer": new_party,
            "distractor_answer": orig_party,
            "gold_span": amendment_text,
            "distractor_span": orig_party,
        }
    return None


def _inject_obligation_addition(
    sections: List[Tuple[int, str]], rng: random.Random
) -> Optional[Dict]:
    """Find a permissive phrase; add a superseding restrictive obligation."""
    for idx, (sec_idx, sec_text) in enumerate(sections[:-2]):
        for permissive, restrictive in PERMISSIVE_PHRASES:
            if permissive.lower() in sec_text.lower():
                new_obligation = (
                    f"Notwithstanding the foregoing, the party {restrictive}, "
                    f"except with prior written consent of all parties."
                )
                insert_idx = min(idx + rng.randint(2, 5), len(sections) - 1)
                template = rng.choice(AMENDMENT_TEMPLATES_OBLIGATION)
                amendment_text = template.format(
                    orig_section=sec_idx + 1,
                    new_value=new_obligation,
                )
                return {
                    "type": "OBLIGATION_ADDITION",
                    "original_section_idx": sec_idx,
                    "original_value": permissive,
                    "new_value": restrictive,
                    "amendment_text": amendment_text,
                    "insert_after_section_idx": insert_idx,
                    "query": f"Is the party permitted to {permissive.replace('may ', '').replace('shall have the right to ', '')}?",
                    "gold_answer": f"No — {restrictive} (amended clause).",
                    "distractor_answer": f"Yes — {permissive}.",
                    "gold_span": amendment_text,
                    "distractor_span": permissive,
                }
    return None


def _inject_termination_change(
    sections: List[Tuple[int, str]], rng: random.Random
) -> Optional[Dict]:
    """Find a notice period; inject a later amendment changing it."""
    notice_re = re.compile(
        r"\b(\d+)\s*(days?|months?)\s*(?:written\s*)?notice", re.IGNORECASE
    )
    for idx, (sec_idx, sec_text) in enumerate(sections[:-2]):
        m = notice_re.search(sec_text)
        if not m:
            continue
        orig_value = m.group(0)
        # Pick a different notice period
        new_period = rng.choice([p for p in NOTICE_PERIODS
                                 if p.split()[0] != m.group(1)])
        new_value = f"{new_period} written notice"
        insert_idx = min(idx + rng.randint(2, 4), len(sections) - 1)
        template = rng.choice(AMENDMENT_TEMPLATES_NUMERIC)
        amendment_text = template.format(
            orig_section=sec_idx + 1,
            field="notice period",
            new_value=new_value,
        )
        return {
            "type": "TERMINATION_CHANGE",
            "original_section_idx": sec_idx,
            "original_value": orig_value,
            "new_value": new_value,
            "amendment_text": amendment_text,
            "insert_after_section_idx": insert_idx,
            "query": "What is the required notice period for termination?",
            "gold_answer": new_value,
            "distractor_answer": orig_value,
            "gold_span": amendment_text,
            "distractor_span": orig_value,
        }
    return None


# ---------------------------------------------------------------------------
# Injection dispatcher
# ---------------------------------------------------------------------------

_INJECTION_STRATEGIES = [
    _inject_numeric_amendment,
    _inject_date_amendment,
    _inject_party_substitution,
    _inject_obligation_addition,
    _inject_termination_change,
]


def _attempt_injection(
    sections: List[Tuple[int, str]],
    rng: random.Random,
) -> Optional[Dict]:
    """Try each strategy in shuffled order; return first success."""
    order = _INJECTION_STRATEGIES[:]
    rng.shuffle(order)
    for fn in order:
        result = fn(sections, rng)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_benchmark(
    source_documents: Dict[str, str],
    output_path: str,
    num_documents: int = 400,
    contradictions_per_doc: int = 5,
    validation_sample: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate the synthetic cross-page consistency benchmark.

    Targets 2 000+ items by default (400 docs × 5 contradictions each).

    Parameters
    ----------
    source_documents : dict
        {doc_id: full_text} — typically CUAD contracts from LegalBench-RAG.
    output_path : str
        Path for the output .jsonl benchmark file.
    num_documents : int
        Number of source documents to process (default 400).
    contradictions_per_doc : int
        Target injections per document (default 5).
    validation_sample : int
        Number of items to write to validation_sample.jsonl (default 100).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        All generated benchmark items.
    """
    rng = random.Random(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path = output_path.parent / "validation_sample.jsonl"
    stats_path = output_path.parent / "benchmark_stats.json"

    all_items: List[Dict] = []
    type_counter: Counter = Counter()
    gap_distribution: List[int] = []   # section-index gap between original and amendment

    doc_ids = list(source_documents.keys())[:num_documents]

    for doc_id in tqdm(doc_ids, desc="Generating synthetic benchmark"):
        text = source_documents[doc_id]
        sections = _split_sections(text)
        if len(sections) < 5:
            continue

        injections_done = 0
        max_tries = contradictions_per_doc * 3  # retry budget
        tries = 0
        while injections_done < contradictions_per_doc and tries < max_tries:
            tries += 1
            injection = _attempt_injection(sections, rng)
            if injection is None:
                continue

            # Build modified document
            modified_sections = list(sections)
            insert_pos = injection["insert_after_section_idx"]
            modified_sections.insert(
                insert_pos + 1,
                (9000 + injections_done, injection["amendment_text"]),
            )
            modified_text = "\n\n".join(s for _, s in modified_sections)

            # Sanity check: injected text must be exactly recoverable
            if injection["amendment_text"] not in modified_text:
                logger.warning(
                    f"Sanity check FAILED for doc={doc_id} type={injection['type']}; skipping"
                )
                continue

            section_gap = insert_pos - injection["original_section_idx"]
            gap_distribution.append(section_gap)

            item = {
                "item_id": f"{doc_id}_inj{injections_done}",
                "doc_id": doc_id,
                "injection_type": injection["type"],
                "query": injection["query"],
                "modified_document": modified_text,
                "original_section_idx": injection["original_section_idx"],
                "amendment_section_idx": insert_pos + 1,
                "section_gap": section_gap,
                "gold_answer": injection["gold_answer"],
                "distractor_answer": injection["distractor_answer"],
                "gold_span": injection["gold_span"],
                "distractor_span": injection["distractor_span"],
                "amendment_text": injection["amendment_text"],
                "original_value": injection["original_value"],
                "new_value": injection["new_value"],
            }
            all_items.append(item)
            type_counter[injection["type"]] += 1
            injections_done += 1

    logger.info(
        f"Generated {len(all_items)} synthetic benchmark items "
        f"from {len(doc_ids)} documents."
    )

    # Write main benchmark (exclude full modified_document to keep file size reasonable)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_items:
            row = {k: v for k, v in item.items() if k != "modified_document"}
            f.write(json.dumps(row) + "\n")

    # Write validation sample with full document for manual review
    sample = rng.sample(all_items, min(validation_sample, len(all_items)))
    with open(validation_path, "w", encoding="utf-8") as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")

    # Coverage statistics
    avg_gap = sum(gap_distribution) / max(len(gap_distribution), 1)
    stats = {
        "total_items": len(all_items),
        "source_documents_used": len(doc_ids),
        "type_breakdown": dict(type_counter),
        "avg_section_gap": round(avg_gap, 2),
        "min_section_gap": min(gap_distribution, default=0),
        "max_section_gap": max(gap_distribution, default=0),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Benchmark → {output_path}\n"
        f"Validation sample ({len(sample)} items) → {validation_path}\n"
        f"Coverage stats → {stats_path}\n"
        f"Type breakdown: {dict(type_counter)}\n"
        f"Avg section gap: {avg_gap:.1f}"
    )
    return all_items


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class SyntheticBenchmarkDataset:
    """Load a previously generated synthetic benchmark from disk."""

    def __init__(self, path: str, source_documents: Dict[str, str]):
        self.path = Path(path)
        self.source_documents = source_documents
        self.items: List[Dict] = []
        self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                doc_id = item["doc_id"]
                if doc_id in self.source_documents:
                    item["document"] = self.source_documents[doc_id]
                self.items.append(item)
        logger.info(f"Loaded {len(self.items)} synthetic benchmark items from {self.path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]

    def by_type(self, injection_type: str) -> List[Dict]:
        """Return all items of a specific injection type."""
        return [it for it in self.items if it.get("injection_type") == injection_type]
