"""The join layer: problems + budgets + hidden states, aligned on `id`.

This is the only module that knows what the artifact files look like. Feature
blocks and models talk to `OracleDataset`, so changing a file format means
changing this file and nothing else.

The join is an inner join, and it's deliberately loud about what it drops. A
silent mismatch -- extracting states for 300 problems but labeling 80, then
training on 12 because of an id format change -- is the kind of thing that
produces a plausible-looking but meaningless number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from frugalprover.common.io import read_jsonl, read_table
from frugalprover.common.records import BudgetRecord, ProblemRecord

#: Pooled vector columns look like "L14_mean"; geometry scalars like
#: "L14_l2_norm". Both start with L<digits>_, so they're told apart by dtype
#: (array vs scalar) rather than by parsing the suffix.
LAYER_COL_RE = re.compile(r"^L(\d+)_(.+)$")


@dataclass
class OracleDataset:
    """Problems aligned with their hidden states and (optionally) budget labels."""

    problems: list[ProblemRecord]
    hidden: "object" = None                      # pd.DataFrame indexed by id, or None
    budgets: dict[str, BudgetRecord] = field(default_factory=dict)

    # ------------------------------------------------------------------ load

    @classmethod
    def load(
        cls,
        problems: str,
        hidden_states: str | None = None,
        budgets: str | None = None,
        verbose: bool = True,
    ) -> OracleDataset:
        """Read the artifacts and inner-join them on `id`."""
        prob_records = [ProblemRecord.from_dict(d) for d in read_jsonl(problems)]
        by_id = {p.id: p for p in prob_records}
        keep = set(by_id)
        counts = {"problems": len(prob_records)}

        hidden_df = None
        if hidden_states is not None:
            hidden_df = read_table(hidden_states)
            if "id" not in hidden_df.columns:
                raise ValueError(
                    f"{hidden_states} has no 'id' column, so it can't be joined to "
                    f"problems. See docs/ARTIFACTS.md (A3)."
                )
            hidden_df = hidden_df.set_index("id")
            counts["hidden_states"] = len(hidden_df)
            keep &= set(hidden_df.index)

        budget_map: dict[str, BudgetRecord] = {}
        if budgets is not None:
            budget_records = [BudgetRecord.from_dict(d) for d in read_jsonl(budgets)]
            budget_map = {r.id: r for r in budget_records}
            counts["budgets"] = len(budget_map)
            keep &= set(budget_map)

        if not keep:
            raise ValueError(
                f"no overlapping ids across artifacts ({counts}). Ids must match "
                f"exactly across files - if these were produced by different "
                f"sampling runs, regenerate them from one."
            )

        aligned = [by_id[i] for i in sorted(keep)]
        if verbose:
            dropped = {k: v - len(aligned) for k, v in counts.items() if v != len(aligned)}
            print(f"joined {len(aligned)} problems" + (f" (dropped: {dropped})" if dropped else ""))

        return cls(
            problems=aligned,
            hidden=hidden_df.loc[[p.id for p in aligned]] if hidden_df is not None else None,
            budgets={p.id: budget_map[p.id] for p in aligned} if budget_map else {},
        )

    # -------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self.problems)

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.problems]

    def activations(self, layer: str) -> np.ndarray:
        """The pooled vectors at `layer` as an (n, hidden_size) array."""
        if self.hidden is None:
            raise ValueError("this dataset has no hidden states")
        if layer not in self.hidden.columns:
            raise ValueError(
                f"column {layer!r} not in hidden states. Available: {self.pooled_columns()}"
            )
        return np.stack([np.asarray(v, dtype=float) for v in self.hidden[layer]])

    def scalars(self, columns: list[str]) -> np.ndarray:
        return self.hidden[columns].to_numpy(dtype=float)

    def _columns_by_kind(self, vector: bool) -> list[str]:
        """Split L*_ columns into pooled vectors vs geometry scalars by looking
        at the first cell's type -- more robust than matching suffix names,
        which a custom extractor is free to change."""
        if self.hidden is None or len(self.hidden) == 0:
            return []
        out = []
        for col in self.hidden.columns:
            if not LAYER_COL_RE.match(col):
                continue
            value = self.hidden[col].iloc[0]
            is_vector = isinstance(value, (list, tuple, np.ndarray))
            if is_vector == vector:
                out.append(col)
        return sorted(out, key=_layer_sort_key)

    def pooled_columns(self) -> list[str]:
        """Candidate layers for the sweep, in layer order."""
        return self._columns_by_kind(vector=True)

    def geometry_columns(self) -> list[str]:
        return self._columns_by_kind(vector=False)

    # ---------------------------------------------------------------- labels

    def has_budgets(self) -> bool:
        return bool(self.budgets)

    def swept_budgets(self) -> list[int]:
        """Union of budgets across records. A problem missing from some budgets
        is fine -- classification just gets fewer rows for it."""
        return sorted({b for r in self.budgets.values() for b in r.budgets})

    def single_pass(self) -> bool:
        """True when only one budget was ever tried, so b_star is a
        solved/unsolved label and regression is not meaningful."""
        return len(self.swept_budgets()) <= 1

    def b_star(self) -> np.ndarray:
        """B* per problem, NaN where censored."""
        return np.array(
            [
                self.budgets[p.id].b_star if self.budgets[p.id].b_star is not None else np.nan
                for p in self.problems
            ],
            dtype=float,
        )

    def levels(self) -> np.ndarray:
        """Human difficulty labels. For stratifying and plots only -- never a
        model feature (see features.py)."""
        return np.array([p.level if p.level is not None else np.nan for p in self.problems], dtype=float)

    def subset(self, ids: list[str]) -> OracleDataset:
        """A new dataset over a subset of ids, preserving order."""
        keep = [i for i in ids if i in set(self.ids)]
        by_id = {p.id: p for p in self.problems}
        return OracleDataset(
            problems=[by_id[i] for i in keep],
            hidden=self.hidden.loc[keep] if self.hidden is not None else None,
            budgets={i: self.budgets[i] for i in keep if i in self.budgets},
        )

    def solved_subset(self) -> OracleDataset:
        """Only problems with a real B*. Regression trains on these; censored
        problems have no target to regress onto."""
        return self.subset([p.id for p in self.problems if self.budgets[p.id].b_star is not None])


def _layer_sort_key(col: str) -> tuple:
    m = LAYER_COL_RE.match(col)
    return (int(m.group(1)), m.group(2)) if m else (10**9, col)
