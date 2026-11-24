"""VIB5 molecular stages must run the checked graph sizes of their own orbital basis."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.f12_vib5_directional import check_graph_budget


class VIB5GraphBudgetTest(unittest.TestCase):
    def test_archived_d_panel(self) -> None:
        case = {"layout": {"space": [5, 66, 220]}}
        check_graph_budget(case, Path("panel.json"), Path("layout.npz"), 4096)
        with self.assertRaisesRegex(ValueError, "missing VIB5 graph preflight"):
            check_graph_budget(case, Path("panel.json"), Path("layout.npz"), 8192)

    def test_larger_graph_with_matched_layout(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            folder = Path(work)
            path = folder / "graph_plan.json"
            layout = folder / "layout.npz"
            panel = folder / "panel.json"
            np.savez(layout, ao_sizes=np.array([364, 288]))
            space = [5, 125, 364]
            peak = 8392942208
            path.write_text(
                json.dumps(
                    {
                        "space": space,
                        "fitting_aos": 288,
                        "per_graph": {"ao_sources_reverse": {"combined_bytes": peak}},
                    }
                )
            )
            case = {
                "layout": {"space": space},
                "native_graph_plan": {
                    "space": space,
                    "fitting_aos": 288,
                    "budget_mib": 8192,
                    "largest_combined_bytes": peak,
                    "graph_plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            }
            check_graph_budget(case, panel, layout, 8192)
            with self.assertRaisesRegex(ValueError, "budget"):
                check_graph_budget(case, panel, layout, 4096)
            np.savez(layout, ao_sizes=np.array([220, 288]))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                check_graph_budget(case, panel, layout, 8192)
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash"):
                check_graph_budget(case, panel, layout, 8192)


if __name__ == "__main__":
    unittest.main()
