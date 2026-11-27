"""Compile the actual host/resident controllers against a deterministic backend."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

SOURCE = r"""
#include "resident_gmres_test_backend.hpp"

int main() {
  try {
    const auto plan = prepare_gmres(2, GmresOptions{});
    const auto count = resident_gmres_workspace(plan).vector_slots;
    const std::array<double, 2> rhs{1.0, 2.0}, guess{0.5, 0.5};
    const double maximum = std::numeric_limits<double>::max();
    const std::array<double, 2> huge{maximum, maximum};

    Backend input(count);
    const auto refused = solve_gmres_resident(plan, input, huge, guess);
    require(refused.result.status == GmresStatus::nonfinite_input,
            "overflowing RHS norm did not return nonfinite_input");
    require(input.uploads == 0 && input.actions == 0 && input.downloads == 0,
            "invalid RHS norm reached device execution");
    require(std::equal(guess.begin(), guess.end(), refused.result.solution.begin()),
            "invalid RHS did not preserve the supplied initial guess");

    Backend initial(count);
    initial.nan_operator = true;
    const auto invalid = solve_gmres_resident(plan, initial, rhs, guess);
    require(invalid.result.status == GmresStatus::nonfinite_operator,
            "nonfinite initial operator residual escaped its failure status");
    require(initial.downloads == 1, "failure must publish one final solution");

    for (std::size_t checkpoint : {2U, 3U}) {
      Backend backend(count);
      backend.fail_norm = checkpoint;
      const auto failed = solve_gmres_resident(plan, backend, rhs);
      require(failed.result.status == GmresStatus::nonfinite_operator,
              "Arnoldi/candidate norm failure escaped its failure status");
      require(backend.downloads == 1, "checkpoint failure downloaded more than once");
    }

    Backend valid(count);
    const auto solved = solve_gmres_resident(plan, valid, rhs);
    require(solved.converged() && valid.downloads == 1,
            "valid resident identity solve regressed");

    const LinearResponseProblem problem(2, [](auto input, auto output) {
      output[0] = 2.0 * input[0];
      output[1] = 4.0 * input[1];
    });
    const auto bound = prepare_response(problem, GmresOptions{});
    Backend preconditioned(count, bound);
    preconditioned.diagonal_operator = true;
    preconditioned.right_preconditioned = true;
    const auto accelerated = solve_response_resident(bound, problem, preconditioned, rhs);
    require(accelerated.converged() && accelerated.result.iterations == 1 &&
                accelerated.result.preconditioner_actions == 1,
            "resident right preconditioning failed");
    require(std::abs(accelerated.result.solution[0] - 0.5) < 1e-12 &&
                std::abs(accelerated.result.solution[1] - 0.5) < 1e-12,
            "right preconditioning changed the physical solution");
    require(preconditioned.uploads == 1 && preconditioned.downloads == 1,
            "resident preconditioner staged an iteration vector");
    preconditioned.nan_preconditioner = true;
    const auto rejected = solve_response_resident(bound, problem, preconditioned, rhs);
    require(rejected.result.status == GmresStatus::nonfinite_preconditioner &&
                rejected.result.operator_actions == 0 &&
                rejected.result.preconditioner_actions == 1,
            "nonfinite preconditioner reached the physical operator");
    preconditioned.nan_preconditioner = false;
    require(solve_response_resident(bound, problem, preconditioned, rhs).converged(),
            "resident backend could not recover after preconditioner failure");

    const LinearResponseProblem nonsymmetric(2, [](auto input, auto output) {
      output[0] = 2.0 * input[0] + input[1];
      output[1] = 4.0 * input[1];
    });
    GmresOptions exact_options;
    const auto workspace = resident_gmres_workspace(bound);
    const auto exact_bytes = workspace.host_scalar_bytes + workspace.host_result_bytes;
    for (const auto budget : {exact_bytes, exact_bytes + 64}) {
      exact_options.max_workspace_bytes = budget;
      const auto nonsymmetric_plan = prepare_response(nonsymmetric, exact_options);
      Backend triangular(count, nonsymmetric_plan);
      triangular.diagonal_operator = triangular.upper_operator = true;
      triangular.right_preconditioned = true;
      const auto physical = solve_response_resident(
          nonsymmetric_plan, nonsymmetric, triangular, rhs);
      require(physical.converged() && physical.result.residual_norm < 1e-12 &&
                  std::abs(physical.result.solution[0] - 0.25) < 1e-12 &&
                  std::abs(physical.result.solution[1] - 0.5) < 1e-12,
              "resident right preconditioner changed a nonsymmetric problem");
      require(triangular.uploads == 1 && triangular.downloads == 1 &&
                  physical.result.measured_workspace_peak_bytes <= budget,
              "resident exact host workspace/transfer contract");
    }
    exact_options.max_workspace_bytes = exact_bytes - 1;
    const auto insufficient_plan = prepare_response(nonsymmetric, exact_options);
    Backend insufficient(count, insufficient_plan);
    const auto insufficient_result = solve_response_resident(
        insufficient_plan, nonsymmetric, insufficient, rhs);
    require(insufficient_result.result.status == GmresStatus::workspace_limit &&
                insufficient.uploads == 0 && insufficient.actions == 0,
            "resident workspace refusal performed device work");

    const LinearResponseProblem other(2, [](auto input, auto output) {
      std::copy(input.begin(), input.end(), output.begin());
    });
    const auto other_plan = prepare_response(other, GmresOptions{});
    const auto before_uploads = preconditioned.uploads;
    for (bool wrong_backend : {false, true}) {
      bool threw = false;
      try {
        if (wrong_backend)
          solve_response_resident(other_plan, other, preconditioned, rhs);
        else
          solve_response_resident(bound, other, preconditioned, rhs);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      require(threw && preconditioned.uploads == before_uploads,
              "wrong resident problem binding reached uploads");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""


def test_resident_gmres_preserves_numerical_failure_statuses(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "resident_gmres_failures.cpp"
    binary = tmp_path / "resident_gmres_failures"
    source.write_text(SOURCE, encoding="utf-8")
    native_cxx.build_executable(
        [
            source,
            root / "src/response/native_gmres.cpp",
            root / "src/response/resident_gmres.cpp",
        ],
        binary,
        compile_args=(
            "-std=c++20",
            "-O0",
            "-pthread",
            "-I",
            str(root / "src"),
            "-I",
            str(root / "include"),
            "-I",
            str(root / "tests/native"),
        ),
        link_args=("-pthread",),
    )
    subprocess.run([str(binary)], check=True, timeout=30)
