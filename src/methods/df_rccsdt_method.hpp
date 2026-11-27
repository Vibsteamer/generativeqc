#pragma once

#include "cc/solver.hpp"
#include "core/correlation_execution.hpp"
#include "methods/method.hpp"

namespace generativeqc::hf {
struct PhysicalReference;
}
namespace generativeqc::molecule {
class ScalarHamiltonianIdentity;
}

namespace generativeqc::methods::detail {

generativeqc_status validate_df_rccsdt_system(generativeqc_method, const core::System&,
                                              std::string&);
std::unique_ptr<PreparedCalculation> prepare_df_rccsdt_calculation(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&, const core::System* auxiliary_basis = nullptr);

struct FocalRccF12bOptions {
  bool forces{};
  std::size_t frozen_occupied{};
  cc::FixedCuspF12 f12;
  core::DfPreconvergence preconvergence{core::DfPreconvergence::Automatic};
};

struct FocalRccF12bValues {
  double energy{};
  std::vector<double> t1, t2, lambda1, lambda2;
  std::vector<double> correlation_gradient, bar_f, bar_c;
  bool ccsd_fp32_executed{}, lambda_fp32_executed{};
  std::size_t ccsd_iterations{}, lambda_iterations{};
  std::size_t f12_cuda_preparations{}, f12_cuda_rhs_evaluations{};
  double ccsd_seconds{}, lambda_seconds{}, source_response_seconds{}, total_seconds{};
};

FocalRccF12bValues focal_rcc_f12b_values(
    PreparedCalculation&, const std::shared_ptr<const hf::PhysicalReference>&,
    const molecule::ScalarHamiltonianIdentity& orbital,
    const molecule::ScalarHamiltonianIdentity& jk, std::size_t retained_reference_bytes,
    FocalRccF12bOptions options);

}  // namespace generativeqc::methods::detail
