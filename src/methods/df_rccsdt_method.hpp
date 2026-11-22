#pragma once

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

double focal_rcc_triples_energy(
    PreparedCalculation&, const std::shared_ptr<const hf::PhysicalReference>&,
    const molecule::ScalarHamiltonianIdentity& orbital,
    const molecule::ScalarHamiltonianIdentity& jk, std::size_t retained_reference_bytes);

}  // namespace generativeqc::methods::detail
