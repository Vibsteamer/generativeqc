#pragma once

#include "hf/reference.hpp"
#include "methods/method.hpp"

namespace generativeqc::methods::detail {

/** Common conventional UHF admission for native UCCSD and UCCSD(T). */
generativeqc_status validate_uccsd_system(generativeqc_method method, const core::System& system,
                                          std::string& detail);

/** Prepare the resource-bounded native CPU/CUDA unrestricted energy/force owner.
 * Optional fitted operators preserve their explicitly selected UHF reference;
 * unsupported requests never select prototypes or reference-oracle work. */
std::unique_ptr<PreparedCalculation> prepare_uccsd_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor,
    const core::System* auxiliary_basis = nullptr);

std::unique_ptr<PreparedCalculation> prepare_uccsd_with_reference_fit(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor, const core::System* correlation_auxiliary,
    const core::System* reference_auxiliary);

/** Conventional triples on an already qualified molecular determinant and fitting model. */
Result focal_ucc_triples(PreparedCalculation& plan,
                         const std::shared_ptr<const hf::UnrestrictedPhysicalReference>& reference,
                         const molecule::ScalarHamiltonianIdentity& jk, bool forces);

}  // namespace generativeqc::methods::detail

namespace generativeqc::methods {
std::unique_ptr<PreparedCalculation> prepare_split_uccsd_calculation(
    core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor, const core::System* correlation_auxiliary,
    const core::System& reference_auxiliary);
}  // namespace generativeqc::methods
