// CPU qualification of contracted spherical CGTG and Coulomb AO derivative shells.
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <psi4/libmints/basisset.h>
#include <psi4/libmints/eri.h>
#include <psi4/libmints/gshell.h>
#include <psi4/libmints/integral.h>
#include <psi4/libmints/molecule.h>
#include <psi4/libmints/twobody.h>

namespace py = pybind11;
using Basis = std::shared_ptr<psi::BasisSet>;
using Data = py::array_t<double, py::array::c_style>;
using Geminal = std::vector<std::pair<double, double>>;

namespace {
std::unique_ptr<psi::TwoBodyAOInt> derivative_engine(
    psi::IntegralFactory& factory, const Geminal& geminal, const std::string& op) {
    if (op == "coulomb") return factory.eri(1, false);
    // Set the zero screening threshold and first derivative as separate arguments.
    if (op == "f12")
        return std::make_unique<psi::Libint2F12>(geminal, &factory, 0., 1, false);
    if (op == "f12_over_coulomb")
        return std::make_unique<psi::Libint2F12G12>(geminal, &factory, 0., 1, false);
    if (op == "f12_squared")
        return std::make_unique<psi::Libint2F12Squared>(geminal, &factory, 0., 1, false);
    throw std::invalid_argument("Unsupported analytic operator");
}

Data contract(Basis aux, Basis combined, const std::string& op, const Geminal& geminal,
              bool metric, const Data& weights) {
    auto zero = psi::BasisSet::zero_ao_basis_set();
    Basis right = metric ? aux : combined;
    psi::IntegralFactory factory(aux, zero, right, metric ? zero : combined);
    auto engine = derivative_engine(factory, geminal, op);
    const int n_aux = aux->nbf(), n_right = right->nbf();
    const int n_atoms = combined->molecule()->natom();
    if (aux->molecule()->natom() != n_atoms)
        throw std::invalid_argument("Auxiliary and combined bases must have the same atom ordering");
    for (int atom = 0; atom < n_atoms; ++atom) {
        if (aux->molecule()->x(atom) != combined->molecule()->x(atom) ||
            aux->molecule()->y(atom) != combined->molecule()->y(atom) ||
            aux->molecule()->z(atom) != combined->molecule()->z(atom))
            throw std::invalid_argument("Auxiliary centers differ from the combined basis geometry");
    }
    if (weights.ndim() != (metric ? 3 : 4) || weights.shape(1) != n_aux ||
        weights.shape(2) != n_right || (!metric && weights.shape(3) != n_right))
        throw std::invalid_argument("AO weight shape differs from operator basis");

    const py::ssize_t n_seeds = weights.shape(0);
    Data result({n_seeds, py::ssize_t(n_atoms), py::ssize_t(3)});
    auto* gradient = result.mutable_data();
    std::fill_n(gradient, result.size(), 0.);
    const auto& buffers = engine->buffers();
    const auto* seed_values = weights.data();
    const py::ssize_t stride = weights.strides(0) / sizeof(double);
    const int n_centers = metric ? 2 : 3;
    // Iterate all ordered shells because the transpose of the fit produces
    // unsymmetrized Frobenius weights, even for symmetric AO integral tensors.
    for (int p = 0; p < aux->nshell(); ++p) {
        const auto& P = aux->shell(p);
        for (int q = 0; q < right->nshell(); ++q) {
            const auto& Q = right->shell(q);
            for (int r = 0; r < (metric ? 1 : right->nshell()); ++r) {
                const auto& R = metric ? zero->shell(0) : right->shell(r);
                // A common translation of all centers leaves the integral invariant.
                if (P.ncenter() == Q.ncenter() && (metric || Q.ncenter() == R.ncenter()))
                    continue;
                const std::array<int, 3> centers{P.ncenter(), Q.ncenter(), R.ncenter()};
                engine->compute_shell_deriv1(p, 0, q, r);
                if (buffers.size() < static_cast<size_t>(3 * n_centers))
                    throw std::runtime_error("Missing analytic derivative shell buffers");
                int local = 0;
                for (int i = 0; i < P.nfunction(); ++i) {
                    for (int j = 0; j < Q.nfunction(); ++j) {
                        for (int k = 0; k < R.nfunction(); ++k, ++local) {
                            auto index = py::ssize_t(P.function_index() + i) * n_right
                                         + Q.function_index() + j;
                            if (!metric) index = index * n_right + R.function_index() + k;
                            for (py::ssize_t s = 0; s < n_seeds; ++s) {
                                const double weight = seed_values[s * stride + index];
                                for (int center = 0; center < n_centers; ++center) {
                                    auto* force = gradient + (s * n_atoms + centers[center]) * 3;
                                    for (int axis = 0; axis < 3; ++axis) {
                                        const double* block = buffers[3 * center + axis];
                                        if (block) force[axis] += weight * block[local];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}
} // namespace

PYBIND11_MODULE(f12_psi4_geminal_nuclear, module) {
    module.def("contract", &contract, py::arg("aux"), py::arg("combined"),
               py::arg("op"), py::arg("geminal"), py::arg("metric"), py::arg("weights"),
               "Contract ordered two- or three-center derivative weights to Cartesian gradients.");
}
