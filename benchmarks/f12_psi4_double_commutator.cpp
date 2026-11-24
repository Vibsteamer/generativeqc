// Stream raw double-commutator nuclear derivative blocks in Psi4's spherical AO order.
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <psi4/libmints/basisset.h>
#include <psi4/libmints/eri.h>
#include <psi4/libmints/gshell.h>
#include <psi4/libmints/integral.h>
#include <psi4/libmints/molecule.h>

namespace py = pybind11;

class ShellDerivatives {
    using Basis = std::shared_ptr<psi::BasisSet>;
    using Geminal = std::vector<std::pair<double, double>>;
    Basis aux_, combined_, right_, fourth_, zero_;
    bool metric_;
    psi::IntegralFactory factory_;
    psi::Libint2F12DoubleCommutator engine_;

 public:
    ShellDerivatives(Basis aux, Basis combined, const Geminal& geminal, bool metric)
        : aux_(aux), combined_(combined), right_(metric ? aux : combined),
          fourth_(metric ? psi::BasisSet::zero_ao_basis_set() : combined),
          zero_(psi::BasisSet::zero_ao_basis_set()), metric_(metric),
          factory_(aux_, zero_, right_, fourth_), engine_(geminal, &factory_, 0., 1, false) {
        const auto a = aux_->molecule(), b = combined_->molecule();
        if (a->natom() != b->natom()) throw std::invalid_argument("Geometry mismatch");
        for (int i = 0; i < a->natom(); ++i)
            if (a->x(i) != b->x(i) || a->y(i) != b->y(i) || a->z(i) != b->z(i))
                throw std::invalid_argument("Auxiliary and combined centers differ");
    }

    py::array_t<double> block(int p) {
        if (p < 0 || p >= aux_->nshell()) throw std::out_of_range("Auxiliary shell");
        const auto& P = aux_->shell(p);
        const int nat = combined_->molecule()->natom(), nb = right_->nbf();
        std::vector<py::ssize_t> shape{nat, 3, P.nfunction(), nb};
        if (!metric_) shape.push_back(nb);
        py::array_t<double> out(shape);
        std::fill_n(out.mutable_data(), out.size(), 0.);
        const py::ssize_t stride = out.strides(1) / sizeof(double);
        const int ncenters = metric_ ? 2 : 3;
        const auto& buffers = engine_.buffers();
        for (int q = 0; q < right_->nshell(); ++q) {
            const auto& Q = right_->shell(q);
            for (int r = 0; r < fourth_->nshell(); ++r) {
                const auto& R = fourth_->shell(r);
                if (P.ncenter() == Q.ncenter() && (metric_ || P.ncenter() == R.ncenter()))
                    continue;
                engine_.compute_shell_deriv1(p, 0, q, r);
                if (buffers.size() < static_cast<size_t>(3*ncenters))
                    throw std::runtime_error("Missing double-commutator derivative buffers");
                std::array<int,3> centers{P.ncenter(), Q.ncenter(), R.ncenter()};
                for (int center = 0; center < ncenters; ++center) {
                    for (int axis = 0; axis < 3; ++axis) {
                        const double* values = buffers[3*center+axis];
                        if (!values) continue;
                        double* tile = out.mutable_data() + (3*centers[center]+axis)*stride;
                        int local = 0;
                        for (int i = 0; i < P.nfunction(); ++i) {
                            for (int j = 0; j < Q.nfunction(); ++j) {
                                auto index = py::ssize_t(i)*nb + Q.function_index()+j;
                                if (!metric_) index = index*nb + R.function_index();
                                for (int k = 0; k < R.nfunction(); ++k)
                                    tile[index+k] += values[local++];
                            }
                        }
                    }
                }
            }
        }
        return out;
    }
};

PYBIND11_MODULE(f12_psi4_double_commutator, m) {
    py::class_<ShellDerivatives>(m, "ShellDerivatives")
        .def(py::init<std::shared_ptr<psi::BasisSet>, std::shared_ptr<psi::BasisSet>,
                      const std::vector<std::pair<double, double>>&, bool>())
        .def("block", &ShellDerivatives::block, py::arg("auxiliary_shell"),
             "All nuclear derivatives for one auxiliary shell, in ordered spherical AO coordinates.");
}
