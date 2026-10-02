// Python bindings of native simulation modules. A module source declares its protocols with
// GRADYSIM_PROTOCOLS({"name", gradysim::make<Class>}, ...) and is compiled by gradysim.native.build.
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine.hpp"

namespace gradysim {

namespace py = pybind11;

inline py::dict simulate(const std::map<std::string, ProtocolFactory> &registry,
                         const std::vector<std::string> &protocol_names, const std::vector<int> &kinds,
                         py::array_t<double, py::array::c_style | py::array::forcecast> positions_array,
                         const py::dict &configuration_dict, const std::map<std::string, double> &params,
                         const py::dict &arrays_dict, int message_size) {
    auto positions_view = positions_array.unchecked<2>();
    if (positions_view.shape(0) != static_cast<py::ssize_t>(kinds.size()) || positions_view.shape(1) != 3)
        throw std::invalid_argument("positions must have shape (nodes, 3)");
    std::vector<Vec3> positions;
    for (py::ssize_t node = 0; node < positions_view.shape(0); ++node)
        positions.push_back(Vec3{positions_view(node, 0), positions_view(node, 1), positions_view(node, 2)});
    for (int kind : kinds)
        if (kind < 0 || kind >= static_cast<int>(protocol_names.size()))
            throw std::invalid_argument("Every kind must be the index of a protocol");

    Configuration configuration;
    configuration.duration = configuration_dict["duration"].cast<double>();
    configuration.transmission_range = configuration_dict["transmission_range"].cast<double>();
    configuration.delay = configuration_dict["delay"].cast<double>();
    configuration.mobility = configuration_dict["mobility"].cast<bool>();
    configuration.update_rate = configuration_dict["update_rate"].cast<double>();
    configuration.telemetry_decimation = configuration_dict["telemetry_decimation"].cast<int64_t>();
    configuration.timer_ids = configuration_dict["timer_ids"].cast<int>();
    configuration.seed = configuration_dict["seed"].cast<uint64_t>();

    std::map<std::string, Array> arrays;
    for (auto item : arrays_dict) {
        auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(item.second);
        if (!array) throw std::invalid_argument("Arrays must be convertible to float64");
        Array converted;
        converted.values.assign(array.data(), array.data() + array.size());
        for (py::ssize_t dimension = 0; dimension < array.ndim(); ++dimension)
            converted.shape.push_back(array.shape(dimension));
        arrays[item.first.cast<std::string>()] = std::move(converted);
    }

    Outputs outputs;
    Setup setup(static_cast<int>(kinds.size()), params, std::move(arrays), outputs);
    std::vector<std::unique_ptr<Protocol>> owned;
    std::vector<Protocol *> protocols;
    for (const std::string &name : protocol_names) {
        auto found = registry.find(name);
        if (found == registry.end()) throw std::invalid_argument("Unknown protocol: " + name);
        owned.push_back(found->second(setup));
        protocols.push_back(owned.back().get());
    }

    Context context(configuration, kinds, positions);
    context.set_message_size(message_size);
    Context::Result result;
    {
        py::gil_scoped_release release;
        result = context.run(protocols);
    }

    py::array_t<double> final_positions({static_cast<py::ssize_t>(result.positions.size()), py::ssize_t{3}});
    auto final_view = final_positions.mutable_unchecked<2>();
    for (size_t node = 0; node < result.positions.size(); ++node) {
        final_view(node, 0) = result.positions[node].x;
        final_view(node, 1) = result.positions[node].y;
        final_view(node, 2) = result.positions[node].z;
    }
    py::dict output_arrays;
    for (auto &[name, values] : outputs) output_arrays[py::str(name)] = py::array_t<double>(values.size(), values.data());

    py::dict returned;
    returned["events"] = result.events;
    returned["time"] = result.time;
    returned["positions"] = final_positions;
    returned["outputs"] = output_arrays;
    return returned;
}

}  // namespace gradysim

#ifndef GRADYSIM_MODULE_NAME
#error "Compile native modules with gradysim.native.build, which defines GRADYSIM_MODULE_NAME"
#endif

#define GRADYSIM_PROTOCOLS(...)                                                                                  \
    PYBIND11_MODULE(GRADYSIM_MODULE_NAME, module) {                                                             \
        static const std::map<std::string, gradysim::ProtocolFactory> registry{__VA_ARGS__};                    \
        module.def(                                                                                              \
            "simulate",                                                                                          \
            [](const std::vector<std::string> &names, const std::vector<int> &kinds,                             \
               pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> positions,       \
               const pybind11::dict &configuration, const std::map<std::string, double> &params,                \
               const pybind11::dict &arrays, int message_size) {                                                 \
                return gradysim::simulate(registry, names, kinds, positions, configuration, params, arrays,     \
                                          message_size);                                                         \
            });                                                                                                  \
        module.def("protocols", []() {                                                                           \
            std::vector<std::string> names;                                                                      \
            for (const auto &entry : registry) names.push_back(entry.first);                                    \
            return names;                                                                                        \
        });                                                                                                      \
    }
