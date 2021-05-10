#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ndarray_converter.h"
#include "FGOOD.h"
#include "TemplatedVocabulary.h"
#include "TemplatedDatabase.h"
#include "QueryResults.h"



namespace py = pybind11;


typedef DBoW2::TemplatedVocabulary<DBoW2::FGOOD::TDescriptor, DBoW2::FGOOD>
  GOODVocabulary;

typedef DBoW2::TemplatedDatabase<DBoW2::FGOOD::TDescriptor, DBoW2::FGOOD>
  GOODDatabase;


PYBIND11_MAKE_OPAQUE(std::vector<DBoW2::Result>);

PYBIND11_MODULE(DBoW, m) {
    NDArrayConverter::init_numpy();

    py::class_<GOODVocabulary>(m, "GOODVocabulary")
        .def(py::init<>())
        .def("load_json", &GOODVocabulary::load_json);

    py::class_<DBoW2::FeatureVector>(m, "FeatureVector")
        .def(py::init<>());

    py::class_<DBoW2::Result>(m, "Result")
    .def_readwrite("Id", &DBoW2::Result::Id)
    .def_readwrite("Score", &DBoW2::Result::Score)
    .def_readwrite("nWords", &DBoW2::Result::nWords);

    py::class_<std::vector<DBoW2::Result> >(m, "ResultVector")
        .def(py::init<>())
        .def("clear", &std::vector<DBoW2::Result>::clear)
        .def("pop_back", &std::vector<DBoW2::Result>::pop_back)
        .def("__len__", [](const std::vector<DBoW2::Result> &v) { return v.size(); })
        .def("__iter__", [](std::vector<DBoW2::Result> &v) {
           return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0, 1>());

    py::class_<DBoW2::QueryResults, std::vector<DBoW2::Result> >(m, "QueryResults");

    py::class_<GOODDatabase>(m, "GOODDatabase")
        .def(py::init<const GOODVocabulary &, bool, int>())
        .def("add",   static_cast<DBoW2::EntryId (GOODDatabase::*)(const std::vector<GOODDatabase::TDesc> &)> (&GOODDatabase::add),
			"add an item to DBoW\n \
			features - list of cv::Mat, each cv::Mat 1x256 matrix",
             py::arg("features"))

        .def("query", static_cast<DBoW2::QueryResults (GOODDatabase::*)(const std::vector<GOODDatabase::TDesc> &, int, int) const>
	        (&GOODDatabase::query) , "query database with a feature matrix",
             py::arg("features"), py::arg("max_results")=1, py::arg("max_id")=-1)

	.def("size", &GOODDatabase::size);

}
