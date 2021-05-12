#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ndarray_converter.h"
#include "FORB.h"
#include "FGOOD.h"
#include "TemplatedVocabulary.h"
#include "TemplatedDatabase.h"
#include "QueryResults.h"



namespace py = pybind11;


typedef DBoW2::TemplatedVocabulary<DBoW2::FGOOD::TDescriptor, DBoW2::FGOOD>
  GOODVocabulary;

typedef DBoW2::TemplatedDatabase<DBoW2::FGOOD::TDescriptor, DBoW2::FGOOD>
  GOODDatabase;

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBVocabulary;

typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBDatabase;


#define VOCAB(X, Name)  \
    py::class_<X>(m, Name) \
        .def(py::init<>()) \
        .def("load_json", &X::load_json) \
        .def("load", static_cast<void(X::*)(const std::string &)>(&X::load));

#define DB(X, Name, Voc) \
    py::class_<X>(m, Name) \
        .def(py::init<const Voc &, bool, int>()) \
        .def("add",   static_cast<DBoW2::EntryId (X::*)(const std::vector<X::TDesc> &)> (&X::add), \
			"add an item to DBoW\nfeatures - list of cv::Mat, each cv::Mat 1x256 matrix", \
             py::arg("features")) \
        .def("query", static_cast<DBoW2::QueryResults (X::*)(const std::vector<X::TDesc> &, int, int) const> \
	        (&X::query) , "query database with a feature matrix", \
             py::arg("features"), py::arg("max_results")=1, py::arg("max_id")=-1) \
	.def("size", &X::size);


PYBIND11_MAKE_OPAQUE(std::vector<DBoW2::Result>);

PYBIND11_MODULE(DBoW, m) {
    NDArrayConverter::init_numpy();

    VOCAB(GOODVocabulary, "GOODVocabulary")
    VOCAB(ORBVocabulary, "ORBVocabulary")

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

    DB(GOODDatabase, "GOODDatabase", GOODVocabulary)
    DB(ORBDatabase, "ORBDatabase", ORBVocabulary)
}
