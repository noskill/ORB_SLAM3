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

    py::class_<DBoW2::QueryResults>(m, "QueryResults");
    
    py::class_<GOODDatabase>(m, "GOODDatabase")
        .def(py::init<const GOODVocabulary &, bool, int>())
        .def("add",   static_cast<DBoW2::EntryId (GOODDatabase::*)(const DBoW2::BowVector &, const DBoW2::FeatureVector&)> (&GOODDatabase::add), "add an item to DBoW",
             py::arg("features"), py::arg("fec") = DBoW2::FeatureVector())

        .def("query", static_cast<DBoW2::QueryResults (GOODDatabase::*)(const DBoW2::BowVector &, int, int) const>
        (&GOODDatabase::query) , "query database with a feature matrix",
             py::arg("vec"), py::arg("max_results")=1, py::arg("max_id")=-1);

}
