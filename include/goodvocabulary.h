#ifndef GOODVOCABULARY_H
#define GOODVOCABULARY_H

#include "DBoW2/FGOOD.h"
#include "DBoW2/TemplatedVocabulary.h"
#include "DBoW2/TemplatedDatabase.h"

namespace ORB_SLAM3
{

typedef DBoW2::TemplatedVocabulary<DBoW2::FGOOD::TDescriptor, DBoW2::FGOOD>
  GOODVocabulary;

typedef DBoW2::TemplatedDatabase<DBoW2::FGOOD::TDescriptor, DBoW2::FGOOD>
  GOODDatabase;

} //namespace ORB_SLAM

#endif // GOODVOCABULARY_H

