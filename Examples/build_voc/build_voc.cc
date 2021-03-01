#include <ostream>
#include <iostream>
#include <DBoW2/BowVector.h>
#include "DBoW2/FGOOD.h"
#include <opencv2/opencv.hpp>
#include "goodvocabulary.h"

using namespace DBoW2;

std::vector<std::vector<cv::Mat > > loadFeatures();
void createVoc(const std::vector<std::vector<cv::Mat > > & features);
void testDatabase(const std::vector<std::vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, std::vector<cv::Mat > &out);

typedef std::vector<std::vector<cv::Mat > > feature_vector;


feature_vector loadFeatures()
{
  feature_vector features;
  features.clear();
  features.reserve(10000);

  std::string delimiter = " ";
  std::cout << "loading GOODPOINT features..." << std::endl;
  long img_num = 0;
  long line_num = 0;
  long start = 0;
  cv::Mat descriptors;
  for (std::string line; std::getline(std::cin, line);) {
      std::stringstream ss;
      if (line.find_first_not_of(" \t\n\v\f\r") == std::string::npos)
      {
         ss << "image " << img_num << std::endl;
      	 features.push_back(std::vector<cv::Mat >());
         changeStructure(descriptors, features.back());
	 std::cout << "lines read: " << line_num - start << std::endl;
	 std::cout << "descriptors shape: " << descriptors.size() << std::endl;
	 img_num ++;
	 start = line_num;
	 descriptors = cv::Mat();
	 if (img_num % 300 == 0){
             std::cout << "img num " << img_num << std::endl;
	 }
         if (30000 == img_num) {
             break;
         }
	 continue;
      }

      std::stringstream check1(line);

      std::string current;

      cv::Mat row(1, 256, CV_32FC1);
      // Tokenizing w.r.t. space ' '
      unsigned short i = 0;
      while(getline(check1, current, ' '))
      {
	 float curr_float = ::atof(current.c_str());
	 row.at<float>(0, i) = curr_float;
	 assert(row.at<float>(0, i) == curr_float);
	 i++ ;
	 assert(i < 300);
      }
      assert(i == 256);
      descriptors.push_back(row);
      line_num ++;

  }
  return features;
}

void changeStructure(const cv::Mat &plain, std::vector<cv::Mat > &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

void createVoc(const std::vector<std::vector<cv::Mat > > & features){
  // branching factor and depth levels from DBoW2 paper
  // should be good for now
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  ORB_SLAM3::GOODVocabulary voc(k, L, weight, scoring);

  std::cout << "Creating a goodpoint " << k << "^" << L << " vocabulary..." << std::endl;
  voc.create(features);
  std::cout << "... done!" <<std::endl;

  std::cout << "Vocabulary information: " << std::endl
      << voc << std::endl <<std::endl;

  // save the vocabulary to disk
  std::cout << std::endl << "Saving vocabulary..." << std::endl;
  voc.save("good_voc.yml.gz");
  std::cout << "Done" <<std::endl;
}

void testDatabase(const std::vector<std::vector<cv::Mat > > &features)
{

  std::cout << "Reading dbow from " << "good_voc.json" << std::endl;
  // load the vocabulary from disk
  ORB_SLAM3::GOODVocabulary voc;
  voc.load_json("good_voc.json");

  std::cout << "Creating a small database..." <<std::endl;
  ORB_SLAM3::GOODDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < 1000; i++)
  {
    db.add(features[i]);
  }

  std::cout << "... done!" <<std::endl;

  std::cout << "Database information: " <<std::endl << db <<std::endl;

  // and query the database
  std::cout << "Querying the database: " <<std::endl;

  QueryResults ret;
  for(int i = 0; i < 100; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    std::cout << "Searching for Image " << i << ". " << ret << std::endl;
  }

  std::cout << std::endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  std::cout << "Saving database..." << std::endl;
  db.save("good_db.yml.gz");
  std::cout << "... done!" << std::endl;

  // once saved, we can load it again
  std::cout << "Retrieving database once again..." <<std::endl;
  ORB_SLAM3::GOODDatabase db2("good_db.yml.gz");
  std::cout << "... done! This is: " << std::endl << db2 <<std::endl;
}


int main(int argc, char ** argv){
    feature_vector features = loadFeatures();
//    createVoc(features);
    testDatabase(features);
    return 0;
}
