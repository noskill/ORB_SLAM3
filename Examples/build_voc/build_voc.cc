#include <ostream>
#include <iostream>
#include "DBoW2/FGOOD.h"

using namespace DBoW2;

vector<vector<cv::Mat > > loadFeatures();
void createVoc(const vector<vector<cv::Mat > > & features);
void testDatabase(const vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);


vector<vector<cv::Mat > > loadFeatures()
{
  vector<vector<cv::Mat > > features;
  features.clear();
  features.reserve(10000);

  std::string delimiter = " ";
  cout << "Extracting ORB features..." << endl;
  long i = 0;
  cv::Mat descriptors;
  for (std::string line; std::getline(std::cin, line);) {
      stringstream ss;
      ss << "image " << i << std::endl;
      if (str.find_first_not_of(" \t\n\v\f\r") != std::string::npos)
      {
         // There's a non-space.
      	 features.push_back(vector<cv::Mat >());
         changeStructure(descriptors, features.back());
	 descriptors = cv::Mat();
      }

      stringstream check1(line);

      string current;

      cv::Mat row(1, 256, CV_32FC);
      // Tokenizing w.r.t. space ' '
      unsigned short i = 0;
      while(getline(check1, current, ' '))
      {
	 float curr_float = ::atof(current);
	 row.at(0, i) = curr_float;
	 i++ ;
      }
      assert(i == 255);
      descriptors.push_back(row);
  }
}



  }
}

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

void createVoc(const vector<vector<cv::Mat > > & features){
  // branching factor and depth levels from DBoW2 paper
  // should be good for now
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  OrbVocabulary voc(k, L, weight, scoring);

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

void testDatabase(const vector<vector<cv::Mat > > &features)
{
  std::cout << "Creating a small database..." <<std::endl;

  // load the vocabulary from disk
  OrbVocabulary voc("good_voc.yml.gz");

  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  std::cout << "... done!" <<std::endl;

  std::cout << "Database information: " <<std::endl << db <<std::endl;

  // and query the database
  std::cout << "Querying the database: " <<std::endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
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
  OrbDatabase db2("good_db.yml.gz");
  std::cout << "... done! This is: " << std::endl << db2 <<std::endl;
}


int main(int argc, char ** argv){
    return 0;
}
