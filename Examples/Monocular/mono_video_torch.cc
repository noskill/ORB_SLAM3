/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "System.h"
#include "goodvocabulary.h"
#include "Converter.h"

#if (CV_MAJOR_VERSION > 3)
#include <opencv2/imgproc.hpp>
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#endif

using namespace std;

double ttrack_tot = 0;
int main(int argc, char **argv)
{

    if(argc < 3)
    {
        cerr << endl << "Usage: ./mono_video path_to_vocabulary path_to_settings path_to_video" << endl;
        return 1;
    }

    string file_name;
    file_name = string(argv[argc-1]);
    cout << "file name: " << file_name << endl;

    cv::VideoCapture cap(file_name, cv::CAP_ANY);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System<ORB_SLAM3::GOODVocabulary> SLAM(argv[1], argv[2], ORB_SLAM3::SystemBase::MONOCULAR,true);

    // Main loop
    cv::Mat im, frame;
    std::chrono::milliseconds ms{33}; // interval between frames

    // histogram normalization
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    std::chrono::system_clock::time_point prev = start;
    for(;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        bool ok = cap.read(frame);
        if (not ok) break;
        cv::cvtColor(frame, im, CV_BGR2GRAY);

        // clahe
        clahe->apply(im, im);

        auto t1 = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(prev - start);

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, std::chrono::duration<double>(timestamp).count() / 1000
                            ); // change to monocular_inertial if needed

        auto t2 = std::chrono::system_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        double to_sleep = std::chrono::duration<double>(ms).count() -
                std::chrono::duration<double>(duration).count();
        // Wait to load the next frame if SLAM works too fast
        if(0 < to_sleep)
            usleep(to_sleep * 1000); // microseconds
        prev += ms;
    }

    // cout << "ttrack_tot = " << ttrack_tot << std::endl;
    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
    const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
    SLAM.SaveTrajectoryEuRoC(f_file);
    SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);

    return 0;
}

