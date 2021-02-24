#ifndef SYSTEMBASE_H
#define SYSTEMBASE_H

#include <string>
#include <vector>
#include <iostream>
#include <thread>

#include<opencv2/core/core.hpp>

#include "ImuTypes.h"
#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "Viewer.h"


namespace ORB_SLAM3
{


class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_QUIET=0,
        VERBOSITY_NORMAL=1,
        VERBOSITY_VERBOSE=2,
        VERBOSITY_VERY_VERBOSE=3,
        VERBOSITY_DEBUG=4
    };

    static eLevel th;

public:
    static void PrintMess(std::string str, eLevel lev)
    {
        if(lev <= th)
            std::cout << str << std::endl;
    }

    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

class Viewer;
class FrameDrawer;
class Atlas;
class Tracking;
class LocalMapping;
class LoopClosing;


class SystemBase {
    public:
        // Input sensor
        enum eSensor{
            MONOCULAR=0,
            STEREO=1,
            RGBD=2,
            IMU_MONOCULAR=3,
            IMU_STEREO=4
        };

        // File type
        enum eFileType{
            TEXT_FILE=0,
            BINARY_FILE=1,
        };

    public:

    SystemBase(const eSensor sensor, const bool bUseViewer=true);

        // Proccess the given stereo frame. Images must be synchronized and rectified.
        // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
        // Returns the camera pose (empty if tracking fails).
        cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const std::vector<IMU::Point>& vImuMeas = std::vector<IMU::Point>(), std::string filename="");

        // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
        // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
        // Input depthmap: Float (CV_32F).
        // Returns the camera pose (empty if tracking fails).
        cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, std::string filename="");

        // Proccess the given monocular frame and optionally imu data
        // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
        // Returns the camera pose (empty if tracking fails).
        cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp, const std::vector<IMU::Point>& vImuMeas = std::vector<IMU::Point>(), std::string filename="");


        // This stops local mapping thread (map building) and performs only camera tracking.
        void ActivateLocalizationMode();
        // This resumes local mapping thread and performs SLAM again.
        void DeactivateLocalizationMode();

        // Returns true if there have been a big map change (loop closure, global BA)
        // since last call to this function
        bool MapChanged();

        // Reset the system (clear Atlas or the active map)
        void Reset();
        void ResetActiveMap();

        // All threads will be requested to finish.
        // It waits until all threads have finished.
        // This function must be called before saving the trajectory.
        void Shutdown();

        // Save camera trajectory in the TUM RGB-D dataset format.
        // Only for stereo and RGB-D. This method does not work for monocular.
        // Call first Shutdown()
        // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
        void SaveTrajectoryTUM(const std::string &filename);

        // Save keyframe poses in the TUM RGB-D dataset format.
        // This method works for all sensor input.
        // Call first Shutdown()
        // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
        void SaveKeyFrameTrajectoryTUM(const std::string &filename);

        void SaveTrajectoryEuRoC(const std::string &filename);
        void SaveKeyFrameTrajectoryEuRoC(const std::string &filename);

        // Save data used for initialization debug
        void SaveDebugData(const int &iniIdx);

        // Save camera trajectory in the KITTI dataset format.
        // Only for stereo and RGB-D. This method does not work for monocular.
        // Call first Shutdown()
        // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
        void SaveTrajectoryKITTI(const std::string &filename);

        // TODO: Save/Load functions
        // SaveMap(const std::string &filename);
        // LoadMap(const std::string &filename);

        // Information from most recent processed frame
        // You can call this right after TrackMonocular (or stereo or RGBD)
        int GetTrackingState();
        std::vector<MapPoint*> GetTrackedMapPoints();
        std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

        // For debugging
        double GetTimeFromIMUInit();
        bool isLost();
        bool isFinished();

        void ChangeDataset();

        //void SaveAtlas(int type);
protected:

    eSensor mSensor;


    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer* mpViewer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;
    bool mbResetActiveMap;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // Tracking state
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;
    Atlas* mpAtlas;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;

    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;


};


}

#endif // SYSTEM_H
