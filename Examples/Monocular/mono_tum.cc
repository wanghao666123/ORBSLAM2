/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>  
#include<algorithm> 
#include<fstream>   
#include<chrono>    

#include<opencv2/core/core.hpp> 

#include<System.h>  

using namespace std; 

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

/**
 * @brief 单目摄像头的主函数
 * 
 * @param argc 主函数的入口命令参数（const string类型）。
 * @param argv 主函数的每个命令行参数的具体值（char **类型）。
 * @return 无。
 * 
 * @note ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml data/rgbd_dataset_freiburg1_xyz
 * ./Examples/Monocular/mono_tum：可执行文件
 * Vocabulary/ORBvoc.txt：用来指定程序所需的词汇表文件的路径或者文件名。
 * Examples/Monocular/TUM1.yaml：用来指定程序的设置文件的路径或者文件名。
 * data/rgbd_dataset_freiburg1_xyz：指定程序需要处理的数据序列的路径。
 */
//! orbslam2 单目的主函数，也就是程序入口
int main(int argc, char **argv)
{
    //! 判断入口参数数目
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";

    //! 加载文件名和时间戳
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();            

    //! 初始化整个slam系统
    //! 1. 根据第三方库建立一个新的ORB字典，生成树
    //! 2. 根据预训练好的字典大小设置关键帧数据库，位置后的重定位和回环检测做准备
    //! 3. 创建地图信息
    //! 4. 创建帧绘制器和地图绘制器将会被可视化的Viewer所使用 先不管
    //! 5. 创建tracking线程
    //! 6. 创建mpLocalMapper线程(待定)
    //! 7. 创建mpLoopCloser线程(待定)
    //! 8. 创建mptViewer线程
    //! 9. 设置进程间的指针
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;
    //! 主循环，主要是进行追踪线程
    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        //! 重要，追踪线程
        //! 1. 检查是否开启纯定位或停用纯定位模式
        //! 2. 检查是否复位
        //! 3. 开始进行真正意义上的追踪线程
        //! 4. 更新追踪状态，追踪到每帧图像的特征点对应的地图点，已经经过畸变矫正的特征点集合
        SLAM.TrackMonocular(im,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    //! 计算追踪总时间和平均时间
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    //! 保存轨迹信息
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

/**
 * @brief 从文件中加载时间戳和图像路径。
 * 
 * @param strFile 包含图像的文件路径（const string类型）。
 * @param vstrImageFilenames 存储图像文件名的向量（vector<string>类型）。
 * @param vTimestamps 存储图像时间戳的向量（vector<double>类型）。
 * @return 无。
 * 
 * @note 函数声明。
 */
void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;//ifstream 是 C++ 中用于读取文件的输入文件流对象。
    f.open(strFile.c_str());//打开文件data/rgbd_dataset_freiburg1_xyz/rgb.txt

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())//循环读取文件直到文件末尾
    {   
        string s;
        getline(f,s);//读取文件的每一行到字符串 s 中。
        if(!s.empty())//如果字符串 s 不为空，表示读取成功
        {
            stringstream ss;            //使用 stringstream 对象 ss 解析字符串 s。
            ss << s;                    // 将读取到的一行字符串插入到 stringstream 中
            double t;
            string sRGB;
            ss >> t;                    // 从 stringstream 中提取时间戳并存储到 t 中
            vTimestamps.push_back(t);   // 将时间戳存入向量 vTimestamps
            ss >> sRGB;                 // 从 stringstream 中提取图像文件名并存储到 t 中
            vstrImageFilenames.push_back(sRGB);// 将图像文件名存入向量 vstrImageFilenames
        }
    }
}
