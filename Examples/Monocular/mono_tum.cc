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


#include<iostream>  //用于输入和输出操作
#include<algorithm> //包含了一组通用算法的定义和实现，这些算法可以对容器（如数组、std::vector等）进行操作
#include<fstream>   //用于文件输入和输出操作
#include<chrono>    //用于处理时间点、时间间隔和时钟

#include<opencv2/core/core.hpp> //用于图像处理和计算机视觉应用

#include<System.h>  //它通常包含了主要的系统类或与 SLAM 系统相关的定义。

using namespace std; //它用于指示编译器使用标准库（std 命名空间）中的所有符号，而不需要显式地在每个标准库的成员前加上 std:: 前缀。

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
int main(int argc, char **argv)
{
    if(argc != 4)//需要保证命令参数个数为4
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";

    LoadImages(strFile, vstrImageFilenames, vTimestamps);//成功加载文件名和时间戳

    int nImages = vstrImageFilenames.size();             //返回 vstrImageFilenames 中的字符串元素的数量。

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // ORB_SLAM2::System代表命名空间ORB_SLAM2下的System类
    // System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true);
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

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
