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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}
//!线程主函数
void LocalMapping::Run()
{
    //!标记状态，表示当前run函数正在运行，尚未结束
    mbFinished = false;
    //!主循环
    while(1)
    {
        // Tracking will see that Local Mapping is busy
        //!告诉Tracking，LocalMapping正处于繁忙状态，请不要给我发送关键帧打扰我
        //!LocalMapping线程处理的关键帧都是Tracking线程发来的
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        //!等待处理的关键帧列表不为空
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            //!处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
            ProcessNewKeyFrame();

            // Check recent MapPoints
            //!根据地图点的观测情况剔除质量不好的地图点
            MapPointCulling();

            // Triangulate new MapPoints
            //!当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
            CreateNewMapPoints();
            //!已经处理完队列中的最后的一个关键帧
            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                //!检查并融合当前关键帧与相邻关键帧帧（两级相邻）中重复的地图点
                SearchInNeighbors();
            }
            //!终止BA的标志
            mbAbortBA = false;
            //!已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                //!当局部地图中的关键帧大于2个的时候进行局部地图的BA 暂时未看
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
                //!检测并剔除当前帧相邻的关键帧中冗余的关键帧
                //!冗余的判定：该关键帧的90%的地图点可以被其它关键帧观测到
                KeyFrameCulling();
            }
            //!将当前帧加入到闭环检测队列中
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())//!当要终止当前线程的时候
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                // 如果还没有结束利索,那么等
                // usleep(3000);
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
            //!然后确定终止了就跳出这个线程的主循环
            if(CheckFinish())
                break;
        }
        //!查看是否有复位线程的请求
        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        //!设置"允许接受关键帧"的状态标志
        SetAcceptKeyFrames(true);
        //!如果当前线程已经结束了就跳出主循环
        if(CheckFinish())
            break;

        //usleep(3000);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    //!设置线程已经终止
    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    //!Tracking线程向LocalMapping中插入关键帧是先插入到该队列中
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}

//!查看列表中是否有等待被插入的关键帧
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    //!从缓冲队列中取出一帧关键帧
    //!该关键帧队列是Tracking线程向LocalMapping中插入的关键帧组成
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    //!计算该关键帧特征点的词袋向量
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    //!当前处理关键帧中有效的地图点，更新normal，描述子等信息
    //!TrackLocalMap中和当前帧新匹配上的地图点和当前关键帧进行关联绑定
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    //!对当前处理的这个关键帧中的所有的地图点展开遍历
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                //!检查该地图点是否在关键帧中（有对应的二维特征点）
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    //!如果地图点不是来自当前帧的观测（比如来自局部地图点），为当前地图点添加观测
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    //!获得该点的平均观测方向和观测距离范围
                    pMP->UpdateNormalAndDepth();
                    //!更新地图点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    //!这些地图点可能来自双目或RGBD在创建关键帧中新生成的地图点，或者是CreateNewMapPoints 中通过三角化产生
                    //!将上述地图点放入mlpRecentAddedMapPoints，等待后续MapPointCulling函数的检验
                    //!虽然该地图点已经有对应的二维特征点，但是可能还未加观测
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    //!更新关键帧间的连接关系（共视图）
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    //!将该关键帧插入到地图中
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    //!存储当前关键帧生成的地图点,也是等待检查的地图点列表
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    //!获取当前关键帧的id
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    //!根据相机类型设置不同的观测阈值
    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;
    //!遍历检查新添加的地图点
    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            //!已经是坏点的地图点仅从队列中删除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            //!跟踪到该地图点的帧数相比预计可观测到该地图点的帧数的比例小于25%，从地图中删除
            //!(mnFound/mnVisible） < 25%
            //!mnFound ：地图点被多少帧（包括普通帧）看到，次数越多越好
            //!mnVisible：地图点应该被看到的次数
            //!(mnFound/mnVisible）：对于大FOV镜头这个比例会高，对于窄FOV镜头这个比例会低
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            //!从该点建立开始，到现在已经过了不小于2个关键帧
            //!但是观测到该点的相机数却不超过阈值cnThObs，从地图中删除
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            //!从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
            //!因此没有SetBadFlag()，仅从队列中删除
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    //!nn表示搜索最佳共视关键帧的数目
    //!不同传感器下要求不一样,单目的时候需要有更多的具有较好共视关系的关键帧来建立地图
    int nn = 10;
    if(mbMonocular)
        nn=20;
    //!在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    //!特征点匹配配置 最佳距离 < 0.6*次佳距离，比较苛刻了。不检查旋转
    ORBmatcher matcher(0.6,false);
    //!取出当前帧从世界坐标系到相机坐标系的变换矩阵
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    //!得到当前关键帧（左目）光心在世界坐标系中的坐标、内参
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;
    //!用于后面的点深度的验证;这里的1.5是经验值
    //!mfScaleFactor = 1.2
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
    //!记录与所有相邻关键帧的三角化成功的地图点数目
    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    //!遍历相邻关键帧，搜索匹配并用极线约束剔除误匹配，最终三角化
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        // ? 疑似bug，正确应该是 if(i>0 && !CheckNewKeyFrames())
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        //!相邻的关键帧光心在世界坐标系中的坐标
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        //!基线向量，两个关键帧间的相机位移
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);
        //!判断相机运动的基线是不是足够长
        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            //!单目相机情况
            //!相邻关键帧的场景深度中值(其实过程就是对当前关键帧下所有地图点的深度进行从小到大排序,返回距离头部其中1/q处的深度值作为当前场景的平均深度)
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            //!基线与景深的比例
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            //!如果比例特别小，基线太短恢复3D点不准，那么跳过当前邻接的关键帧，不生成3D点
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        //!根据两个关键帧的位姿计算它们之间的基础矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        //!通过词袋对两关键帧的未匹配的特征点快速匹配，用极线约束抑制离群点，生成新的匹配点对
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        //!对每对匹配通过三角化生成3D点,和 Triangulate函数差不多
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            //!当前匹配对在当前关键帧中的索引
            const int &idx1 = vMatchedIndices[ikp].first;
            //!当前匹配对在邻接关键帧中的索引
            const int &idx2 = vMatchedIndices[ikp].second;
            //!当前匹配在当前关键帧中的特征点
            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            //!mvuRight中存放着双目的深度值，如果不是双目，其值将为-1
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            //!bStereo1 : false
            bool bStereo1 = kp1_ur>=0;
            //!当前匹配在邻接关键帧中的特征点
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            //!mvuRight中存放着双目的深度值，如果不是双目，其值将为-1
            const float kp2_ur = pKF2->mvuRight[idx2];
            //!bStereo2 : false
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            //!利用匹配点反投影得到视差角
            //!两个关键点（kp1和kp2）的归一化坐标
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);
            //!由相机坐标系转到世界坐标系(得到的是那条反投影射线的一个同向向量在世界坐标系下的表示,还是只能够表示方向)，得到视差角余弦值
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            //!匹配点射线夹角余弦值
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));
            //!加1是为了让cosParallaxStereo随便初始化为一个很大的值
            float cosParallaxStereo = cosParallaxRays+1;
            //!cosParallaxStereo1、cosParallaxStereo2在后面可能不存在，需要初始化为较大的值
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            //!对于双目，利用双目得到视差角；单目相机没有特殊操作
            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);
            
            //!三角化恢复3D点
            //!cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)表明视差角正常,0.9998 对应1°
            //!cosParallaxRays < cosParallaxStereo 表明匹配点对夹角大于双目本身观察三维点夹角
            //!匹配点对夹角大，用三角法恢复3D点
            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                //!见Initializer.cc的 Triangulate 函数,实现是一样的,顶多就是把投影矩阵换成了变换矩阵
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();
                //!归一化之前的检查
                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                //!归一化成为齐次坐标,然后提取前面三个维度作为欧式坐标
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax
            //!为方便后续计算，转换成为了行向量
            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            //!检测生成的3D点是否在相机前方,不在的话就放弃这个点
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            //!计算3D点在当前关键帧下的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                //!单目情况下
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                //!假设测量有一个像素的偏差，2自由度卡方检验阈值是5.991
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            //!计算3D点在另一个关键帧下的重投影误差，操作同上
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                //!单目情况下
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                //!假设测量有一个像素的偏差，2自由度卡方检验阈值是5.991
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            //!检查尺度连续性
            //!世界坐标系下，3D点与相机间的向量，方向由相机指向3D点
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;
            //!ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioDist = dist2/dist1;
            //!金字塔尺度因子的比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            //!距离的比例和图像金字塔的比例不应该差太多，否则就跳过
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            //!三角化生成3D点成功，构造成MapPoint
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            //!为该MapPoint添加属性：
            //!观测到该MapPoint的关键帧
            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);
            //!该MapPoint的最具有代表性的描述子
            pMP->ComputeDistinctiveDescriptors();
            //!该MapPoint的平均观测方向和深度范围
            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            //!将新产生的点放入检测队列
            //!这些MapPoints都会经过MapPointCulling函数的检验
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    //!获得当前关键帧在共视图中权重排名前nn的邻接关键帧
    //!开始之前先定义几个概念
    //!当前关键帧的邻接关键帧，称为一级相邻关键帧，也就是邻居
    //!与一级相邻关键帧相邻的关键帧，称为二级相邻关键帧，也就是邻居的邻居

    //!单目情况要20个邻接关键帧，双目或者RGBD则要10个
    int nn = 10;
    if(mbMonocular)
        nn=20;
    //!和当前关键帧相邻的关键帧，也就是一级相邻关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    //!存储一级相邻关键帧及其二级相邻关键帧
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        //!没有和当前帧进行过融合的操作
        //!mnFuseTargetForKF  标记在局部建图线程中,和哪个关键帧进行融合的操作
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        //!加入一级相邻关键帧 
        vpTargetKFs.push_back(pKFi);
        //!标记已经加入
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        //!以一级相邻关键帧的共视关系最好的5个相邻关键帧 作为二级相邻关键帧
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        //!遍历得到的二级相邻关键帧
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            //!当然这个二级相邻关键帧要求没有和当前关键帧发生融合,并且这个二级相邻关键帧也不是当前关键帧
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            //!存入二级相邻关键帧
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    //!使用默认参数, 最优和次优比例0.6,匹配时检查特征点的旋转
    ORBmatcher matcher;
    //!将当前帧的地图点分别投影到两级相邻关键帧，寻找匹配点对应的地图点进行融合，称为正向投影融合
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        //!将地图点投影到关键帧中进行匹配和融合；融合策略如下
        //!1.如果地图点能匹配关键帧的特征点，并且该点有对应的地图点，那么选择观测数目多的替换两个地图点
        //!2.如果地图点能匹配关键帧的特征点，并且该点没有对应的地图点，那么为该点添加该投影地图点
        //!注意这个时候对地图点融合的操作是立即生效的
        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    //!将两级相邻关键帧地图点分别投影到当前关键帧，寻找匹配点对应的地图点进行融合，称为反向投影融合
    //!用于进行存储要融合的一级邻接和二级邻接关键帧所有MapPoints的集合
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            //!如果地图点是坏点，或者已经加进集合vpFuseCandidates，跳过
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }
    //!进行地图点投影融合,和正向融合操作是完全相同的
    //!不同的是正向操作是"每个关键帧和当前关键帧的地图点进行融合",而这里的是"当前关键帧和所有邻接关键帧的地图点进行融合"
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);

    // Update points
    //!更新当前帧地图点的描述子、深度、平均观测方向等属性
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                //!在所有找到pMP的关键帧中，获得最佳的描述子
                pMP->ComputeDistinctiveDescriptors();
                //!更新平均观测方向和观测距离
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    //!更新当前帧与其它帧的共视连接关系
    mpCurrentKeyFrame->UpdateConnections();
}

//!根据两关键帧的姿态计算两个关键帧之间的基本矩阵
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    //!先构造两帧之间的R12,t12
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
    //!得到 t12 的反对称矩阵
    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    //!Essential Matrix: t12叉乘R12
    //!Fundamental Matrix: inv(K1)*E*inv(K2)
    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points

    //!该函数里变量层层深入，这里列一下：
    //!mpCurrentKeyFrame：当前关键帧，本程序就是判断它是否需要删除
    //!pKF： mpCurrentKeyFrame的某一个共视关键帧
    //!vpMapPoints：pKF对应的所有地图点
    //!pMP：vpMapPoints中的某个地图点
    //!observations：所有能观测到pMP的关键帧
    //!pKFi：observations中的某个关键帧
    //!scaleLeveli：pKFi的金字塔尺度
    //!scaleLevel：pKF的金字塔尺度

    //!根据共视图提取当前关键帧的所有共视关键帧
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
    //!对所有的共视关键帧进行遍历
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        //!第1个关键帧不能删除，跳过
        if(pKF->mnId==0)
            continue;
        //!提取每个共视关键帧的地图点
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        //!记录某个点被观测次数，后面并未使用
        int nObs = 3;
        //!观测次数阈值，默认为3
        const int thObs=nObs;
        //!记录冗余观测点的数目
        int nRedundantObservations=0;
        int nMPs=0;
        //!遍历该共视关键帧的所有地图点，其中能被其它至少3个关键帧观测到的地图点为冗余地图点
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    //!pMP->Observations() 是观测到该地图点的相机总数目（单目1，双目2）
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        //!Observation存储的是可以看到该地图点的所有关键帧的集合
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        //!遍历观测到该地图点的关键帧
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            //!尺度约束：为什么pKF 尺度+1 要大于等于 pKFi 尺度？
                            //!回答：因为同样或更低金字塔层级的地图点更准确
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                //!已经找到3个满足条件的关键帧，就停止不找了
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        //!地图点至少被3个关键帧观测到，就记录为冗余点，更新冗余点计数数目
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  
        //!如果该关键帧90%以上的有效地图点被判断为冗余的，则认为该关键帧是冗余的，需要删除该关键帧
        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    //!执行复位操作:清空关键帧缓冲区,清空待cull的地图点缓冲
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        //!恢复为false表示复位过程完成
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    //!线程已经被结束
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;  //!既然已经都结束了,那么当前线程也已经停止工作了
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
