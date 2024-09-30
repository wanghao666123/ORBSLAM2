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
* You s1hould have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    //!从参考帧中获取相机的内参数矩阵
    mK = ReferenceFrame.mK.clone();
    //!从参考帧中获取去畸变后的特征点
    mvKeys1 = ReferenceFrame.mvKeysUn;
    //!sigma即为标准差，通常用于描述图像特征点检测的精度或误差模型，标准差的定义是反映数据点相对于均值的偏离程度。
    //!较小的 sigma：意味着检测精度高，误差小，特征点位置非常接近真实值。
    //!较大的 sigma：表示较大的不确定性或测量误差，这可能是因为图像质量差、光照变化大、特征点附近纹理不够明显等原因
    mSigma = sigma;//!1.0
    mSigma2 = sigma*sigma;//!平方
    //!最大迭代次数，RANSAC（随机抽样一致性算法）200
    mMaxIterations = iterations;
}

bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    //!获取当前帧的去畸变之后的特征点
    mvKeys2 = CurrentFrame.mvKeysUn;
    //!mvMatches12记录匹配上的特征点对，记录的是帧2在帧1的匹配索引
    mvMatches12.clear();
    //!预分配空间，大小和关键点数目一致mvKeys2.size()
    mvMatches12.reserve(mvKeys2.size());
    //!记录参考帧1中的每个特征点是否有匹配的特征点
    //!这个成员变量后面没有用到，后面只关心匹配上的特征点 	
    mvbMatched1.resize(mvKeys1.size());
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        //!vMatches12[i]解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值
        //!没有匹配关系的话，vMatches12[i]值为 -1
        if(vMatches12[i]>=0)
        {
            //!mvMatches12 中只记录有匹配关系的特征点对的索引值
            //!i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }
    //!有匹配的特征点的对数
    const int N = mvMatches12.size();

    // Indices for minimum set selection
    //!新建一个容器vAllIndices存储特征点索引，并预分配空间
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);

    //!在RANSAC的某次迭代中，还可以被抽取来作为数据样本的特征点对的索引，所以这里起的名字叫做可用的索引
    vector<size_t> vAvailableIndices;
    //!初始化所有特征点对的索引，索引值0到N-1
    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    //!mMaxIterations = 200
    //!mMaxIterations: 这是外层向量的大小，即 mvSets 中会包含 mMaxIterations 个向量。通常这个值表示 RANSAC 的最大迭代次数
    //!vector<size_t>(8, 0): 这是内层向量的定义。每个内层向量都是大小为 8 的 size_t 类型向量，并初始化为 0。在 RANSAC 的场景下，这可能用于存储一次迭代中的 8 个随机样本的索引。
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));
    //!用于进行随机数据样本采样，设置随机数种子
    DUtils::Random::SeedRandOnce(0);
    //!开始每一次的迭代
    //!迭代mMaxIterations次，选取各自迭代时需要用到的最小数据集
    for(int it=0; it<mMaxIterations; it++)
    {
        //!迭代开始的时候，先假设所有的点都是可用的
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        //!8点法 8对点 1对点有2个点(x1,y1)(x2,y2)
        for(size_t j=0; j<8; j++)
        {
            //!随机产生某一个对点的id（索引）,范围从0到N-1
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            //!idx代表选择的是第几个特征点对，即特征点对索引
            int idx = vAvailableIndices[randi];
            //!it表示第几次迭代，j表示在某一次迭代中选取的某一个点
            mvSets[it][j] = idx;
            //!由于这对点在本次迭代中已经被使用了,所以我们为了避免再次抽到这个点,就在"点的可选列表"中,
            //!将这个点原来所在的位置用vector最后一个元素的信息覆盖,并且删除尾部的元素
            //!这样就相当于将这个点的信息从"点的可用列表"中直接删除了
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }
    //!计算fundamental 矩阵 和homography 矩阵，为了加速分别开了线程计算
    // Launch threads to compute in parallel a fundamental matrix and a homography
    //!这两个变量用于标记在H和F的计算中哪些特征点对被认为是Inlier
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    //!计算出来的单应矩阵和基础矩阵的RANSAC评分，这里其实是采用重投影误差来计算的
    float SH, SF;
    //!这两个是经过RANSAC算法后计算出来的单应矩阵和基础矩阵
    cv::Mat H, F;
    //!构造线程来计算H矩阵及其得分
    //!如果相机在拍摄时只发生了旋转（没有平移），那么整个视角变化可以通过单应矩阵来描述
    //!ref 是 std::ref，它用于传递参数的引用。这样做是为了避免对这些参数进行复制，因为线程可能需要修改这些参数的值。
    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    //!当相机不仅有旋转还有平移时，基础矩阵可以描述这些点之间的约束关系
    //!计算fundamental matrix并打分，参数定义和H是一样的，这里不再赘述
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    //!等待两个计算线程结束
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    //!计算得分比例来判断选取哪个模型来求位姿R,t
    //!通过这个规则来判断谁的评分占比更多一些，注意不是简单的比较绝对评分大小，而是看评分的占比
    float RH = SH/(SH+SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    //!注意这里更倾向于用H矩阵恢复位姿。如果单应矩阵的评分占比达到了0.4以上,则从单应矩阵恢复运动,否则从基础矩阵恢复运动
    if(RH>0.40)
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6) 更偏向于非平面，从基础矩阵恢复
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}


void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    //!获取已经成功匹配之后的特征点对数目
    const int N = mvMatches12.size();

    // Normalize coordinates
    //!归一化步骤用于将特征点的原始坐标转换为零均值、单位标准差的坐标，这样可以避免在后续计算单应矩阵或基础矩阵时，由于数值的不同量级而引入的误差。
    //!归一化后的参考帧1和当前帧2中的特征点坐标
    vector<cv::Point2f> vPn1, vPn2;
    //!记录各自的归一化矩阵
    cv::Mat T1, T2;
    //!进行归一化
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    //!求解归一化矩阵的逆
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    //!记录最佳评分
    score = 0.0;
    //!取得历史最佳评分时,特征点对的inliers标记
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    //!某次迭代中，参考帧的特征点坐标
    vector<cv::Point2f> vPn1i(8);
    //!某次迭代中，当前帧的特征点坐标
    vector<cv::Point2f> vPn2i(8);
    //!以及计算出来的单应矩阵、及其逆矩阵
    cv::Mat H21i, H12i;
    //!每次RANSAC记录Inliers与得分
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    //!迭代200次 得到最佳的模型符合第一张图像到第二张图像的变化，即求出最佳的H
    //!每一次迭代，会随机的拿出8个已经匹配成功的特征点对，用于计算H
    //!然后根据基于当前8个匹配点对计算出来的H，遍历第一张和第二张图像的所有特征点对，对先前计算出来的H进行卡方校验（通过H12,H21分别计算针对第一张图像和第二张图像的重投影误差，符合阈值，则累加得分，并把遍历到的当前符合阈值的特征点对设置为内点，不符合的设置为外点）
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            //!从mvSets中获取当前次迭代的某个特征点对的索引信息
            int idx = mvSets[it][j];
            //!mvMatches12[idx].first：为已成功匹配特征点对的第一张图像中的特征点索引
            //!vPn1[mvMatches12[idx].first]：为第一张图像中已经经过归一化之后的特征点坐标
            //!mvMatches12[idx].second：为已成功匹配特征点对的第二张图像中的特征点索引
            //!vPn1[mvMatches12[idx].second]：为第二张图像中已经经过归一化之后的特征点坐标
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        //!Hn：为经过奇异值分解得到的单应矩阵的第9列的特征向量代表的最优解H（归一化坐标系下的H）
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        //!需要求得反归一化下的H
        //!单应矩阵原理：X2=H21*X1，其中X1,X2 为归一化后的特征点    
        //!特征点归一化：vPn1 = T1 * mvKeys1, vPn2 = T2 * mvKeys2  得到:T2 * mvKeys2 =  Hn * T1 * mvKeys1   
        //!进一步得到:mvKeys2  = T2.inv * Hn * T1 * mvKeys1
        H21i = T2inv*Hn*T1;
        //!然后计算逆
        H12i = H21i.inv();
        //!mSigma：1.0
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);
        //!如果当前迭代的得分大于上一次迭代的得分，则更新相关信息
        if(currentScore>score)
        {
            //!进行深拷贝
            H21 = H21i.clone();
            //!保存匹配好的特征点对的Inliers标记
            vbMatchesInliers = vbCurrentInliers;
            //!更新历史最优评分
            score = currentScore;
        }
    }
}


void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    //?源代码出错！请使用下面代替
    //?得到第一张图片和第二张图片中的已经成功匹配成功的特征点对
    //?const int N = mvMatches12.size();
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    //!归一化步骤用于将特征点的原始坐标转换为零均值、单位标准差的坐标，这样可以避免在后续计算单应矩阵或基础矩阵时，由于数值的不同量级而引入的误差。
    //!归一化后的参考帧1和当前帧2中的特征点坐标
    vector<cv::Point2f> vPn1, vPn2;
    //!记录各自的归一化矩阵
    cv::Mat T1, T2;
    //!进行归一化处理
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    //!求解归一化矩阵的转置
    cv::Mat T2t = T2.t();
    // Best Results variables
    //!记录最佳评分
    score = 0.0;
    //!取得历史最佳评分时,特征点对的inliers标记
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    //!某次迭代中，参考帧的特征点坐标
    vector<cv::Point2f> vPn1i(8);
    //!某次迭代中，当前帧的特征点坐标
    vector<cv::Point2f> vPn2i(8);
    //!以及计算出来的基础矩阵
    cv::Mat F21i;
    //!每次RANSAC记录Inliers与得分
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    //!迭代200次 得到最佳的模型符合第一张图像到第二张图像的变化，即求出最佳的F
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            //!从mvSets中获取当前次迭代的某个特征点对的索引信息
            int idx = mvSets[it][j];
            //!mvMatches12[idx].first：为已成功匹配特征点对的第一张图像中的特征点索引
            //!vPn1[mvMatches12[idx].first]：为第一张图像中已经经过归一化之后的特征点坐标
            //!mvMatches12[idx].second：为已成功匹配特征点对的第二张图像中的特征点索引
            //!vPn1[mvMatches12[idx].second]：为第二张图像中已经经过归一化之后的特征点坐标
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        //!八点法计算基础矩阵F
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);
        //!需要求得反归一化下的F
        //!基础矩阵约束：p2^t*F21*p1 = 0，其中p1,p2 为齐次化特征点坐标    
        //!特征点归一化：vPn1 = T1 * mvKeys1, vPn2 = T2 * mvKeys2  
        //!根据基础矩阵约束得到:(T2 * mvKeys2)^t* Fn * T1 * mvKeys1 = 0   
        //!进一步得到:mvKeys2^t * T2^t * Fn * T1 * mvKeys1 = 0
        F21i = T2t*Fn*T1;
        //!mSigma：1.0
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);
        //!如果当前迭代的得分大于上一次迭代的得分，则更新相关信息
        if(currentScore>score)
        {
            //!进行深拷贝
            F21 = F21i.clone();
            //!保存匹配好的特征点对的Inliers标记
            vbMatchesInliers = vbCurrentInliers;
            //!更新历史最优评分
            score = currentScore;
        }
    }
}


cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    //!得到第一张图像中的归一化之后的特征点数目，应该是8，一组为8，8点法
    const int N = vP1.size();
    //!8点法，16个方程，N=8,H矩阵为3*3的，9维矩阵
    cv::Mat A(2*N,9,CV_32F);
    //!看附件 3.1多视图几何基础
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }
    //!定义输出变量
    //!u：矩阵U 奇异值分解的左边
    //!w：奇异值矩阵 奇异值分解的中间
    //!vt：矩阵V^T 奇异值分解的右边
    cv::Mat u,w,vt;
    //!MODIFY_A是指允许计算函数可以修改待分解的矩阵，官方文档上说这样可以加快计算速度、节省内存
    //!FULL_UV=把U和VT补充成单位正交方阵
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    //!从矩阵 vt 中提取第 9 个奇异向量，并将其转换为 3×3 的矩阵形式
    //!返回最小奇异值所对应的右奇异向量
    //!注意前面说的是右奇异值矩阵的最后一列，但是在这里因为是vt，转置后了，所以是行；由于A有9列数据，故最后一列的下标为8
    return vt.row(8).reshape(0, 3);
}

cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    //!得到第一张图像中的归一化之后的特征点数目，应该是8，一组为8，8点法
    const int N = vP1.size();
    //!8点法，16个方程，N=8,H矩阵为3*3的，9维矩阵
    cv::Mat A(N,9,CV_32F);
    //!看附件 3.1多视图几何基础
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }
    //!定义输出变量
    //!u：矩阵U 奇异值分解的左边
    //!w：奇异值矩阵 奇异值分解的中间
    //!vt：矩阵V^T 奇异值分解的右边
    cv::Mat u,w,vt;
    //!MODIFY_A是指允许计算函数可以修改待分解的矩阵，官方文档上说这样可以加快计算速度、节省内存
    //!FULL_UV=把U和VT补充成单位正交方阵
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    //!从矩阵 vt 中提取第 9 个奇异向量，并将其转换为 3×3 的矩阵形式
    //!返回最小奇异值所对应的右奇异向量
    //!注意前面说的是右奇异值矩阵的最后一列，但是在这里因为是vt，转置后了，所以是行；由于A有9列数据，故最后一列的下标为8
    cv::Mat Fpre = vt.row(8).reshape(0, 3);
    //!基础矩阵的秩为2,而我们不敢保证计算得到的这个结果的秩为2,所以需要通过第二次奇异值分解,来强制使其秩为2
    //!对初步得来的基础矩阵进行第2次奇异值分解
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    //!秩2约束，强制将第3个奇异值设置为0
    w.at<float>(2)=0;
    //!重新组合好满足秩约束的基础矩阵，作为最终计算结果返回 
    return  u*cv::Mat::diag(w)*vt;
}

float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    //!得到匹配成功的特征点对数
    const int N = mvMatches12.size();
    //!将输入的两个 3×3 单应矩阵 H21 和 H12 的元素提取出来，以便后续使用。
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    //!给特征点对的Inliers标记预分配空间
    vbMatchesInliers.resize(N);
    //!初始化score值
    float score = 0;
    //!基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
	//!自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
    const float th = 5.991;
    //!信息矩阵，方差平方的倒数
    const float invSigmaSquare = 1.0/(sigma*sigma);
    //!遍历已经成功匹配的特征点对数
    for(int i=0; i<N; i++)
    {
        bool bIn = true;
        //!提取出第一张图像和第二张图像已经匹配成功的特征点坐标
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2
        //!计算 img2 到 img1 的重投影误差
        //!将图像2中的特征点通过单应变换投影到图像1中
        //!|u1|   |h11inv h12inv h13inv||u2|   |u2in1|
        //!|v1| = |h21inv h22inv h23inv||v2| = |v2in1| * w2in1inv
        //!|1 |   |h31inv h32inv h33inv||1 |   |  1  |
        //!归一化，同时除以w2in1inv
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;
        //!计算重投影误差 = ||p1(i) - H12 * p2(i)||2
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        const float chiSquare1 = squareDist1*invSigmaSquare;
        //!用阈值标记离群点，内点的话累加得分
        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1
        //!计算从img1 到 img2 的投影变换误差
        //!x1in2 = H21*x1
        //!将图像2中的特征点通过单应变换投影到图像1中
        //!|u2|   |h11 h12 h13||u1|   |u1in2|
        //!|v2| = |h21 h22 h23||v1| = |v1in2| * w1in2inv
        //!|1 |   |h31 h32 h33||1 |   |  1  |
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;
        //!用阈值标记离群点，内点的话累加得分
        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;
        //!如果满足阈值，则将对应遍历到的特征点对设置为true，反之为false
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    //!得到匹配成功的特征点对数
    const int N = mvMatches12.size();
    //!将输入的 3×3 基础矩阵 H21的元素提取出来，以便后续使用。
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);
    //!给特征点对的Inliers标记预分配空间
    vbMatchesInliers.resize(N);
    //!初始化score值
    float score = 0;
    //!基于卡方检验计算出的阈值
	//!自由度为1的卡方分布，显著性水平为0.05，对应的临界阈值
    //?是因为点到直线距离是一个自由度吗？
    const float th = 3.841;
    //!自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
    const float thScore = 5.991;
    //!信息矩阵，方差平方的倒数
    const float invSigmaSquare = 1.0/(sigma*sigma);
    //!遍历已经成功匹配的特征点对数
    for(int i=0; i<N; i++)
    {
        bool bIn = true;
        //!提取出第一张图像和第二张图像已经匹配成功的特征点坐标
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        //!计算 img1 上的点在 img2 上投影得到的极线 l2 = F21 * p1 = (a2,b2,c2)
        //!基础矩阵 FF 与点 (u1,v1,1)(u1​,v1​,1) 相乘后，会得到三个值 a2,b2,c2​，它们构成了极线的线性方程：a2​u2​+b2​v2​+c2​=0,其中 (u2,v2)(u2​,v2​) 是图像2中的点。这个方程描述了图像1中的点 (u1,v1)(u1​,v1​) 在图像2中所对应的极线。
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;
        //!计算第二张图片上对应的那个特征点到极线的距离的平方
        //!点 (u2,v2) 到极线的欧氏距离的平方。
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2);
        //!带权重误差
        const float chiSquare1 = squareDist1*invSigmaSquare;
        //!用阈值标记离群点，内点的话累加得分
        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        //!计算 img2 上的点在 img1 上投影得到的极线 l1 = F12 * p2 = (a1,b1,c1)
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;
        //!计算第一张图片上对应的那个特征点到极线的距离的平方
        //!点 (u1,v1) 到极线的欧氏距离的平方
        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        //!带权重误差
        const float chiSquare2 = squareDist2*invSigmaSquare;
        //!用阈值标记离群点，内点的话累加得分
        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;
        //!如果满足阈值，则将对应遍历到的特征点对设置为true，反之为false
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    //!统计有效匹配点个数，并用 N 表示
    //!vbMatchesInliers 中存储匹配点对是否是有效
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    //!基础矩阵 F 是一个 3×3 的矩阵，用于描述两个图像之间的点对应关系。它将一个图像上的点映射到另一个图像的极线。
    //!本质矩阵 E 是一个 3×3 的矩阵，用于描述两个相机之间的相对姿态和尺度信息。
    cv::Mat E21 = K.t()*F21*K;
    //!定义本质矩阵分解结果，形成四组解,分别是：
    //!(R1, t) (R1, -t) (R2, t) (R2, -t)
    cv::Mat R1, R2, t;

    //!从本质矩阵求解两个R解和两个t解，共四组解
    //!不过由于两个t解互为相反数，因此这里先只获取一个
    //!虽然这个函数对t有归一化，但并没有决定单目整个SLAM过程的尺度. 
    //!因为 CreateInitialMapMonocular 函数对3D点深度会缩放，然后反过来对 t 有改变.
    //!注意下文中的符号“'”表示矩阵的转置
    //!                         |0 -1  0|
    //!E = U Sigma V'   let W = |1  0  0|
    //!                         |0  0  1|
    //!得到4个解 E = [R|t]
    //!R1 = UWV' R2 = UW'V' t1 = U3 t2 = -U3
    // Recover the 4 motion hypotheses
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    //!分别验证求解的4种R和t的组合，选出最佳组合
    //!原理：若某一组合使恢复得到的3D点位于相机正前方的数量最多，那么该组合就是最佳组合
    //!实现：根据计算的解组合成为四种情况,并依次调用 Initializer::CheckRT() 进行检查,得到可以进行三角化测量的点的数目
	//!定义四组解分别在对同一匹配点集进行三角化测量之后的特征点空间坐标
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    //!定义四组解分别对同一匹配点集的有效三角化结果，True or False
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    //!定义四种解对应的比较大的特征点对视差角
    float parallax1,parallax2, parallax3, parallax4;
    //!使用同样的匹配点分别检查四组解，记录当前计算的3D点在摄像头前方且投影误差小于阈值的个数，记为有效3D点个数
    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    //!统计匹配的特征点对中属于内点(Inlier)或有效点个数
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    //!参考SLAM十四讲第二版p170-p171
    //!H = K * (R - t * n / d) * K_inv
    //!其中: K表示内参数矩阵
    //!      K_inv 表示内参数矩阵的逆
    //!      R 和 t 表示旋转和平移向量
    //!      n 表示平面法向量
    //!令 H = K * A * K_inv
    //!则 A = k_inv * H * k
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;
    //!对矩阵A进行SVD分解
    //!A 等待被进行奇异值分解的矩阵
    //!w 奇异值矩阵
    //!U 奇异值分解左矩阵
    //!Vt 奇异值分解右矩阵，注意函数返回的是转置
    //!cv::SVD::FULL_UV 全部分解
    //!A = U * w * Vt
    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    //!通过转置操作将右奇异向量矩阵 Vt 转为 V
    V=Vt.t();
    //!计算变量s = det(U) * det(V)
    //!左奇异向量矩阵 U 和右奇异向量矩阵 Vt 行列式的乘积
    //!因为det(V)==det(Vt), 所以 s = det(U) * det(Vt)
    float s = cv::determinant(U)*cv::determinant(Vt);
    //!取得矩阵的各个奇异值
    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);
    //!SVD分解正常情况下特征值di应该是正的，且满足d1>=d2>=d3
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }
    //!在ORBSLAM中没有对奇异值 d1 d2 d3按照论文中描述的关系进行分类讨论, 而是直接进行了计算
    //!定义8中情况下的旋转矩阵、平移向量和空间向量
    //!存在 4 种可能的相机运动解组合。单应矩阵 H 通过表示两个图像平面之间的映射，理论上可以提取出 4 组不同的旋转矩阵 RR 和平移向量 TT 的组合。原因在于解的符号不唯一（正向或反向的旋转和平移），并且在一些情况下也涉及到平面法向量的不同可能性。
    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    //!讨论法向量vn大于0时的4组解
    //!根据论文eq.(12)有
    //!x1 = e1 * sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3))
    //!x2 = 0
    //!x3 = e3 * sqrt((d2 * d2 - d2 * d2) / (d1 * d1 - d3 * d3))
    //!令 aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3))
    //!   aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3))
    //!则
    //!x1 = e1 * aux1
    //!x3 = e3 * aux2
    //!因为 e1,e2,e3 = 1 or -1
    //!所以有x1和x3有四种组合
    //!x1 =  {aux1,aux1,-aux1,-aux1}
    //!x3 =  {aux3,-aux3,aux3,-aux3}
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    //!这些数组的存在是为了表达四种可能的旋转和平移组合（也就是 4 个候选解）：
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    //!根据论文eq.(13)有
    //!sin(theta) = e1 * e3 * sqrt(( d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) /(d1 + d3)/d2
    //!cos(theta) = (d2* d2 + d1 * d3) / (d1 + d3) / d2 
    //!令  aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2)
    //!则  sin(theta) = e1 * e3 * aux_stheta
    //!    cos(theta) = (d2*d2+d1*d3)/((d1+d3)*d2)
    //!因为 e1 e2 e3 = 1 or -1
    //!所以 sin(theta) = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta}
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);
    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    //!计算旋转矩阵 R'
    //!根据不同的e1 e3组合所得出来的四种R t的解
    //!      | ctheta      0   -aux_stheta|       | aux1|
    //! Rp = |    0        1       0      |  tp = |  0  |
    //!      | aux_stheta  0    ctheta    |       |-aux3|

    //!      | ctheta      0    aux_stheta|       | aux1|
    //! Rp = |    0        1       0      |  tp = |  0  |
    //!      |-aux_stheta  0    ctheta    |       | aux3|

    //!      | ctheta      0    aux_stheta|       |-aux1|
    //! Rp = |    0        1       0      |  tp = |  0  |
    //!      |-aux_stheta  0    ctheta    |       |-aux3|

    //!      | ctheta      0   -aux_stheta|       |-aux1|
    //! Rp = |    0        1       0      |  tp = |  0  |
    //!      | aux_stheta  0    ctheta    |       | aux3|
    for(int i=0; i<4; i++)
    {   
        //!生成Rp，就是eq.(8) 的 R'
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;
        //!eq.(8) 计算R
        cv::Mat R = s*U*Rp*Vt;
        //!保存
        vR.push_back(R);
        //!eq. (14) 生成tp
        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;
        //!这里虽然对t有归一化，并没有决定单目整个SLAM过程的尺度
        //!因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
        //!eq.(8)恢复原始的t
        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));
        //!构造法向量np
        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];
        //!eq.(8) 恢复原始的法向量
        cv::Mat n = V*np;
        //!保持平面法向量向上
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    //!讨论 d' < 0 时的 4 组解
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    //!考虑到e1,e2的取值，这里的sin_theta有两种可能的解
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};
    //!对于每种由e1 e3取值的组合而形成的四种解的情况
    for(int i=0; i<4; i++)
    {
        //!计算旋转矩阵 R'
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;
        //!恢复出原来的R
        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);
        //!构造tp
        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;
        //!恢复出原来的t
        cv::Mat t = U*tp;
        //!归一化之后加入到vector中,要提供给上面的平移矩阵都是要进行过归一化的
        vt.push_back(t/cv::norm(t));
        //!构造法向量np
        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];
        //!恢复出原来的法向量
        cv::Mat n = V*np;
        //!保证法向量指向上方
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //!最好的good点
    int bestGood = 0;
    //!其次最好的good点
    int secondBestGood = 0;
    //!最好的解的索引，初始值为-1    
    int bestSolutionIdx = -1;
    //!最大的视差角
    float bestParallax = -1;
    //!存储最好解对应的，对特征点对进行三角化测量的结果
    vector<cv::Point3f> bestP3D;
    //!最佳解所对应的，那些可以被三角化测量的点的标记 针对于参考帧的
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    //!对 8 组解进行验证，并选择产生相机前方最多3D点的解为最优解
    for(size_t i=0; i<8; i++)
    {
        //!第i组解对应的比较大的视差角
        float parallaxi;
        //!三角化测量之后的特征点的空间坐标
        vector<cv::Point3f> vP3Di;
        //!特征点对是否被三角化的标记
        vector<bool> vbTriangulatedi;
        //!记录每一组R,T解中遍历完所有的特征点对的good数
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);
        //!更新历史最优和次优的解
        //!保留最优的和次优的解.保存次优解的目的是看看最优解是否突出
        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        //!如果当前组的good计数小于历史最优但却大于历史次优
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    //!原理
    //!Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
    //!x' = P'X  x = PX
    //!它们都属于 x = aPX模型
    //!                        |X|
    //!|x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
    //!|y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
    //!|z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
    //!采用DLT的方法：x叉乘PX = 0
    //!|yp2 -  p1|     |0|
    //!|p0 -  xp2| X = |0|
    //!|xp1 - yp0|     |0|
    //!两个点:
    //!|yp2   -  p1  |     |0|
    //!|p0    -  xp2 | X = |0| ===> AX = 0
    //!|y'p2' -  p1' |     |0|
    //!|p0'   - x'p2'|     |0|
    //!变成程序中的形式：
    //!|xp2  - p0 |     |0|
    //!|yp2  - p1 | X = |0| ===> AX = 0
    //!|x'p2'- p0'|     |0|
    //!|y'p2'- p1'|     |0|
    //!然后就组成了一个四元一次正定方程组，SVD求解，右奇异矩阵的最后一行就是最终的解.
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);
    //!进行奇异值分解
    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    //!根据前面的结论，奇异值分解右矩阵的最后一行其实就是解，原理类似于前面的求最小二乘解，四个未知数四个方程正好正定
	//!@别忘了我们更习惯用列向量来表示一个点的空间坐标
    x3D = vt.row(3).t();
    //!为了符合齐次坐标的形式，使最后一维为1
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    //!计算特征点X,Y坐标的均值 meanX, meanY
    float meanX = 0;
    float meanY = 0;
    //!获取当前帧的所有特征点数目
    const int N = vKeys.size();
    //!设置用来存储归一后特征点的向量大小，和归一化前保持一致
    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }
    //!计算X、Y坐标的均值
    meanX = meanX/N;
    meanY = meanY/N;
    //!计算特征点X,Y坐标离均值的平均偏离程度 meanDevX, meanDevY，注意不是标准差
    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;
        //!累计这些特征点偏离横纵坐标均值的程度
        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }
    //!求出平均到每个点上，其坐标偏离横纵坐标均值的程度；将其倒数作为一个尺度缩放因子
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        //!对，就是简单地对特征点的坐标进行进一步的缩放
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }
    //!计算归一化矩阵：其实就是前面做的操作用矩阵变换来表示而已
    //!|sX  0  -meanx*sX|
    //!|0   sY -meany*sY|
    //!|0   0      1    |
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}


int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{

    //!对给出的特征点对及其R t , 通过三角化检查解的有效性，也称为 cheirality check
    // Calibration parameters
    //!从相机内参数矩阵获取相机的校正参数
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);
    //!特征点是否是good点的标记，这里的特征点指的是参考帧中的特征点
    vbGood = vector<bool>(vKeys1.size(),false);
    //!重设存储空间坐标的点的大小 设置的是以参考帧的特征点数量大小
    vP3D.resize(vKeys1.size());
    //!存储计算出来的每对特征点的视差
    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    //!Step 1：计算相机的投影矩阵  
    //!投影矩阵P是一个 3x4 的矩阵，可以将空间中的一个点投影到平面上，获得其平面坐标，这里均指的是齐次坐标。
    //!对于第一个相机是 P1=K*[I|0] P=K[R∣t]
    //!投影矩阵 P1​ 用于将三维点 X=[X,Y,Z,1]T 投影到图像平面上
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));
    //!第一个相机的光心设置为世界坐标系下的原点  创建了一个 3 行 1 列 的矩阵，所有元素初始化为 0，数据类型为 32 位浮点数（CV_32F）
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    //!计算第二个相机的投影矩阵 P2=K*[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;
    //!第二个相机的光心在世界坐标系下的坐标 平移向量 t 描述了相机2的原点（光心）相对于相机1的原点的平移
    cv::Mat O2 = -R.t()*t;
    //!在遍历开始前，先将good点计数设置为0
    int nGood=0;
    //!开始遍历所有的特征点对
    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        //!跳过outliers
        if(!vbMatchesInliers[i])
            continue;
        //!kp1和kp2是匹配好的有效特征点
        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        //!存储三维点的的坐标
        cv::Mat p3dC1;
        //!通过奇异值分解后得到经过投影矩阵之后的三维点坐标
        Triangulate(kp1,kp2,P1,P2,p3dC1);
        //!检查三角化的三维点坐标是否合法（非无穷值）
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        //!通过三维点深度值正负、两相机光心视差角大小来检查是否合法 
        //!得到向量PO1
        cv::Mat normal1 = p3dC1 - O1;
        //!求取模长，其实就是距离
        float dist1 = cv::norm(normal1);
        //!同理构造向量PO2
        cv::Mat normal2 = p3dC1 - O2;
        //!求模长
        float dist2 = cv::norm(normal2);
        //!根据公式：a.*b=|a||b|cos_theta 可以推导出来下面的式子
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        //!如果深度值为负值，为非法三维点跳过该匹配点对
        //!视差是指相同的三维点在两个相机（或两个视角）中投影到图像平面上的位置差异。
        //!重投影误差是指将三维点重投影到图像平面时，投影点与实际图像点之间的距离误差。它衡量的是三维点与图像点在投影过程中的不一致性。通常，重投影误差越小，三维重建的准确性越高。
        //!对于远处的物体，视差非常小，使得计算得到的三维点深度高度不准确
        //?视差比较小时，重投影误差比较大。这里0.99998 对应的角度为0.36°,这里不应该是 cosParallax>0.99998 吗？
        //?因为后面判断vbGood 点时的条件也是 cosParallax<0.99998 
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        //!将空间点p3dC1变换到第2个相机坐标系下变为p3dC2
        cv::Mat p3dC2 = R*p3dC1+t;
        //!判断过程和上面的相同
        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        //!计算空间点在参考帧和当前帧上的重投影误差，如果大于阈值则舍弃
        //!计算3D点在第一个图像上的投影误差
        //!投影到参考帧图像上的点的坐标x,y
        float im1x, im1y;
        //!这个使能空间点的z坐标的倒数
        float invZ1 = 1.0/p3dC1.at<float>(2);
        //!投影到参考帧图像上。因为参考帧下的相机坐标系和世界坐标系重合，因此这里就直接进行投影就可以了
        //!先将三维坐标x，y，z都除以z，进行归一化，然后将归一化坐标转换成图像坐标
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;
        //!参考帧上的重投影误差，这个的确就是按照定义来的
        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);
        //!重投影误差太大，跳过淘汰
        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        //!计算3D点在第二个图像上的投影误差，计算过程和第一个图像类似
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;
        //!计算重投影误差
        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);
        //!重投影误差太大，跳过淘汰
        if(squareError2>th2)
            continue;
        //!统计经过检验的3D点个数，记录3D点视差角
        //!如果运行到这里就说明当前遍历的这个特征点对靠谱，经过了重重检验，说明是一个合格的点，称之为good点 
        vCosParallax.push_back(cosParallax);
        //!存储这个三角化测量后的3D点在世界坐标系下的坐标 基于的参考帧
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        //!good点计数++
        nGood++;
        //!判断视差角，只有视差角稍稍大一丢丢的才会给打good点标记
		//? bug 我觉得这个写的位置不太对。你的good点计数都++了然后才判断，不是会让good点标志和good点计数不一样吗
        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {   
        //!vCosParallax 存储的是不同匹配点对的视差角余弦值
        //!视差角余弦值 cos⁡(θ)cos(θ) 越接近1，说明两个相机看到的点的视差角度 θθ 越小，三角化出的深度精度越低
        //!从小到大排序，注意vCosParallax值越大，视差越小
        sort(vCosParallax.begin(),vCosParallax.end());
        //!为了从 vCosParallax 中选择一个具有较大视差的点对。
		//!确保选出的视差点对数量最多为50个，但如果实际点对数少于50，则选择所有点对。这一步的目的是为了避免选取太多的点对，影响计算效率，同时保证至少有足够的点用于计算。
        size_t idx = min(50,int(vCosParallax.size()-1));
        //!将这个选中的角弧度制转换为角度制
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    //!对本质矩阵进行奇异值分解
	//!准备存储对本质矩阵进行奇异值分解的结果
    cv::Mat u,w,vt;
    //!对本质矩阵进行奇异值分解
    cv::SVD::compute(E,w,u,vt);
    //!左奇异值矩阵U的最后一列就是t，对其进行归一化
    u.col(2).copyTo(t);
    t=t/cv::norm(t);
    //!构造一个绕Z轴旋转pi/2的旋转矩阵W，按照下式组合得到旋转矩阵 R1 = u*W*vt
    //!计算完成后要检查一下旋转矩阵行列式的数值，使其满足行列式为1的约束
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;
    //!计算
    R1 = u*W*vt;
    //!旋转矩阵有行列式为+1的约束，所以如果算出来为负值，需要取反
    if(cv::determinant(R1)<0)
        R1=-R1;
    //!同理将矩阵W取转置来按照相同的公式计算旋转矩阵R2 = u*W.t()*vt
    R2 = u*W.t()*vt;
    //!旋转矩阵有行列式为1的约束
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
