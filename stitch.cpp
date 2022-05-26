#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
using namespace std;
using namespace cv;
using namespace cv::detail;

//��������

//ͼ�����ݼ�·��
vector<String> img_names;

//��Ԥ��ģʽ���г��򣬱�����ģʽҪ�죬�����ͼ��ֱ��ʵͣ�ƴ�ӵķֱ���Ϊcompose_megapix
bool preview = false;
double compose_megapix = -1;

//cuda����
bool try_cuda = false;

//ͼ��ƥ��ֱ���
double work_megapix = 0.6;
//ƴ�ӽӷ�ֱ���
double seam_megapix = 0.1;

//����ͼ����ͬһȫ��ͼ�����Ŷ�
float conf_thresh = 1.f;
//�����������ŵȼ��������ƥ�������ν���ƥ�����ı�ֵ
float match_conf = 0.3f;

//����У��
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;

//��ƥ���ͼ���Ե����ʽ���浽�ļ���
bool save_graph = false;
string save_graph_to;

//�����任�ں�ƽ�����ͣ��������������
string warp_type = "spherical";

//�ع⣨���գ����������Լ�����
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;

//ƴ�ӷ�϶���Ʒ���
string seam_find_type = "gc_color";

//�ںϷ���
int blend_type = Blender::MULTI_BAND;
//�ں�ǿ��
float blend_strength = 5;

//�������
string result_name = "result.jpg";


int main(int argc, char* argv[]){
    int64 app_start_time = getTickCount();

    //����ͼ��洢·��
    img_names.push_back("1.jpg");
    img_names.push_back("2.jpg");
    img_names.push_back("3.jpg");
    //img_names.push_back("4.jpg");
    //img_names.push_back("5.jpg");

    int num_images = img_names.size();
    if (num_images < 2){
        cout<<"������Ҫ����ͼƬ���ܽ���ƴ��" << endl;
        return -1;
    }

    //����ߴ�
    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    cout<<"��ʼ�����������" << endl;
    int64 t = getTickCount();

    Ptr<Feature2D> finder = SIFT::create();

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; i++){
        full_img = imread(img_names[i]);
        full_img_sizes[i] = full_img.size();

        if (work_megapix < 0){
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else{
            if (!is_work_scale_set){
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set){
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        computeImageFeatures(finder, img, features[i]);
        features[i].img_idx = i;
        cout << "��" << i + 1 << "��ͼƬ����������" << features[i].keypoints.size() << endl;

        resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    cout << "���������ʱ" << ((getTickCount() - t) / getTickFrequency()) << "��" << endl;
    cout << "��ʼƥ���������" << endl;
    t = getTickCount();

    vector<MatchesInfo> pairwise_matches;
    Ptr<FeaturesMatcher> matcher;
    matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();

    cout << "ƥ�����������ʱ" << ((getTickCount() - t) / getTickFrequency()) << "��" << endl;

    if (save_graph){
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }

    //����������ͼ�ν���������ƥ�䣬Ȼ��ʹ�ò鲢��������ͼƬ��ƥ���ϵ�ҳ���ֻ����ͬ����ͼƬ
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (int i = 0; i < indices.size(); ++i){
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;

    // ���ͬ����ͼƬ�Ƿ���
    num_images = (img_names.size());
    if (num_images < 2){
        cout << "ͬ����ͼƬ����" << endl;
        return -1;
    }

    //���㵥Ӧ�����������������
    Ptr<Estimator> estimator = makePtr<HomographyBasedEstimator>();

    vector<CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras)){
        cout << "�޷����㵥Ӧ����";
        return -1;
    }

    for (int i = 0; i < cameras.size(); i++){
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        cout << "���#" << indices[i] + 1 << "����:\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R << endl;
    }


    //ʹ��Bundle Adjustment����������ͼƬ�����������У��
    Ptr<detail::BundleAdjusterBase> adjuster;

    adjuster = makePtr<detail::BundleAdjusterRay>();
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::ones(3, 3, CV_8U);
    refine_mask(2, 2) = 0;
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras)){
        cout << "�����������ʧ��" << endl;
        return -1;
    }

    // Ѱ����λ����
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i){
        cout << "���#" << indices[i] + 1 << "����:\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R;
        focals.push_back(cameras[i].focal);
    }
    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    //���ν���
    vector<Mat> rmats;
    for (int i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, wave_correct);
    for (int i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];
    

    cout << "��ʼ���������任" << endl;
    t = getTickCount();

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

    // ������Ĥ
    for (int i = 0; i < num_images; ++i){
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // ��ͼ���Լ���Ĥ������ͬ�ı任

    Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarperGpu>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarperGpu>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            warper_creator = makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            warper_creator = makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            warper_creator = makePtr<cv::StereographicWarper>();
    }

    if (!warper_creator)
    {
        cout << "����������任��ʽ" << warp_type << "'\n";
        return 1;
    }

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0, 0) *= swa; K(0, 2) *= swa;
        K(1, 1) *= swa; K(1, 2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    cout << "�����任��ʱ��" << ((getTickCount() - t) / getTickFrequency()) << "��" << endl;

    cout << "��ʼ�����عⲹ��" << endl;
    t = getTickCount();


    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    if (dynamic_cast<GainCompensator*>(compensator.get()))
    {
        GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
        gcompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
    {
        ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
        ccompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<BlocksCompensator*>(compensator.get()))
    {
        BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
        bcompensator->setNrFeeds(expos_comp_nr_feeds);
        bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
        bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
    }

    compensator->feed(corners, images_warped, masks_warped);

    cout << "�عⲹ����ʱ��" << ((getTickCount() - t) / getTickFrequency()) << "��" << endl;

    cout << "����Ѱ�ҽӷ��Խ����ں�";

    t = getTickCount();


    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = makePtr<detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = makePtr<detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!seam_finder){
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return 1;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    cout << "Ѱ�ҽӷ���ʱ��" << ((getTickCount() - t) / getTickFrequency()) << "��" << endl;

    //�ͷ��ڴ�
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    cout << "���ڽ���ƴ���ں�" << endl;
    t = getTickCount();


    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    Ptr<Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        cout << "����ƴ��ͼ��#" << indices[img_idx] + 1 << endl;

        //�ж��Ƿ�����
        full_img = imread(img_names[img_idx]);
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        //��ͼ���Լ���Ĥ����ͬ���������任
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // �عⲹ��
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;

        if (!blender){
            blender = Blender::createDefault(blend_type, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_cuda);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
                cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                fb->setSharpness(1.f / blend_width);
                cout << "Feather blender, sharpness: " << fb->sharpness() << endl;
            }
            blender->prepare(corners, sizes);
        }


        // �ں�
        blender->feed(img_warped_s, mask_warped, corners[img_idx]);       
    }


    Mat result, result_mask;
    blender->blend(result, result_mask);

    cout << "ƴ���ں���ʱ " << ((getTickCount() - t) / getTickFrequency()) << "��" << endl;

    imwrite(result_name, result);
    

    cout << "����ʱ��" << ((getTickCount() - app_start_time) / getTickFrequency()) << "��" << endl;
    return 0;
}