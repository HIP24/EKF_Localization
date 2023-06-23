
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <laser_geometry/laser_geometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <pcl_ros/point_cloud.h>
//#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <map>

class My_Filter {
public:
    My_Filter();
    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan);

private:
    ros::NodeHandle node_;
    laser_geometry::LaserProjection projector_;
    tf::TransformListener tfListener_;
    ros::Publisher point_cloud_publisher_;
    ros::Publisher marker_pub;
    ros::Publisher valuePub;
    ros::Subscriber scan_sub_;
    struct Landmark {
    double landmarkinnerthresh;
    double landmarkouterthresh;
    double groundtruth_X;
    double groundtruth_Y;
    int signature;};
    std::map<int, Landmark> landmarkMap;
};

My_Filter::My_Filter(){
    scan_sub_ = node_.subscribe<sensor_msgs::LaserScan> ("/scan", 100, &My_Filter::scanCallback, this);
    point_cloud_publisher_ = node_.advertise<sensor_msgs::PointCloud2> ("/cloud", 1000, false);
    marker_pub = node_.advertise<visualization_msgs::MarkerArray>("detected_circles_markers", 1);
    valuePub = node_.advertise<geometry_msgs::PoseArray>("circle_angle_range", 1);
    Landmark landmark1;
    landmark1.landmarkinnerthresh = 0.30;
    landmark1.landmarkouterthresh = 0.40;
    landmark1.groundtruth_X = 1.1;
    landmark1.groundtruth_Y = -1.1;
    landmark1.signature = 1;

    Landmark landmark2;
    landmark2.landmarkinnerthresh = 0.40;
    landmark2.landmarkouterthresh = 0.50;
    landmark2.groundtruth_X = 0;
    landmark2.groundtruth_Y = 1.1;
    landmark2.signature = 2;

    Landmark landmark3;
    landmark3.landmarkinnerthresh = 0.20;
    landmark3.landmarkouterthresh = 0.30;
    landmark3.groundtruth_X = -1.1;
    landmark3.groundtruth_Y = -1.1;
    landmark3.signature = 3;

    landmarkMap[landmark1.signature] = landmark1;
    landmarkMap[landmark2.signature] = landmark2;
    landmarkMap[landmark3.signature] = landmark3;

}


void My_Filter::scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan){
    sensor_msgs::PointCloud2 cloud;
    projector_.transformLaserScanToPointCloud("base_scan", *scan, cloud, tfListener_);
    point_cloud_publisher_.publish(cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(cloud, *pcl_cloud);

    pcl::PointIndices inlierIndices;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZ> segmentation;
    segmentation.setInputCloud(pcl_cloud);
    segmentation.setMaxIterations(100000);
    segmentation.setModelType(pcl::SACMODEL_CIRCLE2D);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(0.5);
    segmentation.setOptimizeCoefficients(true);
    visualization_msgs::MarkerArray markerArray;
    int markerId = 0;
    geometry_msgs::PoseArray poseArray;
    poseArray.header.stamp = ros::Time::now();
    //poseArray.header.frame_id = "base_link";

    for (const auto& landmarkPair : landmarkMap) {
        ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);  // Suppress Warning messages
        ros::console::notifyLoggerLevelsChanged();
        const Landmark& landmark = landmarkPair.second;
        segmentation.setRadiusLimits(landmark.landmarkinnerthresh, landmark.landmarkouterthresh);

        segmentation.segment(inlierIndices, *coefficients);

        if (inlierIndices.indices.size() == 0) {
            continue;
        }

        double x = coefficients->values[0];
        double y = coefficients->values[1];
        double radius = coefficients->values[2];

        std::cout << "Landmark: "<< landmark.signature << std::endl;    
        std::cout << "x: "<< x << std::endl;
        std::cout << "y: " << y << std::endl;
        std::cout << "radius: " << radius << std::endl;
        std::cout << "-----" << std::endl;

        // Populate the detected landmark pose
        geometry_msgs::Pose pose;
        pose.position.x = x;
        pose.position.y = y;
        pose.position.z = landmark.signature;
        pose.orientation.x = landmark.groundtruth_X;
        pose.orientation.y = landmark.groundtruth_Y;
        pose.orientation.z = 0.0;
        pose.orientation.w = 1.0;

        poseArray.poses.push_back(pose);
        visualization_msgs::Marker marker;
        marker.header.frame_id = "base_scan";
        marker.header.stamp = ros::Time::now();
        marker.ns = "circles";
        marker.id = markerId++;
        marker.type = visualization_msgs::Marker::CYLINDER;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = 0.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = radius * 2;
        marker.scale.y = radius * 2;
        marker.scale.z = 0.01;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        markerArray.markers.push_back(marker);
    }

    // Publish the detected landmark poses as a single message
    valuePub.publish(poseArray);
    poseArray.poses.clear();
    marker_pub.publish(markerArray);
}

int main(int argc, char** argv){
    ros::init(argc, argv, "combined_node");

    My_Filter filter;

    ros::spin();

    return 0;
}
