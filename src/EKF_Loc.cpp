#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>

#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>

#include <sensor_msgs/LaserScan.h>
#include <vector>
#include <map>
#include <string>

#include <geometry_msgs/PoseStamped.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <visualization_msgs/Marker.h>

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

// Define counters
int odomPrintCount = 0;
int scanPrintCount = 0;
int cmd_velPrintCount = 0;
double robot_x, robot_y, robot_quat_z, robot_quat_w;

// Create a struct to hold the x, y coordinates, and signature vector of each landmark
struct Landmark
{
  double landX;
  double landY;
  std::vector<double> signature;
};

// Create a map to hold all the landmarks
std::map<std::string, Landmark> landmarks = {
    {"bottom", {-2.5, 0, {1, 0, 0}}},     // Red landmark
    {"head", {2.2, 0, {0, 1, 0}}},          // Green landmark
    {"left_wall", {0, 2.4, {0, 0, 1}}},   // Blue landmark
    {"right_wall", {0, -2.4, {1, 1, 0}}}, // Yellow landmark
};

class EKF_Loc
{
private:
  Eigen::Vector2d u_t;
  std::vector<std::pair<double, double>> z_t; // Each pair is <range, angle>

  Eigen::VectorXd mt;
  Eigen::MatrixXd cov;

  Eigen::MatrixXd A;
  Eigen::MatrixXd B;
  Eigen::MatrixXd R;

  Eigen::MatrixXd Q;

  ros::Publisher posePub;
  ros::Publisher marker_pub;

  double roll, pitch, yaw;
  
  double theta = 0; 
  double delta_t = 0.1; // time step
  double sigma = 0.1;   // noise standard deviation
  double sigma_x = 0.1; 
  double sigma_y = 0.1; 
  double sigma_theta = 0.1;
  
  double sigma_x0 = 0.1;
  double sigma_y0 = 0.1;
  double sigma_theta0 = 0.1;


    ros::NodeHandle node_;
    laser_geometry::LaserProjection projector_;
    tf::TransformListener tfListener_;
    ros::Publisher point_cloud_publisher_;
    ros::Publisher marker_pub2;
    ros::Publisher valuePub;
    ros::Subscriber scan_sub_;
    struct Landmark {
    double landmarkinnerthresh;
    double landmarkouterthresh;
    double groundtruth_X;
    double groundtruth_Y;
    int signature;};
    std::map<int, Landmark> landmarkMap;

public:
  EKF_Loc()
  {
    // initialize with the initial pose [x, y, theta, v, omega]
    mt.resize(3);
    mt << 0.5, 0.5, 0; 

    // Initialize initial covariance
    cov.resize(3, 3);
    cov << pow(sigma_x0,2),    0 ,            0 ,
        0 ,            pow(sigma_y0 ,2),    0 , 
        0 ,            0 ,            pow(sigma_theta0 ,2);

    // initialize process noise
    R.resize(3, 3);
    R << pow(sigma_x,2),    0 ,            0 ,       
        0 ,            pow(sigma_y ,2),    0 , 
        0 ,            0 ,            pow(sigma_theta ,2);

    // initialize measurement noise
    Q = pow(sigma, 2) * Eigen::Matrix2d::Identity();

    // initialize nodeHandle for ekf_pose
    ros::NodeHandle nodeHandle;
    posePub = nodeHandle.advertise<geometry_msgs::PoseStamped>("/ekf_pose", 10);

    marker_pub = nodeHandle.advertise<visualization_msgs::Marker>("/ellipse", 10);

    scan_sub_ = node_.subscribe<sensor_msgs::LaserScan> ("/scan", 100, &EKF_Loc::scanCallback, this);
    point_cloud_publisher_ = node_.advertise<sensor_msgs::PointCloud2> ("/cloud", 1000, false);
    marker_pub2 = node_.advertise<visualization_msgs::MarkerArray>("detected_circles_markers", 1);
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

  // cmd_vel callback function
  void cmd_velCallback(const geometry_msgs::Twist::ConstPtr& commandMsg)
{
    ++cmd_velPrintCount;
    double v = commandMsg->linear.x;;  // Forward velocity
    double w = commandMsg->angular.z;; // Rotational velocity
    u_t << v, w;
    poseEstimation();
}

  // odom callback function
  void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
  { // std::cout << "odomPrintCount: " << odomPrintCount << std::endl;
    // std::cout << "Received odom: (" << *msg << ")" << std::endl;
    ++odomPrintCount;

  // Update the robot's x and y coordinates
  robot_x = msg->pose.pose.position.x;
  robot_y = msg->pose.pose.position.y;

  // Update the robot's orientation quaternion (z and w components)
  //robot_quat_x = msg->pose.pose.orientation.x;
  //robot_quat_y = msg->pose.pose.orientation.y;
  robot_quat_z = msg->pose.pose.orientation.z;
  robot_quat_w = msg->pose.pose.orientation.w;
    // std::cout << "Robot's position: (" << robot_x << ", " << robot_y << ")" << std::endl;

      // Convert robot's orientation quaternion to yaw angle
  tf2::Quaternion q;
  q.setX(0);
  q.setY(0);
  q.setZ(robot_quat_z);
  q.setW(robot_quat_w);
  tf2::Matrix3x3 m(q);
  m.getRPY(roll, pitch, yaw);

  // Update theta with the current yaw angle
  theta = yaw;
  }

  // scan callback function
  void scanCallback(const sensor_msgs::LaserScan::ConstPtr &scan)
{
  ++scanPrintCount;
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
    marker_pub2.publish(markerArray);
  
}

  void printMtCov()
  {
    //std::cout << "Cov: " << std::endl << cov << std::endl;
    //std::cout << "Pose -> x = " << mt(0) << ", y = " << mt(1) << ", theta = " << mt(2) << ", v = " << mt(3) << ", w = " << mt(4) << std::endl;
      std::cout << "Robot's real position: (x = " << robot_x << ", y = " << robot_y << ")" << std::endl;
      std::cout << "Robot's estimated position: (x = " << mt(0) << ", y = " << mt(1) << ")" << std::endl;
  }

  // Getter methods
  Eigen::VectorXd getMt() const
  {
    std::cout << "mt: \r\n"<< mt << std::endl;
    return mt;
  }

  Eigen::MatrixXd getCov() const
  {
    std::cout << "cov: \r\n" << cov << std::endl;
    return cov;
  }

  // Setter methods
  void setMt(const Eigen::VectorXd &newMt)
  {
    mt = newMt;
  }

  void setCov(const Eigen::MatrixXd &newCov)
  {
    cov = newCov;
  }

  void predictionStep()
{
  // prediction
  Eigen::VectorXd newMt;
  Eigen::MatrixXd newCov;
  std::tie(newMt, newCov) = predict(u_t, getMt(), getCov());
  setMt(newMt);
  setCov(newCov);
  std::cout << "\n\r############## After prediction ##############" << std::endl;
  printMtCov();
}

  void correctionStep()
{
    
  }

  void correct(const std::vector<std::pair<double, double>> &z_t, const std::vector<std::string> &c_t)
{

}
std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(const Eigen::Vector2d &ut, const Eigen::VectorXd &mt_prev, const Eigen::MatrixXd &cov_prev)
  {
    std::cout << "Theta:\r\n " << theta << std::endl;
    std::cout << "v:\r\n " << u_t(0) << std::endl;
    std::cout << "w:\r\n " << u_t(1) << std::endl;
    //std::cout << "A:\r\n " << A << std::endl;
    //std::cout << "B:\r\n " << B << std::endl;

    A.resize(3, 3);
    A << 1, 0, -u_t[0] * delta_t * sin(theta),
         0, 1, u_t[0]* delta_t * cos(theta),
         0, 0, 1;

    mt(0) += u_t[0] * delta_t * cos(theta);
    mt(1) += u_t[0] * delta_t * sin(theta); 
    mt(2) += u_t[1] * delta_t; 
    mt(2) = std::atan2(std::sin(mt(2)), std::cos(mt(2)));

    B.resize(3, 2);
    B << u_t[0] * delta_t * cos(theta), 0,
         u_t[0] * delta_t * sin(theta), 0,
         0, u_t[1] * delta_t;

    //mt = A * mt_prev + B * ut;              // prediction
    cov = A * cov_prev * A.transpose() + R; // update error covariance
    return std::make_pair(mt, cov);
  }

  void poseEstimation()
{
  predictionStep();
    //if(landmarkDetected){
  correctionStep();
  //}


  publishPose();
  publishCovarianceEllipse();
}

  void publishPose()
{
  geometry_msgs::PoseStamped poseMsg;

  // Set the frame ID and timestamp
  poseMsg.header.frame_id = "map";
  poseMsg.header.stamp = ros::Time::now();

  // Set the pose data
  poseMsg.pose.position.x = mt(0);
  poseMsg.pose.position.y = mt(1);

  double roll = 0;
  double pitch = 0;
  double yaw = mt(2); // your yaw angle in radians

  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);

  geometry_msgs::Quaternion quat_msg;
  quat_msg.x = q.x();
  quat_msg.y = q.y();
  quat_msg.z = q.z();
  quat_msg.w = q.w();

  poseMsg.pose.orientation = quat_msg;

  //tf2::Quaternion quat;
  //quat.setRPY(0, 0, yaw); 
  //poseMsg.pose.orientation = tf2::toMsg(quat);

  // Publish the pose
  posePub.publish(poseMsg);
}

  void publishCovarianceEllipse()
{
    // Compute the eigenvalues and eigenvectors of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov.block<2, 2>(0, 0));
    Eigen::Vector2d eigenvalues = solver.eigenvalues();
    Eigen::Matrix2d eigenvectors = solver.eigenvectors();

    // Compute the angle of rotation and the semi-major and semi-minor axes of the ellipse
    double angle = atan2(eigenvectors(1, 0), eigenvectors(0, 0));
    double a = sqrt(eigenvalues(0));
    double b = sqrt(eigenvalues(1));

    // Create a marker message to represent the ellipse
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "ellipse";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = mt(0);
    marker.pose.position.y = mt(1);
    marker.pose.position.z = 0;

    double roll = 0;
    double pitch = 0;
    double yaw = mt(2); // your yaw angle in radians

    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);

    geometry_msgs::Quaternion quat_msg;
    quat_msg.x = q.x();
    quat_msg.y = q.y();
    quat_msg.z = q.z();
    quat_msg.w = q.w();

    marker.pose.orientation = quat_msg;

    //tf2::Quaternion quat;
    //quat.setRPY(0, 0, angle);
    //marker.pose.orientation = tf2::toMsg(quat);

    marker.scale.x = a * 2;
    marker.scale.y = b * 2;
    marker.scale.z = 0.1;
    marker.color.a = 0.5;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

    // Publish the marker message
    marker_pub.publish(marker);
}
};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "ekf_loc");
  ros::NodeHandle nodeHandle;

  EKF_Loc ekf;

  ros::Subscriber odom_sub = nodeHandle.subscribe<nav_msgs::Odometry>("/odom", 3000, std::bind(&EKF_Loc::odomCallback, &ekf, std::placeholders::_1));
  ros::Subscriber scan_sub = nodeHandle.subscribe<sensor_msgs::LaserScan>("/scan", 500, std::bind(&EKF_Loc::scanCallback, &ekf, std::placeholders::_1));
  ros::Subscriber cmd_vel_sub = nodeHandle.subscribe<geometry_msgs::Twist>("/cmd_vel", 1000, std::bind(&EKF_Loc::cmd_velCallback, &ekf, std::placeholders::_1));

  ros::spin();

  return 0;
}











