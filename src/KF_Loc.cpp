#include <ros/ros.h>

#include <Eigen/Dense>

#include <vector>
#include <map>
#include <string>

#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>

#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <nav_msgs/Odometry.h>

#include <tf/transform_listener.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <laser_geometry/laser_geometry.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>


class KF_Loc
{
private:
  // Control vector 
  Eigen::Vector2d u_t;

  // State and covariance of the turtlebot
  Eigen::VectorXd m_t;
  Eigen::MatrixXd cov_t;
  // Matrices for prediction and correction
  Eigen::MatrixXd A;
  Eigen::MatrixXd R;
  Eigen::MatrixXd Q;
  double delta_t = 0.1;

  // Position and orientation of the turtlebot
  double roll, pitch, yaw;
  double robot_x_odom, robot_y_odom, robot_quat_z_odom, robot_quat_w_odom, robot_theta_odom;

  // Messages
  visualization_msgs::Marker landmark_shape;

  // Publishers
  ros::Publisher point_cloud_pub;
  ros::Publisher landmark_shape_pub;
  ros::Publisher landmark_pose_pub;
  ros::Publisher turtle_pose_with_cov_pub;
  ros::Publisher error_pub;

  // Landmark struct 
   struct Landmark
  {
    double innerTH;
    double outerTH;
    double landX;
    double landY;
    int signature;
  };
  // Landmark map 
  std::map<int, Landmark> landmark_map;

public:
  KF_Loc()
  {
    // Initialize with the initial pose m_t [x, y, theta]
    m_t.resize(3);
    m_t << -2, -0.5, 0;
    // Initialize initial covariance cov_t
    cov_t.resize(3, 3);
    cov_t << 0.01, 0, 0,
        0, 0.01, 0,
        0, 0, 0.01;

    // Initialize process noise R
    R.resize(3, 3);
    R << 0.001, 0, 0,
        0, 0.001, 0,
        0, 0, 0.001;

    // Initialize measurement noise Q
    Q.resize(3, 3);
    Q = Eigen::Matrix<double, 3, 3>::Identity();

    // Initialize nodeHandle for advertizing
    ros::NodeHandle nodeHandle_adv;
    turtle_pose_with_cov_pub = nodeHandle_adv.advertise<geometry_msgs::PoseWithCovarianceStamped>("/turtle_pose_with_cov", 10);
    point_cloud_pub = nodeHandle_adv.advertise<sensor_msgs::PointCloud2>("/point_cloud", 1000, false);
    landmark_shape_pub = nodeHandle_adv.advertise<visualization_msgs::MarkerArray>("landmark_shape", 10);
    landmark_pose_pub = nodeHandle_adv.advertise<geometry_msgs::PoseArray>("landmark_pose", 10);
    error_pub = nodeHandle_adv.advertise<geometry_msgs::Vector3>("/error", 10);

    // Create two Landmark objects [innerTH, outerTH, landX, landY, signature]
    landmark_map[1] = {0.3, 0.4, 1.1, -1.1, 1};
    landmark_map[3] = {0.2, 0.3, -1.1, 1.1, 3};
  }

  /*Callback for the /cmd_vel topic. It receives the linear and angular velocity commands
   for the turtlebot and updates the control input (u_t). It then calls the predict() 
   function to perform the prediction step of the Kalman Filter (KF), 
   updates the turtle's pose, and publishes the pose and covariance.*/
  void cmd_velCallback(const geometry_msgs::Twist::ConstPtr &commandMsg)
  {
    // Forward velocity
    double v = commandMsg->linear.x;
    // Rotational velocity
    double w = commandMsg->angular.z;
    // Update the control input
    u_t << v, w;

    predict();
    compareMtToReal();
    publishTurtlePoseWithCovariance(); 
  }

  /*Callback for the /scan topic. It receives laser scan data, converts it to a point
   cloud, and performs circle fitting using the RANSAC algorithm to detect landmarks. 
    For each detected landmark, it calculates the position in the map frame and calls 
    the correct() function to update the turtle's pose estimation. It also publishes
    the detected landmark poses and shapes.*/
  void scanCallback(const sensor_msgs::LaserScan::ConstPtr &scan)
  {
    // MarkerArray message to store the landmark shapes
    visualization_msgs::MarkerArray landmark_shapes; 
    // PoseArray message to store the landmark poses
    geometry_msgs::PoseArray landmark_poses;
    landmark_poses.header.stamp = ros::Time::now();     
    //PointCloud2 message to store the point cloud data
    sensor_msgs::PointCloud2 point_cloud; 
    //LaserProjection object to project laser scans into point clouds
    laser_geometry::LaserProjection laserProjection; 
    tf::TransformListener transformListener;
    laserProjection.transformLaserScanToPointCloud("base_scan", *scan, point_cloud, transformListener); 
    
    // Convert the point_cloud message into a PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(point_cloud, *pcl_cloud); 

    // Object detection thanks to https://pointclouds.org/documentation/tutorials/planar_segmentation.html
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices); 
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients); 
    pcl::SACSegmentation<pcl::PointXYZ> seg; 

    // Optional
    seg.setOptimizeCoefficients(true);
    seg.setDistanceThreshold(0.5);
    seg.setMaxIterations(100000000);

    // Required
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setModelType(pcl::SACMODEL_CIRCLE2D);
    seg.setInputCloud(pcl_cloud);

    // For each detected landmark...
    for (const auto &landmark_map_element : landmark_map)
    {
      const Landmark &landmark = landmark_map_element.second;
      seg.setRadiusLimits(landmark.innerTH, landmark.outerTH);
      seg.segment(*inliers, *coefficients);

      // If landmark detected...
      if (!inliers->indices.size() == 0)
      {
      double robotXtoLM = coefficients->values[0];
      double robotYtoLM = coefficients->values[1];
      double landmark_radius = coefficients->values[2];

      std::cout << "Landmark: "<< landmark.signature << std::endl;
      std::cout << "robotXtoLM: "<< robotXtoLM << std::endl;
      std::cout << "robotYtoLM: " << robotYtoLM << std::endl;
      std::cout << "landmark_radius: " << landmark_radius << std::endl;
      std::cout << "-----" << std::endl;

      geometry_msgs::Pose landmark_pose = populateLandmarkPose(landmark, robotXtoLM, robotYtoLM);
      landmark_poses.poses.push_back(landmark_pose);

      correct(landmark_poses.poses);
      compareMtToReal();
      publishTurtlePoseWithCovariance(); 

      int markerId = 0;
      landmark_shape = publishLandmarkPose(markerId++, robotXtoLM, robotYtoLM, landmark_radius);
      landmark_shapes.markers.push_back(landmark_shape);
      }
      else  continue;
    }

    // Publish the detected landmark_poses and landmark_shapes
    landmark_pose_pub.publish(landmark_poses);
    landmark_poses.poses.clear();
    landmark_shape_pub.publish(landmark_shapes);
  }

  /*Callback for the /odom topic. It receives the odometry data for the turtlebot 
  and extracts the robot's position and orientation. These values are extracted 
  only to compare them to the estimated pose. */
  void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
  { 
    // Update the robot's x and y coordinates
    robot_x_odom = msg->pose.pose.position.x;
    robot_y_odom = msg->pose.pose.position.y;
    robot_quat_z_odom = msg->pose.pose.orientation.z;
    robot_quat_w_odom = msg->pose.pose.orientation.w;

    // Convert robot's orientation quaternion to yaw angle
    tf2::Quaternion q;
    q.setX(0);
    q.setY(0);
    q.setZ(robot_quat_z_odom);
    q.setW(robot_quat_w_odom);
    tf2::Matrix3x3 m(q);
    m.getRPY(roll, pitch, yaw);

    // Update theta with the current yaw angle
    robot_theta_odom = yaw;
  }

  /*Performs the prediction step of the KF Localization. It updates the turtle's pose 
    estimation (m_t) and covariance (cov_t) based on the control input (u_t) and the system dynamics. It 
    calculates the state transition matrix (A) and control input matrix (B) and updates the pose 
    estimation and covariance using the motion model.*/
  void predict()
  {
    A.resize(3, 3);
    A << 1, 0, -u_t[0] * delta_t * sin(m_t(2)),
         0, 1,  u_t[0] * delta_t * cos(m_t(2)),
         0, 0,                   1;

    // Update the estimated state
    m_t(0) += u_t[0] * delta_t * cos(m_t(2));
    m_t(1) += u_t[0] * delta_t * sin(m_t(2));
    m_t(2) += u_t[1] * delta_t;
    m_t(2) = atan2(sin(m_t(2)), cos(m_t(2)));

    // Update the covariance
    cov_t = A * cov_t * A.transpose() + R;
   
  } 

  /*Performs the correction step of the KF localization. It takes a vector of detected landmark 
    poses as input and updates the turtle's pose estimation (m_t) and covariance (cov_t) based on the
    measurement information. It uses the difference between the observed and predicted landmark 
    positions to calculate the Kalman gain and update the pose estimation. */
  void correct(const std::vector<geometry_msgs::Pose>& landmarks)
  {
    Eigen::Matrix<double, 3, 3> I = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::VectorXd m_E(3);
    Eigen::MatrixXd cov_E(3,3);

    // Sum for m_t
    m_E << 0., 0., 0.;

    // Sum for cov_t
    cov_E << 0., 0., 0.,
             0., 0., 0.,
             0., 0., 0.;

    // For every landmark...
    for (const auto &landmark : landmarks)
    {
      double landX_map = landmark.position.x;
      double landY_map = landmark.position.y;
      double landX_turtle = landmark.orientation.x; 
      double landY_turtle = landmark.orientation.y;
      int signature = landmark.position.z;

      // Real landmarks in map frame
      Eigen::Vector2d delta;
      delta << landX_map - m_t(0),
               landY_map - m_t(1);
      double q = delta.transpose() * delta;
      Eigen::Vector3d z_exp_i;
      z_exp_i << sqrt(q),
                 atan2(delta(1), delta(0)) - m_t(2),
                 signature;

      // Observed landmarks in robot frame
      Eigen::Vector2d delta_obs;
      delta_obs << landX_turtle,
                   landY_turtle;
      double q_obs = delta_obs.transpose() * delta_obs ;
      Eigen::Vector3d z_i;
      z_i << sqrt(q_obs),
             atan2(delta_obs(1), delta_obs(0)),
             signature;

      Eigen::Matrix3d H_i;
      H_i << -delta(0) / sqrt(q), -delta(1) / sqrt(q), 0,
             delta(1) / q, -delta(0) / q, -1,
             0, 0, 0;

      Eigen::Matrix3d K_i = cov_t * H_i.transpose() * (H_i * cov_t * H_i.transpose() + Q).inverse();
      m_E += K_i * (z_i - z_exp_i);
      cov_E += K_i * H_i;    
    }
    // Update m_t and cov_t
     m_t += m_E;
     cov_t = (I - cov_E) * cov_t;
    }

  /*Prints the turtle's estimated position (m_t) and covariance (cov_t) to the console. 
    It also prints the real position of the robot (robot_x_odom and robot_y_odom) from odometry data.*/
  void compareMtToReal()
  {
    //std::cout << "Cov: " << std::endl << cov_t << std::endl;
    //std::cout << "Pose: " << std::endl << m_t << std::endl;
    std::cout << "Robot's real position: (x = " << robot_x_odom << ", y = " << robot_y_odom << ", theta = "<< robot_theta_odom << ")" << std::endl;
    std::cout << "Robot's estimated position: (x = " << m_t(0) << ", y = " << m_t(1) << ", theta = " << m_t(2) <<")" << std::endl;
  
    double error_x = m_t(0) - robot_x_odom;
    double error_y = m_t(1) - robot_y_odom;
    double error_theta = m_t(2) - robot_theta_odom;
    
    geometry_msgs::Vector3 error_msg;
    error_msg.x = error_x;
    error_msg.y = error_y;
    error_msg.z = error_theta;
    error_pub.publish(error_msg);
  }
  
  /*Publishes the turtle's estimated pose (m_t) and covariance (cov_t) as a 
  geometry_msgs::PoseWithCovarianceStamped message on the /turtle_pose_with_cov topic. It sets the position, 
  orientation, and covariance of the turtle's pose and publishes the message.*/
  void publishTurtlePoseWithCovariance()
{
    // Create a PoseWithCovarianceStamped message
    geometry_msgs::PoseWithCovarianceStamped pose_cov_msg;
    pose_cov_msg.header.frame_id = "map";
    pose_cov_msg.header.stamp = ros::Time::now();

    // Set the position of the pose
    pose_cov_msg.pose.pose.position.x = m_t(0);
    pose_cov_msg.pose.pose.position.y = m_t(1);

    // Set the orientation of the pose
    double roll = 0;
    double pitch = 0;
    double yaw = m_t(2); // your yaw angle in radians
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    geometry_msgs::Quaternion quat_msg;
    quat_msg.x = q.x();
    quat_msg.y = q.y();
    quat_msg.z = q.z();
    quat_msg.w = q.w();
    pose_cov_msg.pose.pose.orientation = quat_msg;

    // Set the covariance of the pose
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            pose_cov_msg.pose.covariance[i * 6 + j] = cov_t(i, j);

    // Publish the PoseWithCovarianceStamped message
    turtle_pose_with_cov_pub.publish(pose_cov_msg);
}

  /*Returns the pose of a landmark based on its distance from the robot (robotXtoLM and 
    robotYtoLM) and the provided Landmark object.*/
  geometry_msgs::Pose populateLandmarkPose(const Landmark& landmark, double robotXtoLM, double robotYtoLM)
{
    geometry_msgs::Pose landmark_pose;
    landmark_pose.position.x = landmark.landX;
    landmark_pose.position.y = landmark.landY;
    landmark_pose.position.z = landmark.signature;
    landmark_pose.orientation.x = robotXtoLM;
    landmark_pose.orientation.y = robotYtoLM;
    landmark_pose.orientation.z = 0;
    landmark_pose.orientation.w = 1;
    return landmark_pose;
}

  /*Publishes the pose of a landmark as a marker on the specified markerId topic. It uses the distances 
  between the robot and the landmark (robotXtoLM and robotYtoLM), along with the landmark's radius, to create and 
  publish the marker.*/
  visualization_msgs::Marker publishLandmarkPose(int markerId, double robotXtoLM, double robotYtoLM, double landmark_radius)
{
    visualization_msgs::Marker landmark_shape;
    landmark_shape.header.frame_id = "base_scan";
    landmark_shape.header.stamp = ros::Time::now();
    landmark_shape.ns = "landmarks";
    landmark_shape.id = markerId;
    landmark_shape.type = visualization_msgs::Marker::CYLINDER;
    landmark_shape.action = visualization_msgs::Marker::ADD;
    landmark_shape.pose.position.x = robotXtoLM;
    landmark_shape.pose.position.y = robotYtoLM;
    landmark_shape.pose.position.z = 0;
    landmark_shape.pose.orientation.x = 0;
    landmark_shape.pose.orientation.y = 0;
    landmark_shape.pose.orientation.z = 0;
    landmark_shape.pose.orientation.w = 1;
    landmark_shape.scale.x = 2 * landmark_radius;
    landmark_shape.scale.y = 2 * landmark_radius;
    landmark_shape.scale.z = 0.5;
    landmark_shape.color.a = 1;
    landmark_shape.color.b = 1;
    return landmark_shape;
}
 
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "KF_Loc");
  ros::NodeHandle nodeHandle_sub;
  KF_Loc kf_loc;

  ros::Subscriber odom_sub = nodeHandle_sub.subscribe<nav_msgs::Odometry>("/odom", 3000, std::bind(&KF_Loc::odomCallback, &kf_loc, std::placeholders::_1));
  ros::Subscriber scan_sub = nodeHandle_sub.subscribe<sensor_msgs::LaserScan>("/scan", 500, std::bind(&KF_Loc::scanCallback, &kf_loc, std::placeholders::_1));
  ros::Subscriber cmd_vel_sub = nodeHandle_sub.subscribe<geometry_msgs::Twist>("/cmd_vel", 1000, std::bind(&KF_Loc::cmd_velCallback, &kf_loc, std::placeholders::_1));

  ros::spin();

  return 0;
}
