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


class EKF_Loc
{
private:
  // Control vector 
  Eigen::Vector2d u_t;
  // Measurement vector 
  std::vector<std::pair<double, double>> z_t; 

  // State and covariance of the turtlebot
  Eigen::VectorXd mt;
  Eigen::MatrixXd cov;
  double sigma_x_cov = 0.1;
  double sigma_y_cov = 0.1;
  double sigma_theta_cov = 0.1;

  // Matrices for prediction and correction
  Eigen::MatrixXd A;
  Eigen::MatrixXd R;
  Eigen::MatrixXd Q;
  double sigma_x_R = 0.1;
  double sigma_y_R = 0.1;
  double sigma_theta_R = 0.1;
  double theta = 0;
  double delta_t = 0.1;

  // Position and orientation of the turtlebot
  double roll, pitch, yaw;
  double robot_x, robot_y, robot_quat_z, robot_quat_w;

  // Messages
  geometry_msgs::PoseStamped turtle_pose;
  visualization_msgs::Marker turtle_cov;
  visualization_msgs::Marker landmark_shape;

  // Publishers
  ros::Publisher point_cloud_pub;
  ros::Publisher landmark_shape_pub;
  ros::Publisher landmark_pose_pub;
  ros::Publisher turtle_pose_pub;
  ros::Publisher turtle_cov_pub;

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
  EKF_Loc()
  {
    // Initialize with the initial pose mt [x, y, theta]
    mt.resize(3);
    mt << -2, -0.5, 0;

    // Initialize initial covariance cov
    cov.resize(3, 3);
    cov << pow(sigma_x_cov, 2), 0, 0,
        0, pow(sigma_y_cov, 2), 0,
        0, 0, pow(sigma_theta_cov, 2);

    // Initialize process noise R
    R.resize(3, 3);
    R << pow(sigma_x_R, 2), 0, 0,
        0, pow(sigma_y_R, 2), 0,
        0, 0, pow(sigma_theta_R, 2);

    // Initialize measurement noise Q
    Q.resize(3, 3);
    Q = Eigen::Matrix<double, 3, 3>::Identity();

    // Initialize nodeHandle for ekf_pose
    ros::NodeHandle nodeHandle_adv;
    turtle_pose_pub = nodeHandle_adv.advertise<geometry_msgs::PoseStamped>("/turtle_pose", 10);
    turtle_cov_pub = nodeHandle_adv.advertise<visualization_msgs::Marker>("/turtle_cov", 10);
    point_cloud_pub = nodeHandle_adv.advertise<sensor_msgs::PointCloud2>("/point_cloud", 1000, false);
    landmark_shape_pub = nodeHandle_adv.advertise<visualization_msgs::MarkerArray>("landmark_shape", 10);
    landmark_pose_pub = nodeHandle_adv.advertise<geometry_msgs::PoseArray>("landmark_pose", 10);

    // Create two Landmark objects [innerTH, outerTH, landX, landY, signature]
    landmark_map[1] = {0.3, 0.4, 1.1, -1.1, 1};
    landmark_map[3] = {0.2, 0.3, -1.1, 1.1, 3};
  }

  /*This function is the callback for the /cmd_vel topic. It receives the linear 
  and angular velocity commands for the turtlebot and updates the control input (u_t). 
  It then calls the predict() function to perform the prediction step of the Extended 
  Kalman Filter (EKF), updates the turtle's pose, and publishes the pose and covariance.*/
  void cmd_velCallback(const geometry_msgs::Twist::ConstPtr &commandMsg)
  {
    // Forward velocity
    double v = commandMsg->linear.x;
    // Rotational velocity
    double w = commandMsg->angular.z;
    // Updae the control input
    u_t << v, w;

    predict();
    printMtCov();
    publishTurtlePose();
    publishTurtleCovarianceEllipse();
  }

  /*This function is the callback for the /odom topic. It receives the odometry data for the 
  turtlebot and extracts the robot's position and orientation. These values are extracted
  only to compare them to the estimated pose. */
  void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
  { 
    // Update the robot's x and y coordinates
    robot_x = msg->pose.pose.position.x;
    robot_y = msg->pose.pose.position.y;
    robot_quat_z = msg->pose.pose.orientation.z;
    robot_quat_w = msg->pose.pose.orientation.w;

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

  /*This function is the callback for the /scan topic. It receives laser scan data, converts it
   to a point cloud, and performs circle fitting using the RANSAC algorithm to detect landmarks. 
   For each detected landmark, it calculates the position in the map frame and calls the correct() 
   function to update the turtle's pose estimation. It also publishes the detected landmark poses 
   and shapes.*/
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
      printMtCov();
      publishTurtlePose();
      publishTurtleCovarianceEllipse();

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

  /*This function prints the turtle's estimated position (mt) and covariance (cov) to the console. 
  It also prints the real position of the robot (robot_x and robot_y) from odometry data.*/
  void printMtCov()
  {
    //std::cout << "Cov: " << std::endl << cov << std::endl;
    //std::cout << "Pose: " << std::endl << mt << std::endl;
    std::cout << "Robot's real position: (x = " << robot_x << ", y = " << robot_y << ")" << std::endl;
    std::cout << "Robot's estimated position: (x = " << mt(0) << ", y = " << mt(1) << ", theta = " << mt(2) <<")" << std::endl;
  }

  /*This function performs the correction step of the EKF. It takes a vector of detected landmark poses 
  as input and updates the turtle's pose estimation (mt) and covariance (cov) based on the measurement 
  information. It uses the difference between the observed and predicted landmark positions to calculate 
  the Kalman gain and update the pose estimation. */
  void correct(const std::vector<geometry_msgs::Pose>& landmarks)
  {
    Eigen::Matrix<double, 3, 3> I = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::VectorXd mt_E(3);
    Eigen::MatrixXd cov_E(3,3);

    // Sum for mt
    mt_E << 0., 0., 0.;

    // Sum for cov
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
      delta << landX_map - mt(0),
               landY_map - mt(1);
      double q = delta.transpose() * delta;
      Eigen::Vector3d z_exp_i;
      z_exp_i << sqrt(q),
                 atan2(delta(1), delta(0)) - mt(2),
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

      Eigen::Matrix3d K_i = cov * H_i.transpose() * (H_i * cov * H_i.transpose() + Q).inverse();
      mt_E += K_i * (z_i - z_exp_i);
      cov_E += K_i * H_i;    
    }
    // Update mt and cov
     mt += mt_E;
     cov = (I - cov_E) * cov;
    }

  /*This function performs the prediction step of the EKF Localization. It updates the turtle's pose estimation (mt) and 
  covariance (cov) based on the control input (u_t) and the system dynamics. It calculates the state transition
  matrix (A) and control input matrix (B) and updates the pose estimation and covariance using the motion model.*/
  void predict()
  {
    // std::cout << "Theta:\r\n " << theta << std::endl;
    // std::cout << "v:\r\n " << u_t(0) << std::endl;
    // std::cout << "w:\r\n " << u_t(1) << std::endl;
    ////std::cout << "A:\r\n " << A << std::endl;
    ////std::cout << "B:\r\n " << B << std::endl;

    A.resize(3, 3);
    A << 1, 0, -u_t[0] * delta_t * sin(theta),
        0, 1, u_t[0] * delta_t * cos(theta),
        0, 0, 1;

    // Update the estimated state
    mt(0) += u_t[0] * delta_t * cos(theta);
    mt(1) += u_t[0] * delta_t * sin(theta);
    mt(2) += u_t[1] * delta_t;
    mt(2) = std::atan2(std::sin(mt(2)), std::cos(mt(2)));

    // Update the covariance
    cov = A * cov * A.transpose() + R; // update error covariance
   
  }

  /*This function publishes the turtle's estimated pose (mt) as a geometry_msgs::PoseStamped message on 
  the /turtle_pose topic. It sets the position and orientation of the turtle's pose and publishes the message.*/
  void publishTurtlePose()
  {
    // Fill turtle_pose message to represent the pose
    turtle_pose.header.frame_id = "map";
    turtle_pose.header.stamp = ros::Time::now();
    turtle_pose.pose.position.x = mt(0);
    turtle_pose.pose.position.y = mt(1);

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

    turtle_pose.pose.orientation = quat_msg;

    // Publish the turtle_pose message
    turtle_pose_pub.publish(turtle_pose);
  }

  /*This function publishes the covariance ellipse of the turtle's pose as a visualization_msgs::Marker message on 
  the /turtle_cov topic. It computes the eigenvalues and eigenvectors of the covariance matrix and determines the 
  orientation and size of the ellipse. It then publishes the marker representing the covariance ellipse.*/
  void publishTurtleCovarianceEllipse()
  {
    // Compute the eigenvalues and eigenvectors of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov.block<2, 2>(0, 0));
    Eigen::Vector2d eigenvalues = solver.eigenvalues();
    Eigen::Matrix2d eigenvectors = solver.eigenvectors();

    // Compute the angle of rotation and the semi-major and semi-minor axes of the ellipse
    double angle = atan2(eigenvectors(1, 0), eigenvectors(0, 0));
    double a = sqrt(eigenvalues(0));
    double b = sqrt(eigenvalues(1));

    // Fill turtle_cov message to represent the ellipse
    turtle_cov.header.frame_id = "map";
    turtle_cov.header.stamp = ros::Time::now();
    turtle_cov.ns = "ellipse";
    turtle_cov.id = 0;
    turtle_cov.type = visualization_msgs::Marker::CYLINDER;
    turtle_cov.action = visualization_msgs::Marker::ADD;
    turtle_cov.pose.position.x = mt(0);
    turtle_cov.pose.position.y = mt(1);
    turtle_cov.pose.position.z = 0;

    double roll = 0;
    double pitch = 0;
    double yaw = mt(2); 

    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);

    geometry_msgs::Quaternion quat_msg;
    quat_msg.x = q.x();
    quat_msg.y = q.y();
    quat_msg.z = q.z();
    quat_msg.w = q.w();

    turtle_cov.pose.orientation = quat_msg;

    // Scale and colorize the landmark
    turtle_cov.scale.x = a * 2;
    turtle_cov.scale.y = b * 2;
    turtle_cov.scale.z = 0.01;
    turtle_cov.color.a = 0.5;
    turtle_cov.color.r = 1.0;
    turtle_cov.color.g = 0.0;
    turtle_cov.color.b = 0.0;

    // Publish the turtle_cov message
    turtle_cov_pub.publish(turtle_cov);
  }

  /*This function returns the pose of a landmark based on its distance from the robot 
  (robotXtoLM and robotYtoLM) and the provided Landmark object.*/
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

  /*This function publishes the pose of a landmark as a marker on the specified markerId topic. It uses the distances 
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
  ros::init(argc, argv, "ekf_loc");
  ros::NodeHandle nodeHandle_sub;
  EKF_Loc ekf;

  ros::Subscriber odom_sub = nodeHandle_sub.subscribe<nav_msgs::Odometry>("/odom", 3000, std::bind(&EKF_Loc::odomCallback, &ekf, std::placeholders::_1));
  ros::Subscriber scan_sub = nodeHandle_sub.subscribe<sensor_msgs::LaserScan>("/scan", 500, std::bind(&EKF_Loc::scanCallback, &ekf, std::placeholders::_1));
  ros::Subscriber cmd_vel_sub = nodeHandle_sub.subscribe<geometry_msgs::Twist>("/cmd_vel", 1000, std::bind(&EKF_Loc::cmd_velCallback, &ekf, std::placeholders::_1));

  ros::spin();

  return 0;
}
