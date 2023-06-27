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
  Eigen::Vector2d u_t;
  std::vector<std::pair<double, double>> z_t; // Each pair is <range, angle>
  std::vector<std::string> c_t;

  Eigen::VectorXd mt;
  Eigen::MatrixXd cov;

  Eigen::MatrixXd A;
  Eigen::MatrixXd B;
  Eigen::MatrixXd R;
  Eigen::MatrixXd Q;

  double roll, pitch, yaw;
  double robot_x, robot_y, robot_quat_z, robot_quat_w;

  double theta = 0;
  double delta_t = 0.1; // time step
  double sigma_x = 0.1;
  double sigma_y = 0.1;
  double sigma_theta = 0.1;

  double sigma_x0 = 0.1;
  double sigma_y0 = 0.1;
  double sigma_theta0 = 0.1;

  laser_geometry::LaserProjection laserProjection;

  geometry_msgs::PoseStamped turtle_pose;
  visualization_msgs::Marker turtle_cov;
  visualization_msgs::Marker landmark_shape;

  tf::TransformListener transformListener;
  ros::Publisher point_cloud_pub;
  ros::Publisher landmark_shape_pub;
  ros::Publisher landmark_pose_pub;
  ros::Publisher turtle_pose_pub;
  ros::Publisher turtle_cov_pub;

   struct Landmark
  {
    double innerTH;
    double outerTH;
    double landX;
    double landY;
    int signature;
  };
  std::map<int, Landmark> landmark_map;

public:
  EKF_Loc()
  {
    // initialize with the initial pose [x, y, theta, v, omega]
    mt.resize(3);
    mt << -2, -0.5, 0;

    // Initialize initial covariance
    cov.resize(3, 3);
    cov << pow(sigma_x0, 2), 0, 0,
        0, pow(sigma_y0, 2), 0,
        0, 0, pow(sigma_theta0, 2);

    // initialize process noise
    R.resize(3, 3);
    R << pow(sigma_x, 2), 0, 0,
        0, pow(sigma_y, 2), 0,
        0, 0, pow(sigma_theta, 2);

    // initialize measurement noise
    Q.resize(3, 3);
    Q = Eigen::Matrix<double, 3, 3>::Identity();

    // initialize nodeHandle for ekf_pose
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

  // cmd_vel callback function
  void cmd_velCallback(const geometry_msgs::Twist::ConstPtr &commandMsg)
  {
    double v = commandMsg->linear.x;
    // Forward velocity
    double w = commandMsg->angular.z;
    // Rotational velocity
    u_t << v, w;
    predict();
    printMtCov();
    publishTurtlePose();
    publishTurtleCovarianceEllipse();
  }

  // odom callback function
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

  // scan callback function
  void scanCallback(const sensor_msgs::LaserScan::ConstPtr &scan)
  {
    visualization_msgs::MarkerArray landmark_shapes;
    geometry_msgs::PoseArray landmark_poses;
    landmark_poses.header.stamp = ros::Time::now();

    sensor_msgs::PointCloud2 cloud;
    laserProjection.transformLaserScanToPointCloud("base_scan", *scan, cloud, transformListener);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(cloud, *pcl_cloud);

    // Thanks to https://pointclouds.org/documentation/tutorials/planar_segmentation.html
    pcl::PointIndices inlierIndices;
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

    for (const auto &landmark_map_element : landmark_map)
    {
      const Landmark &landmark = landmark_map_element.second;
      seg.setRadiusLimits(landmark.innerTH, landmark.outerTH);
      seg.segment(inlierIndices, *coefficients);

      if (!inlierIndices.indices.size() == 0)
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

    // Publish the detected landmark poses as a single message
    landmark_pose_pub.publish(landmark_poses);
    landmark_poses.poses.clear();
    landmark_shape_pub.publish(landmark_shapes);
  }

  void printMtCov()
  {
    //std::cout << "Cov: " << std::endl << cov << std::endl;
    //std::cout << "Pose: " << std::endl << mt << std::endl;
    std::cout << "Robot's real position: (x = " << robot_x << ", y = " << robot_y << ")" << std::endl;
    std::cout << "Robot's estimated position: (x = " << mt(0) << ", y = " << mt(1) << ", theta = " << mt(2) <<")" << std::endl;
  }

  void correct(const std::vector<geometry_msgs::Pose>& landmarks)
  {
    //std::cout << "corrected" << std::endl;
    Eigen::Matrix<double, 3, 3> I = Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::VectorXd mt_E(3);
    Eigen::MatrixXd cov_E(3,3);

    mt_E << 0., 0., 0.;

    cov_E << 0., 0., 0.,
            0., 0., 0.,
            0., 0., 0.;

   
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
      Eigen::Vector3d z_fake_i;
      z_fake_i << sqrt(q),
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
      mt_E += K_i * (z_i - z_fake_i);
      cov_E += K_i * H_i;    
    }
     mt += mt_E;
     cov = (I - cov_E) * cov;
    }

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

    mt(0) += u_t[0] * delta_t * cos(theta);
    mt(1) += u_t[0] * delta_t * sin(theta);
    mt(2) += u_t[1] * delta_t;
    mt(2) = std::atan2(std::sin(mt(2)), std::cos(mt(2)));

    B.resize(3, 2);
    B << u_t[0] * delta_t * cos(theta), 0,
        u_t[0] * delta_t * sin(theta), 0,
        0, u_t[1] * delta_t;

    // mt = A * mt_prev + B * ut;              // prediction
    cov = A * cov * A.transpose() + R; // update error covariance
   
  }

  void publishTurtlePose()
  {
    // Set the frame ID and timestamp
    turtle_pose.header.frame_id = "map";
    turtle_pose.header.stamp = ros::Time::now();

    // Set the pose data
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

    // Publish the pose
    turtle_pose_pub.publish(turtle_pose);
  }

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

    // Create a turtle_cov message to represent the ellipse
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
    double yaw = mt(2); // your yaw angle in radians

    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw);

    geometry_msgs::Quaternion quat_msg;
    quat_msg.x = q.x();
    quat_msg.y = q.y();
    quat_msg.z = q.z();
    quat_msg.w = q.w();

    turtle_cov.pose.orientation = quat_msg;

    turtle_cov.scale.x = a * 2;
    turtle_cov.scale.y = b * 2;
    turtle_cov.scale.z = 0.1;
    turtle_cov.color.a = 0.5;
    turtle_cov.color.r = 1.0;
    turtle_cov.color.g = 0.0;
    turtle_cov.color.b = 0.0;

    // Publish the turtle_cov message
    turtle_cov_pub.publish(turtle_cov);
  }

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
