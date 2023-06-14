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
    {"head", {2, 0, {0, 1, 0}}},          // Green landmark
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
  double roll, pitch, yaw;
  double theta = 0; 
  double delta_t = 0.1; // time step
  double sigma = 0.1;   // noise standard deviation
  double sigma_x = 0.1; 
  double sigma_y = 0.1; 
  double sigma_theta = 0.1;
  double sigma_v = 0.1; 
  double sigma_omega = 0.1;
  

public:
  EKF_Loc()
  {
    // initialize with the initial pose [x, y, theta, v, omega]
    mt.resize(5);
    mt << 0.5, 0.5, 0, 0, 0; 

    // Initialize initial covariance
    cov.resize(5, 5);
    cov << 1e-5, 0, 0, 0, 0,
           0, 1e-5, 0, 0, 0,
           0, 0, 1e-5, 0 ,0,
           0 ,0 ,0 ,1e-5 ,0,
           0 ,0 ,0 ,0 ,1e-5;

    // initialize process noise
    R.resize(5, 5);
    R << pow(sigma_x, 2), 0, 0, 0, 0,
         0, pow(sigma_y, 2), 0, 0, 0,
         0, 0, pow(sigma_theta, 2), 0 ,0,
         0 ,0 ,0 ,pow(sigma_v, 2) ,0,
         0 ,0 ,0 ,0 ,pow(sigma_omega, 2);

  // initialize measurement noise
    Q = pow(sigma, 2) * Eigen::Matrix2d::Identity();

  // initialize nodeHandle for ekf_pose
    ros::NodeHandle nodeHandle;
    posePub = nodeHandle.advertise<geometry_msgs::PoseStamped>("/ekf_pose", 10);

  }


  // imu callback function
void cmd_velCallback(const geometry_msgs::Twist::ConstPtr& commandMsg)
{
    ++cmd_velPrintCount;
    double v = commandMsg->linear.x;;  // Forward velocity
    double w = commandMsg->angular.z;; // Rotational velocity
    u_t << v, w;
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
  }

  // scan callback function
  void scanCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
  {
    ++scanPrintCount;
  z_t.clear();
  double current_angle = msg->angle_min;

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

  for (const auto &range : msg->ranges)
  {
    // if range is valid
    if (range > msg->range_min && range < msg->range_max)
    {
      // Convert polar coordinates to Cartesian coordinates
      double x = range * cos(current_angle);
      double y = range * sin(current_angle);

      // Convert points from robot's frame to map frame using rotation matrix
      double map_x = robot_x + cos(yaw)*x - sin(yaw)*y;
      double map_y = robot_y + sin(yaw)*x + cos(yaw)*y;

      z_t.emplace_back(map_x, map_y);
    }
    current_angle += msg->angle_increment;
  }


  }

  std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(const Eigen::Vector2d &ut, const Eigen::VectorXd &mt_prev, const Eigen::MatrixXd &cov_prev)
  {
    std::cout << "Theta:\r\n " << theta << std::endl;
    std::cout << "v:\r\n " << u_t(0) << std::endl;
    std::cout << "w:\r\n " << u_t(1) << std::endl;
    std::cout << "A:\r\n " << A << std::endl;
    std::cout << "B:\r\n " << B << std::endl;

A.resize(5, 5);
A << 1, 0, 0, delta_t * cos(theta) * u_t(0) - delta_t * sin(theta) * u_t(1), 0,
     0, 1, 0, delta_t * sin(theta) * u_t(0) + delta_t * cos(theta) * u_t(1), 0,
     0, 0, 1,         0,            delta_t,
     0, 0, 0,         1,            0,
     0, 0, 0,         0,            1;


B.resize(5, 2);
B << 0, 0,
     0, 0,
     0, 0,
     0, 0,
     0, delta_t;



    mt = A * mt_prev + B * ut;              // prediction
    cov = A * cov_prev * A.transpose() + R; // update error covariance
    return std::make_pair(mt, cov);
  }

void correct(const std::vector<std::pair<double, double>> &z_t, const std::vector<std::string> &c_t)
{
  if (!z_t.empty())
  {
    for (int i = 0; i < z_t.size(); ++i)
    {
      // Determine the index of the landmark corresponding to the observed feature
      std::string j = c_t[i];

      // Calculate the vector Δ as the difference between the landmark position and the robot position
      Eigen::Vector2d delta;
      delta << landmarks[j].landX - mt(0), landmarks[j].landY - mt(1);

      // Calculate the variable q as the squared magnitude of Δ
      double q = delta.squaredNorm();

      // Calculate the expected measurement ^zit based on Δ and q
      Eigen::Vector2d expected_z;
      expected_z << sqrt(q), atan2(delta(1), delta(0));

      // Calculate the Jacobian matrix Hit based on Δ and q
      Eigen::Matrix<double, 2, 5> H;
      H << -delta(0) / sqrt(q), -delta(1) / sqrt(q), 0, 0, 0,
            delta(1) / q, -delta(0) / q, 0, 0, 0;

      // Calculate the Kalman gain Kit based on Hit, Qt, and Σt
      Eigen::Matrix<double, 5, 2> K = cov * H.transpose() * (H * cov * H.transpose() + Q).inverse();

      // Compute the innovation
      Eigen::Vector2d z(z_t[i].first, z_t[i].second);
      Eigen::Vector2d innovation = z - expected_z;

      // Compute the posterior state
      mt = mt + K * innovation;
      cov = (Eigen::MatrixXd::Identity(5, 5) - K * H) * cov;
    }
  }
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
    return mt;
  }

  Eigen::MatrixXd getCov() const
  {
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

  void poseEstimation(EKF_Loc &ekf)
  {
    // prediction
    Eigen::VectorXd newMt;
    Eigen::MatrixXd newCov;
    std::tie(newMt, newCov) = ekf.predict(u_t, ekf.getMt(), ekf.getCov());
    ekf.setMt(newMt);
    ekf.setCov(newCov);
    std::cout << "\n\r############## After prediction ##############" << std::endl;
    ekf.printMtCov();

    // correction
      // Define a vector of correspondence variables
      std::vector<std::string> c_t;

      // Print the size of z_t
      std::cout << "\r\nz_t size: " << z_t.size() << "\r\n" << std::endl;
 
      if (!z_t.empty())
      {
        // Define a map to keep track of the number of times each landmark was chosen
        std::map<std::string, int> landmark_counts;

        for (const auto &measurement : z_t)
        {
          // Find the landmark closest to the measurement
          double min_distance = std::numeric_limits<double>::max();
          std::string closest_landmark;
          for (const auto &landmark : landmarks)
          {
            double dx = measurement.first - landmark.second.landX;
            double dy = measurement.second - landmark.second.landY;
            double distance = sqrt(dx * dx + dy * dy);
            if (distance < min_distance)
            {
              min_distance = distance;
              closest_landmark = landmark.first;
            }
          }

          // Add the closest landmark to c_t
          c_t.push_back(closest_landmark);

          // Increment the counter for the closest landmark
          ++landmark_counts[closest_landmark];

          // Print the correspondence
          //std::cout << "Measurement: (" << measurement.first << ", " << measurement.second << ") -> Landmark: " << closest_landmark << ", Distance: " << min_distance << std::endl;
        }

        // Print the number of times each landmark was chosen
        for (const auto &landmark_count : landmark_counts)
        {
          //std::cout << "Landmark: " << landmark_count.first << " -> Count: " << landmark_count.second << std::endl;
        }

        // Call the correct method with z_t and c_t as arguments
        ekf.correct(z_t, c_t);
        std::cout << "\n\r############## After correction ##############" << std::endl;
        ekf.printMtCov();
        publishPose();
      }

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
  tf2::Quaternion quat;
  quat.setRPY(0, 0, yaw); // replace 0.0 with the yaw of your robot
  poseMsg.pose.orientation = tf2::toMsg(quat);


  // Publish the pose
  posePub.publish(poseMsg);
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


  // Create a ros::Rate object with a rate of 1 Hz
  ros::Rate rate(1);

  // Run the EKF update loop
  while (ros::ok())
  {
    ekf.poseEstimation(ekf);
    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}











