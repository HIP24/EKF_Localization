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

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

// Define counters
int odomPrintCount = 0;
int scanPrintCount = 0;
double robot_x, robot_y;


// Create a struct to hold the x and y coordinates of each landmark
struct Landmark
{
  double landX;
  double landY;
};

// Create a map to hold all the landmarks
std::map<std::string, Landmark> landmarks = {
    {"bottom", {-2.5, 0}},
    {"head", {2, 0}},
    {"left_wall", {0, 2.4}},
    {"right_wall", {0, -2.4}}
    };

class EKF_Loc
{
private:
  // Global variable to hold the control input
  Eigen::Vector2d u_t;
  // Global variable to hold the sensor measurements
  std::vector<std::pair<double, double>> z_t; // Each pair is <range, angle>
  Eigen::Vector2d mt;
  Eigen::Matrix2d cov;

  Eigen::Matrix2d A;
  Eigen::Matrix2d B;
  Eigen::Matrix2d R;

  double delta_t; // time step
  double sigma;   // noise standard deviation

  Eigen::Matrix2d C;
  Eigen::Matrix2d Q;

public:
  EKF_Loc()
  {
    delta_t = 0.1;
    sigma = 0.1;

    A << 1, 0,
        0, 1;

    B << delta_t, 0,
        0, delta_t;

    R = pow(sigma, 2) * Eigen::Matrix2d::Identity();

    mt << 0.5, 0.5; // initialize with the initial pose

    // Initialize cov using covariance information from /odom message
    cov << 1e-05, 0,
        0, 1e-05;

    // Inside EKF_Loc constructor
    C << 1, 0,
        0, 1;
    Q = pow(sigma, 2) * Eigen::Matrix2d::Identity();
  }

// odom callback function
void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{   //std::cout << "odomPrintCount: " << odomPrintCount << std::endl;
    //std::cout << "Received odom: (" << *msg << ")" << std::endl;
    ++odomPrintCount;
    double v = msg->twist.twist.linear.x;  // Forward velocity
    double w = msg->twist.twist.angular.z; // Rotational velocity
    u_t << v, w;

    // Update the robot's x and y coordinates
    robot_x = msg->pose.pose.position.x;
    robot_y = msg->pose.pose.position.y;
    //std::cout << "Robot's position: (" << robot_x << ", " << robot_y << ")" << std::endl;
}

// scan callback function
void scanCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
{
  //std::cout << "scanPrintCount: " << scanPrintCount << std::endl;
  //std::cout << "Received scan: ranges size: " << *msg << std::endl;
  ++scanPrintCount;
  z_t.clear();
  double current_angle = msg->angle_min;

  for (const auto &range : msg->ranges)
  {
    // if range is valid
    if (range > msg->range_min && range < msg->range_max)
    {
      // Convert polar coordinates to Cartesian coordinates
      double x = range * cos(current_angle);
      double y = range * sin(current_angle);
      z_t.emplace_back(x, y);
    }
    current_angle += msg->angle_increment;
  }
}

  std::pair<Eigen::Vector2d, Eigen::Matrix2d> predict(const Eigen::Vector2d &ut, const Eigen::Vector2d &mt_prev, const Eigen::Matrix2d &cov_prev)
  {
    Eigen::Vector2d mt = A * mt_prev + B * ut;              // prediction
    Eigen::Matrix2d cov = A * cov_prev * A.transpose() + R; // update error covariance
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
        Eigen::Matrix<double, 2, 2> H;
        H << -delta(0) / sqrt(q), -delta(1) / sqrt(q),
            delta(1) / q, -delta(0) / q;

        // Calculate the Kalman gain Kit based on Hit, Qt, and Σt
        Eigen::Matrix2d K = cov * H.transpose() * (H * cov * H.transpose() + Q).inverse();

        // Compute the innovation
        Eigen::Vector2d z(z_t[i].first, z_t[i].second);
        Eigen::Vector2d innovation = z - expected_z;

        // Compute the posterior state
        mt = mt + K * innovation;
        cov = (Eigen::Matrix2d::Identity() - K * H) * cov;
      }
    }
  }

  void printMtCov()
  {
    std::cout << "Cov: " << std::endl << cov << std::endl;
    std::cout << "Pose -> x = " << mt(0) << ", y = " << mt(1) << std::endl;
  }


  // Getter methods
  Eigen::Vector2d getMt() const
  {
    return mt;
  }

  Eigen::Matrix2d getCov() const
  {
    return cov;
  }

  // Setter methods
  void setMt(const Eigen::Vector2d &newMt)
  {
    mt = newMt;
  }

  void setCov(const Eigen::Matrix2d &newCov)
  {
    cov = newCov;
  }



  void poseEstimation(EKF_Loc &ekf, MoveBaseClient &ac){
     // prediction
    Eigen::Vector2d newMt;
    Eigen::Matrix2d newCov;
    std::tie(newMt, newCov) = ekf.predict(u_t, ekf.getMt(), ekf.getCov());
    ekf.setMt(newMt);
    ekf.setCov(newCov);
    std::cout << "\n\r############## After prediction ##############" << std::endl;
    ekf.printMtCov();

    // correction
    if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    {

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
          std::cout << "Landmark: " << landmark_count.first << " -> Count: " << landmark_count.second << std::endl;
        }

        // Call the correct method with z_t and c_t as arguments
        ekf.correct(z_t, c_t);
        std::cout << "\n\r############## After correction ##############" << std::endl;
        ekf.printMtCov();
      }
    }
  }


};

int main(int argc, char **argv)
{
ros::init(argc, argv, "map_navigation");
ros::NodeHandle nodeHandle;

EKF_Loc ekf;
MoveBaseClient ac("move_base", true);

ros::Subscriber odom_sub = nodeHandle.subscribe<nav_msgs::Odometry>("/odom", 3000, std::bind(&EKF_Loc::odomCallback, &ekf, std::placeholders::_1));
ros::Subscriber scan_sub = nodeHandle.subscribe<sensor_msgs::LaserScan>("/scan", 500, std::bind(&EKF_Loc::scanCallback, &ekf, std::placeholders::_1));

while (!ac.waitForServer(ros::Duration(5.0)))
{
  std::cout << "Waiting for the move_base action server to come up" << std::endl;
}

double points[4][2];

// Retrieve the points from ROS parameters
XmlRpc::XmlRpcValue point1, point2, point3, point4;
if (nodeHandle.getParam("point1", point1) &&
    nodeHandle.getParam("point2", point2) &&
    nodeHandle.getParam("point3", point3) &&
    nodeHandle.getParam("point4", point4))
{
  points[0][0] = static_cast<double>(point1["x"]);
  points[0][1] = static_cast<double>(point1["y"]);
  points[1][0] = static_cast<double>(point2["x"]);
  points[1][1] = static_cast<double>(point2["y"]);
  points[2][0] = static_cast<double>(point3["x"]);
  points[2][1] = static_cast<double>(point3["y"]);
  points[3][0] = static_cast<double>(point4["x"]);
  points[3][1] = static_cast<double>(point4["y"]);
}
else
{
  // Handle the case when the parameters are not found or have incorrect types
  // ...
}

for (int i = 0; i < 4; i++)
{
  move_base_msgs::MoveBaseGoal goal;
  goal.target_pose.header.frame_id = "map";
  goal.target_pose.header.stamp = ros::Time::now();

  goal.target_pose.pose.position.x = points[i][0];
  goal.target_pose.pose.position.y = points[i][1];
  goal.target_pose.pose.orientation.w = 1;

  std::cout << "\r\n\r\n\r\n";
  std::cout << "Sending goal " << i + 1 << std::endl;
  ac.sendGoal(goal);
  ac.waitForResult();
  ros::spinOnce();

    // Print the counts
    std::cout << std::endl;
    std::cout << "odomCallback was called " << odomPrintCount << " times." << std::endl;
    odomPrintCount = 0;
    std::cout << "scanCallback was called " << scanPrintCount << " times." << std::endl;
    scanPrintCount = 0;
    std::cout << "Robot's position: (" << robot_x << ", " << robot_y << ")" << std::endl;

    // Do EKF_Localization with landmarks
    ekf.poseEstimation(ekf, ac);

    if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){
      // ROS_INFO("Hooray, the base moved to point %d", i+1);
      std::cout << "\n\rHooray, the base moved to point " << i + 1 << std::endl;
      std::cout << "\n\r------------------------------------------------------------------------------------------------------" << std::endl;
   } else
      // ROS_INFO("The base failed to move to point %d for some reason", i+1);
      std::cout << "\n\rThe base failed to move to point " << i + 1 << " for some reason" << std::endl;
  }

  return 0;
}
