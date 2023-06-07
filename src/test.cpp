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

// Global variable to hold the control input
Eigen::Vector2d u_t;

// Global variable to hold the sensor measurements
std::vector<std::pair<double, double>> z_t; // Each pair is <range, angle>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

// Global variables to keep track of whether a message has been printed in the callback functions
//bool odomCallbackPrinted = true;
//bool scanCallbackPrinted = true;

// Define counters
int odomPrintCount = 0;
int scanPrintCount = 0;


double robot_x, robot_y;

// odom callback function
void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
  //if (!odomCallbackPrinted)
  //{
    // ROS_INFO("Received odom: (%f, %f, %f)", msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    //std::cout << "Received odom: (" << msg->pose.pose.position.x << ", " << msg->pose.pose.position.y << ", " << msg->pose.pose.position.z << ")" << std::endl;
    //std::cout << "First Odom: " << *msg << std::endl;
    ++odomPrintCount;
    //odomCallbackPrinted = true;

    double v = msg->twist.twist.linear.x;  // Forward velocity
    double w = msg->twist.twist.angular.z; // Rotational velocity
    u_t << v, w;


    // Update the robot's x and y coordinates
    robot_x = msg->pose.pose.position.x;
    robot_y = msg->pose.pose.position.y;
  //}
}

// scan callback function
void scanCallback(const sensor_msgs::LaserScan::ConstPtr &msg)
{

  //if (!scanCallbackPrinted)
  //{
    // ROS_INFO("Received scan: ranges size: %zu", msg->ranges.size());
    //std::cout << "Received scan: ranges size: " << msg->ranges.size() << std::endl;
    ++scanPrintCount;
    //scanCallbackPrinted = true;
  //}

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

// Create a struct to hold the x and y coordinates of each landmark
struct Landmark
{
  double landX;
  double landY;
};

// Create a map to hold all the landmarks
std::map<std::string, Landmark> landmarks = {
    {"head", {2, 0}},
    {"left_wall", {0, 2.4}},
    {"right_wall", {0, -2.4}},
    {"bottom", {-2.5, 0}}};

class EKF_Loc
{
private:
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

  std::pair<Eigen::Vector2d, Eigen::Matrix2d> predict(const Eigen::Vector2d &ut, const Eigen::Vector2d &mt_prev, const Eigen::Matrix2d &cov_prev)
  {
    Eigen::Vector2d mt = A * mt_prev + B * ut;              // prediction
    Eigen::Matrix2d cov = A * cov_prev * A.transpose() + R; // update error covariance
    return std::make_pair(mt, cov);
  }

  void printCov()
  {
    std::cout << "Cov: " << std::endl
              << cov << std::endl;
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
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "map_navigation");
  ros::NodeHandle nodeHandle;

  // Subscribers
  ros::Subscriber odom_sub = nodeHandle.subscribe("/odom", 1000, odomCallback);
  ros::Subscriber scan_sub = nodeHandle.subscribe("/scan", 1000, scanCallback);

  EKF_Loc ekf;

  MoveBaseClient ac("move_base", true);

  while (!ac.waitForServer(ros::Duration(5.0)))
  {
    // ROS_INFO("Waiting for the move_base action server to come up");
    std::cout << "Waiting for the move_base action server to come up" << std::endl;
  }

  // Define an array of 4 points
  //double points[4][2] = {{1.5, 0}, {-4, 0}, {1.5, -2}, {1, 2}};
  double points[4][2] = {{-3, -0.5}, {2.5, -2}, {2.0, 2.0}, {-2.0, 2.0}};
  // (0.5 | 0.5) -> (2 | 0.5) -> (-2 | 0.5) -> (-0.5 | -1.5) -> (0.5 | 0.5)

  for (int i = 0; i < 4; i++)
  {


    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.frame_id = "base_link";
    goal.target_pose.header.stamp = ros::Time::now();

    goal.target_pose.pose.position.x = points[i][0];
    goal.target_pose.pose.position.y = points[i][1];
    goal.target_pose.pose.orientation.w = 1;

    // ROS_INFO("Sending goal %d", i+1);
    std::cout << "\r\n\r\n\r\n";
    std::cout << "Sending goal " << i + 1 << std::endl;
    //std::cout << "----------------------------------------------------------------Entering sendGoal" << std::endl;
    ac.sendGoal(goal);
    //std::cout << "----------------------------------------------------------------Exiting sendGoal" << std::endl;

    //std::cout << "----------------------------------------------------------------Entering waitForResult" << std::endl;
    ac.waitForResult();
    //std::cout << "----------------------------------------------------------------Exiting waitForResult" << std::endl;

    //std::cout << "----------------------------------------------------------------Entering spinOnce" << std::endl;
    // Reset the printed variable in the callback functions
    //odomCallbackPrinted = false;
    //scanCallbackPrinted = false;
    ros::spinOnce();
    //std::cout << "----------------------------------------------------------------Exiting spinOnce" << std::endl;
    
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);

    geometry_msgs::TransformStamped transformStamped;
    try {
      if (tfBuffer.canTransform("map", "base_link", ros::Time(0), ros::Duration(3.0))) {
        transformStamped = tfBuffer.lookupTransform("map", "base_link", ros::Time(0));
      } else {
        ROS_WARN("Transformation not possible");
      }
    } catch (tf2::TransformException &ex) {
      ROS_WARN("%s", ex.what());
      ros::Duration(1.0).sleep();
      // continue processing...
    }

    double robot_x = transformStamped.transform.translation.x;
    double robot_y = transformStamped.transform.translation.y;

    std::cout << "Robot's position: (" << robot_x << ", " << robot_y << ")" << std::endl;

    // prediction
    Eigen::Vector2d newMt;
    Eigen::Matrix2d newCov;
    std::tie(newMt, newCov) = ekf.predict(u_t, ekf.getMt(), ekf.getCov());
    ekf.setMt(newMt);
    ekf.setCov(newCov);
    std::cout << "\n\r############## After prediction ##############" << std::endl;
    ekf.printCov();

    // correction
    if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    {

      // Define a vector of correspondence variables
      std::vector<std::string> c_t;
      // Print the counts
      std::cout << std::endl;
      std::cout << "odomCallback was called " << odomPrintCount << " times." << std::endl;
      odomPrintCount = 0;
      std::cout << "scanCallback was called " << scanPrintCount << " times." << std::endl;
      scanPrintCount = 0;
      // Print the size of z_t
      std::cout << "\r\nz_t size: " << z_t.size() << "\r\n"
                << std::endl;

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
        ekf.printCov();
      }
    }

    if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){
      // ROS_INFO("Hooray, the base moved to point %d", i+1);
      std::cout << "\n\rHooray, the base moved to point " << i + 1 << std::endl;
      std::cout << "\n\r------------------------------------------------------------------------------------------------------" << std::endl;
   } else
      // ROS_INFO("The base failed to move to point %d for some reason", i+1);
      std::cout << "\n\rThe base failed to move to point " << i + 1 << "for some reason" << std::endl;
  }

  return 0;
}
