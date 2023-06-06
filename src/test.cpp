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

// Global variable to hold the control input
Eigen::Vector2d u_t;

// Global variable to hold the sensor measurements
std::vector<std::pair<double, double>> z_t; // Each pair is <range, angle>


typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

// Global variables to keep track of whether a message has been printed in the callback functions
bool odomCallbackPrinted = false;
bool scanCallbackPrinted = false;


// odom callback function
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
  if (!odomCallbackPrinted) {
    //ROS_INFO("Received odom: (%f, %f, %f)", msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    std::cout << "Received odom: (" << msg->pose.pose.position.x << ", " << msg->pose.pose.position.y << ", " << msg->pose.pose.position.z << ")" << std::endl;
    odomCallbackPrinted = true;

    double v = msg->twist.twist.linear.x;  // Forward velocity
    double w = msg->twist.twist.angular.z; // Rotational velocity
    u_t << v, w;

  }
}

// scan callback function
void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
  if (!scanCallbackPrinted) {
    //ROS_INFO("Received scan: ranges size: %zu", msg->ranges.size());
    //std::cout << "Received scan: ranges size: " << msg->ranges.size() << std::endl;
    scanCallbackPrinted = true;
  }

z_t.clear();
    double current_angle = msg->angle_min;

    for (const auto& range : msg->ranges) {
        // if range is valid
        if (range > msg->range_min && range < msg->range_max) {
            z_t.emplace_back(range, current_angle);
        }
        current_angle += msg->angle_increment;
    }

}

/*
// Create a struct to hold the x and y coordinates of each landmark
struct Landmark {
    double x;
    double y;
};

// Create a map to hold all the landmarks
std::map<std::string, Landmark> landmarks = {
    {"one_one",   {-1.1, -1.1}},
    {"one_two",   {-1.1,  0}},
    {"one_three", {-1.1,  1.1}},
    {"two_one",   {0,    -1.1}},
    {"two_two",   {0,     0}},
    {"two_three", {0,     1.1}},
    {"three_one", {1.1,  -1.1}},
    {"three_two", {1.1,   0}},
    {"three_three",{1.1,  1.1}},
    {"head",      {3.5,   0}},
    {"left_hand", {1.8,   2.7}},
    {"right_hand",{1.8,  -2.7}},
    {"left_foot", {-1.8,  2.7}},
    {"right_foot",{-1.8, -2.7}}
};
*/

class EKF_Loc {
private:
    Eigen::Vector2d xt;
    Eigen::Matrix2d cov;

    Eigen::Matrix2d A;
    Eigen::Matrix2d B;
    Eigen::Matrix2d R;

    double delta_t; // time step
    double sigma;   // noise standard deviation
    
    Eigen::Matrix2d C;
    Eigen::Matrix2d Q;


public:
    EKF_Loc() {
        delta_t = 0.1;
        sigma = 0.1;

        A << 1, 0,
             0, 1;

        B << delta_t, 0,
             0, delta_t;

        R = pow(sigma, 2) * Eigen::Matrix2d::Identity();

        xt << 0.5, 0.5;  // initialize with the initial pose

        // Initialize cov using covariance information from /odom message
        cov << 1e-05, 0,
                 0, 1e-05;

        // Inside EKF_Loc constructor
        C << 1, 0,
             0, 1;
        Q = pow(sigma, 2) * Eigen::Matrix2d::Identity();
    }

    void predict(const Eigen::Vector2d& u_t) {
        xt = A * xt + B * u_t;  // prediction
        cov = A * cov * A.transpose() + R;  // update error covariance
    }

    void printCov() {
        std::cout << "Cov: " << std::endl << cov << std::endl;
    }

    void correct(const std::vector<std::pair<double, double>>& z_t) {
        // For simplicity, we'll use the first observation in z_t.
        // In a more realistic setting, you'd need to associate each observed landmark with the predicted landmark.
        if (!z_t.empty()) {
            Eigen::Vector2d z = Eigen::Vector2d(z_t[0].first, z_t[0].second);

            // Compute the Kalman gain
            Eigen::Matrix2d S = C * cov * C.transpose() + Q;
            Eigen::Matrix2d K = cov * C.transpose() * S.inverse();

            // Compute the posterior state estimate
            xt = xt + K * (z - C * xt);

            // Compute the posterior error covariance estimate
            cov = (Eigen::Matrix2d::Identity() - K * C) * cov;
        }
    }


    // TODO: Implement the correct method

};








int main(int argc, char** argv){
  ros::init(argc, argv, "map_navigation");
  ros::NodeHandle n;

  // Subscribers
  ros::Subscriber odom_sub = n.subscribe("/odom", 1, odomCallback);
  ros::Subscriber scan_sub = n.subscribe("/scan", 1, scanCallback);

  EKF_Loc ekf;

  MoveBaseClient ac("move_base", true);

  while(!ac.waitForServer(ros::Duration(5.0))){
    //ROS_INFO("Waiting for the move_base action server to come up");
    std::cout << "Waiting for the move_base action server to come up" << std::endl;
  }

  // Define an array of 4 points
  double points[4][2] = {{1.5, 0},{-4, 0},{1.5, -2},{1, 2}};
  //0.5,0.5 -> 2,0.5 -> -2,0.5 -> -0.5,-1.5 -> 0.5,0.5

  for (int i = 0; i < 4; i++) {
    // Reset the printed variable in the callback functions
    odomCallbackPrinted = false;
    scanCallbackPrinted = false;

    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.frame_id = "base_link";
    goal.target_pose.header.stamp = ros::Time::now();
    goal.target_pose.pose.position.x = points[i][0];
    goal.target_pose.pose.position.y = points[i][1];
    goal.target_pose.pose.orientation.w = 1;

    //ROS_INFO("Sending goal %d", i+1);
    std::cout << "\r\n\r\n\r\n";
    std::cout << "Sending goal " << i+1 << std::endl;
    ac.sendGoal(goal);
    ac.waitForResult();

    ekf.predict(u_t);
    ekf.printCov();
    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) {
    ekf.correct(z_t);
    std::cout << "After correction: " << std::endl;
    ekf.printCov();
    }

    
    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
      //ROS_INFO("Hooray, the base moved to point %d", i+1);
      std::cout << "Hooray, the base moved to point " << i+1 << std::endl;
    else
      //ROS_INFO("The base failed to move to point %d for some reason", i+1);
      std::cout << "The base failed to move to point "<< i+1 << "for some reason" << std::endl;


    ros::spinOnce();
  }

  return 0;
}