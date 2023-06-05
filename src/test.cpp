#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>

#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>

#include <sensor_msgs/LaserScan.h>
#include <vector>

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
    //odomCallbackPrinted = true;

    double v = msg->twist.twist.linear.x;  // Forward velocity
    double w = msg->twist.twist.angular.z; // Rotational velocity
    u_t << v, w;

  }
}




// scan callback function
void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
  //if (!scanCallbackPrinted) {
  //  ROS_INFO("Received scan: ranges size: %zu", msg->ranges.size());
  //  scanCallbackPrinted = true;
  //}

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






class EKF_Loc {
private:
    Eigen::Vector2d mu;
    Eigen::Matrix2d Sigma;

    Eigen::Matrix2d A;
    Eigen::Matrix2d B;
    Eigen::Matrix2d R;

    double delta_t; // time step
    double sigma;   // noise standard deviation

public:
    EKF_Loc() {
        delta_t = 0.1;
        sigma = 0.1;

        A << 1, 0,
             0, 1;

        B << delta_t, 0,
             0, delta_t;

        R = pow(sigma, 2) * Eigen::Matrix2d::Identity();

        mu << 0.5, 0.5;  // initialize with the initial pose
        Sigma = Eigen::Matrix2d::Identity();
    }

    void predict(const Eigen::Vector2d& u_t) {
        mu = A * mu + B * u_t;  // prediction
        Sigma = A * Sigma * A.transpose() + R;  // update error covariance
    }

    void printSigma() {
        std::cout << "Sigma: " << std::endl << Sigma << std::endl;
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
    ROS_INFO("Waiting for the move_base action server to come up");
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

    ROS_INFO("Sending goal %d", i+1);
    ac.sendGoal(goal);
    ac.waitForResult();

    ekf.predict(u_t);
    ekf.printSigma();
    
    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
      ROS_INFO("Hooray, the base moved to point %d", i+1);
    else
      ROS_INFO("The base failed to move to point %d for some reason", i+1);



    ros::spinOnce();
  }

  return 0;
}