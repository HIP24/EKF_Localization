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

class EKF_Loc {
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
    EKF_Loc() {
        delta_t = 0.1;
        sigma = 0.1;

        A << 1, 0,
             0, 1;

        B << delta_t, 0,
             0, delta_t;

        R = pow(sigma, 2) * Eigen::Matrix2d::Identity();

        mt << 0.5, 0.5;  // initialize with the initial pose

        // Initialize cov using covariance information from /odom message
        cov << 1e-05, 0,
                 0, 1e-05;

        // Inside EKF_Loc constructor
        C << 1, 0,
             0, 1;
        Q = pow(sigma, 2) * Eigen::Matrix2d::Identity();
    }
    

    std::pair<Eigen::Vector2d, Eigen::Matrix2d> predict(const Eigen::Vector2d& ut, const Eigen::Vector2d& mt_prev, const Eigen::Matrix2d& cov_prev) {
        Eigen::Vector2d mt = A * mt_prev + B * ut;  // prediction
        Eigen::Matrix2d cov = A * cov_prev * A.transpose() + R;  // update error covariance
        return std::make_pair(mt, cov);
    }


    void printCov() {
        std::cout << "Cov: " << std::endl << cov << std::endl;
    }

void correct(const std::vector<std::pair<double, double>>& z_t, const std::vector<std::string>& c_t) {
    if (!z_t.empty()) {

        for (int i = 0; i < z_t.size(); ++i) {
            // Determine the index of the landmark corresponding to the observed feature
            std::string j = c_t[i];

            // Calculate the vector Δ as the difference between the landmark position and the robot position
            Eigen::Vector2d delta;
            delta << landmarks[j].x - mt(0), landmarks[j].y - mt(1);

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
    Eigen::Vector2d getMt() const {
        return mt;
    }

    Eigen::Matrix2d getCov() const {
        return cov;
    }

    // Setter methods
    void setMt(const Eigen::Vector2d& newMt) {
        mt = newMt;
    }

    void setCov(const Eigen::Matrix2d& newCov) {
        cov = newCov;
    }
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

    //prediction
    Eigen::Vector2d newMt;
    Eigen::Matrix2d newCov;
    std::tie(newMt, newCov) = ekf.predict(u_t, ekf.getMt(), ekf.getCov());
    ekf.setMt(newMt);
    ekf.setCov(newCov);
    std::cout << "\n\rAfter prediction: " << std::endl;
    ekf.printCov();

    //correction
    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) {
      
    // Define a vector of correspondence variables
std::vector<std::string> c_t;
for (const auto& measurement : z_t) {
    // Find the landmark closest to the measurement
    double min_distance = std::numeric_limits<double>::max();
    std::string closest_landmark;
    for (const auto& landmark : landmarks) {
        double dx = measurement.first - landmark.second.x;
        double dy = measurement.second - landmark.second.y;
        double distance = sqrt(dx * dx + dy * dy);
        if (distance < min_distance) {
            min_distance = distance;
            closest_landmark = landmark.first;
        }
    }

    // Add the closest landmark to c_t
    c_t.push_back(closest_landmark);

    // Print the correspondence
    //std::cout << "Measurement: (" << measurement.first << ", " << measurement.second << ") -> Landmark: " << closest_landmark << std::endl;
}



  // Call the correct method with z_t and c_t as arguments
  ekf.correct(z_t, c_t);

    std::cout << "\n\rAfter correction: " << std::endl;
    ekf.printCov();
    }

    
    if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
      //ROS_INFO("Hooray, the base moved to point %d", i+1);
      std::cout << "\n\rHooray, the base moved to point " << i+1 << std::endl;
    else
      //ROS_INFO("The base failed to move to point %d for some reason", i+1);
      std::cout << "\n\rThe base failed to move to point "<< i+1 << "for some reason" << std::endl;


    ros::spinOnce();
  }

  return 0;
}


