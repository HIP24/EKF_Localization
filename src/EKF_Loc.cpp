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
    // correction
    // Define a vector of correspondence variables
    std::vector<std::string> c_t;

    // Print the size of z_t
    std::cout << "\r\nz_t size: " << z_t.size() << "\r\n" << std::endl;

    if (!z_t.empty())
    {
      // Define a map to keep track of the number of times each landmark was chosen
      std::map<std::string, int> landmark_counts;

      // Set the maximum distance threshold for the closest landmark
      double max_distance = 10.0;

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

        // Check if the minimum distance is below the maximum distance threshold
        if (min_distance < max_distance)
        {
          // Add the closest landmark to c_t
          c_t.push_back(closest_landmark);

          // Increment the counter for the closest landmark
          ++landmark_counts[closest_landmark];

          // Print the correspondence
          //std::cout << "Measurement: (" << measurement.first << ", " << measurement.second << ") -> Landmark: " << closest_landmark << ", Distance: " << min_distance << std::endl;
        }
      }

        // Print the number of times each landmark was chosen
        for (const auto &landmark_count : landmark_counts)
        {
          std::cout << "Landmark: " << landmark_count.first << " -> Count: " << landmark_count.second << std::endl;
        }

      // Set the minimum count threshold for each landmark
      int min_count = 200;

      // Find the landmark with the highest count above the minimum count threshold
      int max_count = 0;
      std::string best_landmark;
      for (const auto &landmark_count : landmark_counts)
      {
        if (landmark_count.second > max_count && landmark_count.second > min_count)
        {
          max_count = landmark_count.second;
          best_landmark = landmark_count.first;
        }
      }

        
      // Check if a best landmark was found
      if (!best_landmark.empty())
      {
        // Keep only the measurements associated with the best landmark in z_t
        auto it_z = z_t.begin();
        auto it_c = c_t.begin();
        while (it_z != z_t.end() && it_c != c_t.end())
        {
          if (*it_c != best_landmark)
          {
            it_z = z_t.erase(it_z);
            it_c = c_t.erase(it_c);
          }
          else
          {
            ++it_z;
            ++it_c;
          }
        }

        // Keep only the best landmark in c_t
        c_t.erase(std::remove_if(c_t.begin(), c_t.end(), [&](const std::string &landmark) { return landmark != best_landmark; }), c_t.end());

        std::cout << "filtered z_t size: " << z_t.size() << std::endl;
        std::cout << "filtered c_t size: " << c_t.size() << std::endl;

        if(c_t.size() !=0 && c_t.size() == z_t.size()){
          
        // Call the correct method with z_t and c_t as arguments
        correct(z_t, c_t);
        std::cout << "\n\r############## After correction ##############" << std::endl;
        printMtCov();
        }
        else  std::cout << "Landmark detected but distance too high" << std::endl;
      }
    }
std::cout << "---------------------------------------------" << std::endl;
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
      Eigen::Matrix<double, 2, 3> H;
      H << -delta(0) / sqrt(q), -delta(1) / sqrt(q), 0,
            delta(1) / q, -delta(0) / q, 0;

      // Calculate the Kalman gain Kit based on Hit, Qt, and Σt
      Eigen::Matrix<double, 3, 2> K = cov * H.transpose() * (H * cov * H.transpose() + Q).inverse();

      // Compute the innovation
      Eigen::Vector2d z(z_t[i].first, z_t[i].second);
      Eigen::Vector2d innovation = z - expected_z;

      // Compute the posterior state
      mt = mt + K * innovation;
      cov = (Eigen::MatrixXd::Identity(3, 3) - K * H) * cov;
    }
  }
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











