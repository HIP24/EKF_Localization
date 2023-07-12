#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

//Thanks to http://wiki.ros.org/navigation/Tutorials/SendingSimpleGoals
int main(int argc, char **argv)
{
  ros::init(argc, argv, "map_navigation");
  ros::NodeHandle nodeHandle;

  MoveBaseClient ac("move_base", true);

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

    if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    {
      ROS_INFO("YES! The turtlebot moved to assigned point %d", i+1);
      std::cout << "\n\r------------------------------------------------------------------------------------------------------" << std::endl;
    }
    else
      ROS_INFO("The turtlebot failed to move to assigned point %d for some reason", i+1);
  }

  return 0;
}
