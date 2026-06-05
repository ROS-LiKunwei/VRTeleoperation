一：启动机械臂
1.进入机械臂目录：/home/jialimeng/monkey_king
2.启动底层（自动归零）：ros2 launch sysmo_bringup sysmo_upbody_start.launch.py > debug/limeng.txt

3.给底层发布位置消息（）：
    （1）创建发布方：publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/sysmo_left_arm_controller/commands", 60);

    （2）设置消息类型：        
        // 创建并设置 Float64MultiArray 消息
        auto position_command_msg = std_msgs::msg::Float64MultiArray();

    （3）赋予消息信息：
        // 根据 current_data_index_ 选择 data
        position_command_msg.data = {
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //左臂关节弧度
            0.0, 0.0, 0.0, 0.0, -0.0, 0.0,//右臂关解弧度
            0.0,  //速度位：0.0表示底层5次多项式插值，速度适中随便用；    4.0是上层插值，底层速度很快，用的话上层一定要插值好
            0.0, 0.0, 0.0, 0.0,//这四位不要管，默认用0.0就行
            0.0//脖子关节弧度
        };

    （4）发布： publisher_->publish(cmd_msg_step);


二：启动灵巧手
1.进入灵巧手目录：/home/jialimeng/merger_over/linkerhand-python-sdk-main
2.启动灵巧手：
  （1）cd example/O6/gesture/
  （2）python3 linker_hand_loop_O6_0427.py
3.给灵巧手发布位置消息（）：
    （1）创建发布方：    
        left_hand_publisher_ = this->create_publisher<std_msgs::msg::Int32>("/left_topic_to_hand", 10);
        right_hand_publisher_ = this->create_publisher<std_msgs::msg::Int32>("/right_topic_to_hand", 10);

    （2）设置消息类型：        
        std_msgs::msg::Int32 msg_hand_left;
        std_msgs::msg::Int32 msg_hand_right;

    （3）赋予消息信息：
        msg_hand_left.data = 1;//左手松
        msg_hand_right.data = 1;//右手松
        msg_hand_left.data = 2;//左手抓瓶子
        msg_hand_right.data = 2;//右手抓瓶子

    （4）发布： 
        left_hand_publisher_->publish(msg_hand_left);
        right_hand_publisher_->publish(msg_hand_right);

   





