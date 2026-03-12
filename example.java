// 下行：服务器 -> 客户端；上行：客户端 -> 服务器

// 下行：比赛全局状态  (发送频率 5Hz)
message GameStatus {
    // current_stage 枚举值: 0:未开始比赛, 1:准备阶段, 2:十五秒裁判系统自检阶段, 
    // 3:五秒倒计时, 4:比赛中, 5:比赛结算中
    uint32 current_round = 1;      // 当前局号(从1开始)
    uint32 total_rounds = 2;       // 总局数
    uint32 red_score = 3;          // 红方得分
    uint32 blue_score = 4;         // 蓝方得分
    uint32 current_stage = 5;      // 当前阶段
    int32 stage_countdown_sec = 6; // 当前阶段剩余时间
    int32 stage_elapsed_sec = 7;   // 当前阶段已过时间(秒)
    bool is_paused = 8;            // 是否暂停
}

// 下行：机器人实时数据 (发送频率 10Hz)
message RobotDynamicStatus {
    uint32 current_health = 1;         // 当前血量
    float  current_heat = 2;           // 当前发射热量 （太高会导致扣血）
    uint32 remaining_ammo = 3;         // 剩余允许发弹量
    uint32 current_buffer_energy = 4;  // 当前缓冲能量
    bool   is_out_of_combat = 5;       // 是否脱战
    uint32 current_experience = 6;     // 当前经验值
    uint32 experience_for_upgrade = 7; // 距离下一次升级仍需获得的经验
}

// 上行：传输鼠标输入 (发送频率 75Hz)
message MouseControl {
    int32 mouse_x = 1;          // 鼠标 x 轴移动速度, 负值标识向左移动
    int32 mouse_y = 2;          // 鼠标 y 轴移动速度, 负值标识向下移动
    int32 mouse_z = 3;          // 鼠标滚轮移动速度, 负值标识向后滚动
    bool left_button_down = 4;  // 左键是否按下 (false=抬起, true=按下)
    bool right_button_down = 5; // 右键是否按下 (false=抬起, true=按下)
    bool mid_button_down = 6;   // 中键是否按下 (false=抬起, true=按下)
}

// 上行：地面机器人选择性能体系 (发送频率 1Hz)
message RobotPerformanceSelectionCommand {
    uint32 shooter = 1; // 发射机构性能体系，枚举值：1:冷却优先, 2:爆发优先
    uint32 chassis = 2; // 底盘性能体系，枚举值：1：血量优先, 2：功率优先
}