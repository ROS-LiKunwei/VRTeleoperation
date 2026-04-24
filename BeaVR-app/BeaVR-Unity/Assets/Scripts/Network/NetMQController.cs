using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// ZMQ通信控制器，管理所有NetMQ sockets操作。
/// 负责初始化、套接字创建和清理。
/// </summary>
public class NetMQController : MonoBehaviour
{
    // 单例实例
    private static NetMQController _instance;
    public static NetMQController Instance 
    {
        get 
        {
            if (_instance == null)
            {
                GameObject go = new GameObject("NetMQController");
                _instance = go.AddComponent<NetMQController>();
                DontDestroyOnLoad(go); // 切换场景不销毁
            }
            return _instance;
        }
    }

    // sokets引用
    private Dictionary<string, PushSocket> sockets = new Dictionary<string, PushSocket>();
    private Dictionary<string, bool> socketConnectionStatus = new Dictionary<string, bool>();
    
    // 从JSON加载的网络设置
    private string ipAddress;
    private string rightKeypointPort;
    private string leftKeypointPort;
    private string resolutionPort;
    private string pausePort;
    
    // 初始化标志
    private bool netMQInitialized = false;
    
    // 套接字失败计数
    private Dictionary<string, int> socketFailCounts = new Dictionary<string, int>();

    // 转发频率统计
    private Dictionary<string, int> _sendCounts = new Dictionary<string, int>();
    private Dictionary<string, float> _lastSendFreqLogTime = new Dictionary<string, float>();
    private Dictionary<string, float> _sendFrequencies = new Dictionary<string, float>();
    private const float NETMQ_FREQ_CALC_INTERVAL = 1.0f;
    private float _lastForwardLogTime = 0f;
    private const float FORWARD_LOG_INTERVAL = 2.0f;
    private float _lastWristLogTime = 0f;
    private const float WRIST_LOG_INTERVAL = 2.0f;

    // 26个坐标系数据打印
    private float _lastFullJointLogTime = 0f;
    private const float FULL_JOINT_LOG_INTERVAL = 5.0f;
    
    // 帧索引，用于匹配三个环节的数据
    private int _frameIndex = 0;
    
    /// <summary>
    /// 初始化NetMQController
    /// </summary>
    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }
        
        _instance = this;
        DontDestroyOnLoad(gameObject);
        
        // 早期初始化NetMQ
        InitializeNetMQ();
        
        // 加载网络配置
        LoadNetworkConfig();
    }
    
    /// <summary>
    /// 从JSON文件加载网络配置
    /// </summary>
    private void LoadNetworkConfig()
    {
        try
        {
            // 从Resources文件夹加载JSON文件
            TextAsset configFile = Resources.Load<TextAsset>("Configurations/Network");
            if (configFile == null)
            {
                Debug.LogError("NetMQController: 加载Network.json失败");
                return;
            }
            
            // 解析JSON
            var configJson = JsonUtility.FromJson<NetworkSettings>(configFile.text);
            
            // 存储配置值（包括IP地址）
            ipAddress = configJson.IPAddress;
            rightKeypointPort = configJson.rightkeyptPortNum;
            leftKeypointPort = configJson.leftkeyptPortNum;
            resolutionPort = configJson.resolutionPortNum;
            pausePort = configJson.PausePortNum;
            
            Debug.Log($"NetMQController: 从JSON加载网络配置 - IP: {ipAddress}");
        }
        catch (Exception e)
        {
            Debug.LogError($"NetMQController: 加载网络配置错误 - {e.Message}");
        }
    }
    
    /// <summary>
    /// 初始化NetMQ系统
    /// </summary>
    public void InitializeNetMQ()
    {
        try
        {
            if (!netMQInitialized)
            {
                Debug.Log("NetMQController: 初始化NetMQ...");
                
                // 使用推荐的方法替代过时的ManualTerminationTakeOver
                // 这确保NetMQ为当前线程上下文正确初始化
                AsyncIO.ForceDotNet.Force();
                
                // 标记为已初始化
                netMQInitialized = true;
                Debug.Log("NetMQController: NetMQ初始化成功");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"NetMQController: 初始化NetMQ错误 - {e.GetType().Name}: {e.Message}");
            // 即使初始化失败，也不抛出异常，允许应用继续运行
            netMQInitialized = false;
        }
    }
    
    /// <summary>
    /// 创建具有给定名称和地址的sockets
    /// </summary>
    /// <param name="socketName">套接字名称</param>
    /// <param name="address">套接字地址</param>
    /// <returns>如果创建成功则返回true，否则返回false</returns>
    public bool CreateSocket(string socketName, string address)
    {
        try
        {
            if (sockets.ContainsKey(socketName))
            {
                // 具有此名称的套接字已存在
                Debug.LogWarning($"NetMQController: 套接字 '{socketName}' 已存在");
                return true;
            }
            
            // 检查NetMQ是否已初始化
            if (!netMQInitialized)
            {
                Debug.LogWarning($"NetMQController: NetMQ未初始化，正在尝试初始化...");
                InitializeNetMQ();
                if (!netMQInitialized)
                {
                    Debug.LogError($"NetMQController: NetMQ初始化失败，无法创建套接字 '{socketName}'");
                    socketConnectionStatus[socketName] = false;
                    return false;
                }
            }
            
            // 验证地址格式
            if (string.IsNullOrEmpty(address) || address == "tcp://:")
            {
                Debug.LogError($"NetMQController: 无效的地址格式: {address}");
                socketConnectionStatus[socketName] = false;
                return false;
            }
            
            // 创建新套接字
            Debug.Log($"NetMQController: 创建套接字 '{socketName}' 在 {address}");
            PushSocket socket = new PushSocket();
            socket.Options.SendHighWatermark = 1000;
            socket.Options.Linger = TimeSpan.FromMilliseconds(100);
            socket.Connect(address);
            
            // 存储套接字
            sockets[socketName] = socket;
            socketConnectionStatus[socketName] = true;
            
            Debug.Log($"NetMQController: 套接字 '{socketName}' 创建并连接到 {address}");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"NetMQController: 创建套接字 '{socketName}' 错误 - {e.GetType().Name}: {e.Message}");
            socketConnectionStatus[socketName] = false;
            // 确保不会存储空套接字
            if (sockets.ContainsKey(socketName))
            {
                sockets.Remove(socketName);
            }
            return false;
        }
    }
    
    /// <summary>
    /// 根据网络配置创建标准套接字
    /// </summary>
    public void CreateStandardSockets()
    {
        try
        {
            Debug.Log("NetMQController: 创建标准套接字...");
            
            // 优先使用JSON配置中的IP地址，如果没有则使用PlayerPrefs中的IP
            if (string.IsNullOrEmpty(ipAddress) || ipAddress == "undefined")
            {
                string prefsIP = PlayerPrefs.GetString("ServerIP", string.Empty);
                if (!string.IsNullOrEmpty(prefsIP))
                {
                    ipAddress = prefsIP;
                    Debug.Log($"NetMQController: 使用PlayerPrefs中的IP地址: {ipAddress}");
                }
            }
            else
            {
                Debug.Log($"NetMQController: 使用JSON配置中的IP地址: {ipAddress}");
            }

            // 检查IP是否不可用，跳过套接字创建
            if (string.IsNullOrEmpty(ipAddress) || ipAddress == "undefined")
            {
                Debug.LogWarning("NetMQController: IP地址未定义。必须手动建立连接。");
                return;
            }
            
            // 检查端口配置
            if (string.IsNullOrEmpty(rightKeypointPort) || string.IsNullOrEmpty(leftKeypointPort) ||
                string.IsNullOrEmpty(resolutionPort) || string.IsNullOrEmpty(pausePort))
            {
                Debug.LogError("NetMQController: 端口配置不完整");
                return;
            }
            
            // 创建右手套接字
            string rightHandAddress = $"tcp://{ipAddress}:{rightKeypointPort}";
            CreateSocket("RightHand", rightHandAddress);
            
            // 创建左手套接字
            string leftHandAddress = $"tcp://{ipAddress}:{leftKeypointPort}";
            CreateSocket("LeftHand", leftHandAddress);
            
            // 创建分辨率套接字
            string resolutionAddress = $"tcp://{ipAddress}:{resolutionPort}";
            CreateSocket("Resolution", resolutionAddress);
            
            // 创建暂停套接字
            string pauseAddress = $"tcp://{ipAddress}:{pausePort}";
            CreateSocket("Pause", pauseAddress);
            
            // 记录套接字状态
            LogSocketStatus();
        }
        catch (Exception e)
        {
            Debug.LogError($"NetMQController: 创建标准套接字错误 - {e.Message}");
            // 即使出现错误，也不抛出异常，允许应用继续运行
        }
    }
    
    /// <summary>
    /// 通过命名套接字发送消息，带有超时保护
    /// </summary>
    /// <param name="socketName">套接字名称</param>
    /// <param name="message">要发送的消息</param>
    /// <returns>如果发送成功则返回true，否则返回false</returns>
    public bool SendMessage(string socketName, string message)
    {
        try
        {
            if (!sockets.ContainsKey(socketName))
            {
                return false;
            }

            var socket = sockets[socketName];
            if (socket == null)
            {
                return false;
            }

            // 添加超时保护
            bool sent = socket.TrySendFrame(TimeSpan.FromMilliseconds(10), message);
            
            if (!sent)
            {
                // 如果发送超时，将此套接字标记为可能断开连接
                socketFailCounts[socketName] = socketFailCounts.GetValueOrDefault(socketName, 0) + 1;
                
                // 如果多次失败，尝试重新连接此套接字
                if (socketFailCounts[socketName] > 5)
                {
                    Debug.LogWarning($"套接字 {socketName} 多次失败。尝试重新连接...");
                    ReconnectSocket(socketName);
                    socketFailCounts[socketName] = 0;
                }
                return false;
            }
            
            // 成功时重置失败计数
            socketFailCounts[socketName] = 0;

            // 初始化频率统计字典
            if (!_sendCounts.ContainsKey(socketName))
            {
                _sendCounts[socketName] = 0;
                _lastSendFreqLogTime[socketName] = 0f;
                _sendFrequencies[socketName] = 0f;
            }

            // 转发频率统计
            _sendCounts[socketName]++;
            float currentTime = Time.time;
            if (currentTime - _lastSendFreqLogTime[socketName] >= NETMQ_FREQ_CALC_INTERVAL)
            {
                _sendFrequencies[socketName] = _sendCounts[socketName] / (currentTime - _lastSendFreqLogTime[socketName]);
                _sendCounts[socketName] = 0;
                _lastSendFreqLogTime[socketName] = currentTime;
                Debug.Log($"[App→Bot] {socketName} 转发频率: {_sendFrequencies[socketName]:F1} Hz");
            }

            // 定期打印转发位姿信息
            if (currentTime - _lastForwardLogTime >= FORWARD_LOG_INTERVAL)
            {
                _lastForwardLogTime = currentTime;
                string poseInfo = message.Length > 100 ? message.Substring(0, 100) + "..." : message;
                Debug.Log($"[App→Bot] index={_frameIndex} {socketName} 转发数据: {poseInfo}");
                _frameIndex++;
            }

            // 定期打印手腕部数据（只对Hand类型的消息）
            if ((socketName == "RightHand" || socketName == "LeftHand") &&
                currentTime - _lastWristLogTime >= WRIST_LOG_INTERVAL)
            {
                _lastWristLogTime = currentTime;
                string wristData = ParseWristData(message);
                Debug.Log($"[App→Bot] index={_frameIndex} {socketName} 手腕数据: {wristData}");
                _frameIndex++;
            }

            // 定期打印26个坐标系数据（只对Hand类型的消息）
            if ((socketName == "RightHand" || socketName == "LeftHand") &&
                currentTime - _lastFullJointLogTime >= FULL_JOINT_LOG_INTERVAL)
            {
                _lastFullJointLogTime = currentTime;
                string fullJointData = ParseFullJointData(message);
                Debug.Log($"[App→Bot] index={_frameIndex} {socketName} 26关节数据: {fullJointData}");
                _frameIndex++;
            }

            return true;
        }
        catch (Exception e) // 捕捉底层网络错误：如网络模块崩溃、端口被突然占用...
        {
            Debug.LogError($"NetMQController: 发送消息到 '{socketName}' 错误 - {e.Message}");
            socketFailCounts[socketName] = socketFailCounts.GetValueOrDefault(socketName, 0) + 1;
            
            // 如果异常持续发生，尝试重新连接
            if (socketFailCounts[socketName] > 3)
            {
                ReconnectSocket(socketName);
                socketFailCounts[socketName] = 0;
            }
            return false;
        }
    }
    
    /// <summary>
    /// 关闭所有套接字
    /// </summary>
    public void CloseAllSockets()
    {
        foreach (var socketName in new List<string>(sockets.Keys))
        {
            CloseSocket(socketName);
        }
        
        sockets.Clear();
        socketConnectionStatus.Clear();
    }

    /// <summary>
    /// 解析手腕部数据
    /// </summary>
    /// <param name="message">消息字符串，格式：typeMarker:x,y,z|x,y,z|...</param>
    /// <returns>格式化的手腕部数据字符串</returns>
    private string ParseWristData(string message)
    {
        try
        {
            if (string.IsNullOrEmpty(message))
                return "空数据";

            string[] parts = message.Split(':');
            if (parts.Length < 2)
                return "格式错误";

            string[] joints = parts[1].Split('|');
            if (joints.Length < 2)
                return "关节数据不足";

            string wrist = joints[0].Trim();
            string palm = joints[1].Trim();

            return $"手腕={wrist} 手掌={palm}";
        }
        catch (Exception e)
        {
            return "解析错误: " + e.Message;
        }
    }

    /// <summary>
    /// 解析26个关节数据
    /// </summary>
    /// <param name="message">消息字符串，格式：typeMarker:x,y,z|x,y,z|...</param>
    /// <returns>格式化的26个关节数据字符串</returns>
    private string ParseFullJointData(string message)
    {
        try
        {
            if (string.IsNullOrEmpty(message))
                return "空数据";

            string[] parts = message.Split(':');
            if (parts.Length < 2)
                return "格式错误";

            string[] joints = parts[1].Split('|');
            string result = "";
            
            for (int i = 0; i < Mathf.Min(26, joints.Length); i++)
            {
                string joint = joints[i].Trim();
                result += $"{i}:{joint}" + (i < 25 ? " " : "");
            }

            return result;
        }
        catch (Exception e)
        {
            return "解析错误: " + e.Message;
        }
    }

    /// <summary>
    /// 关闭并释放特定套接字
    /// </summary>
    /// <param name="socketName">套接字名称</param>
    public void CloseSocket(string socketName)
    {
        try
        {
            if (!sockets.ContainsKey(socketName))
            {
                Debug.LogWarning($"NetMQController: 套接字 '{socketName}' 不存在");
                return;
            }
            
            PushSocket socket = sockets[socketName];
            
            if (socket != null)
            {
                socket.Close();
                socket.Dispose();
                Debug.Log($"NetMQController: 套接字 '{socketName}' 已关闭并释放");
            }
            
            sockets.Remove(socketName);
            socketConnectionStatus.Remove(socketName);
        }
        catch (Exception e)
        {
            Debug.LogError($"NetMQController: 关闭套接字 '{socketName}' 错误 - {e.GetType().Name}: {e.Message}");
        }
    }
    
    /// <summary>
    /// 记录所有套接字的状态
    /// </summary>
    public void LogSocketStatus()
    {
        Debug.Log("===== NETMQ 套接字状态 =====");
        Debug.Log($"IP地址: {ipAddress}");
        
        if (sockets.Count == 0)
        {
            Debug.Log("未创建套接字");
        }
        else
        {
            foreach (var socketName in sockets.Keys)
            {
                Debug.Log($"套接字: {socketName} - 已连接: {socketConnectionStatus[socketName]}");
            }
        }
        
        Debug.Log("===============================");
    }
    
    /// <summary>
    /// 应用程序退出时执行清理
    /// </summary>
    private void OnApplicationQuit()
    {
        CleanupNetMQ();
    }
    
    /// <summary>
    /// 清理NetMQ资源
    /// </summary>
    public void CleanupNetMQ()
    {
        try
        {
            // 先关闭所有套接字
            CloseAllSockets();
            
            // 然后清理NetMQ
            if (netMQInitialized)
            {
                Debug.Log("NetMQController: 清理NetMQ...");
                NetMQConfig.Cleanup(false);
                netMQInitialized = false;
                Debug.Log("NetMQController: NetMQ已清理");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"NetMQController: 清理NetMQ错误 - {e.GetType().Name}: {e.Message}");
        }
    }
    
    /// <summary>
    /// 通过发送测试消息执行诊断测试
    /// 作用：在正式开始发送手势之前，确保所有的网络通道都是通畅的
    /// </summary>
    /// <returns>如果所有测试成功则返回true，否则返回false</returns>
    public bool PerformDiagnosticTests()
    {
        Debug.Log("NetMQController: 开始诊断测试...");
        bool allSuccessful = true;
        
        // 如果套接字为空，可能是IP未定义
        if (sockets.Count == 0)
        {
            Debug.LogWarning("NetMQController: 没有可用的套接字进行诊断测试");
            return false;
        }
        
        // 测试每个套接字
        foreach (var socketName in sockets.Keys)
        {
            string testMsg = $"DIAGNOSTIC_TEST_{socketName}_{DateTime.Now:HH:mm:ss.fff}";
            bool success = SendMessage(socketName, testMsg);
            
            Debug.Log($"NetMQController: 诊断测试 '{socketName}' - 成功: {success}");
            
            if (!success)
            {
                allSuccessful = false;
            }
        }
        
        Debug.Log($"NetMQController: 诊断测试完成 - 整体成功: {allSuccessful}");
        return allSuccessful;
    }
    
    /// <summary>
    /// 检查NetMQ是否已初始化
    /// </summary>
    /// <returns>如果已初始化则返回true，否则返回false</returns>
    public bool IsInitialized()
    {
        return netMQInitialized;
    }

    /// <summary>
    /// 使用提供的配置连接到所有套接字
    /// </summary>
    /// <param name="ipAddress">IP地址</param>
    /// <param name="rightHandAddress">右手数据地址</param>
    /// <param name="leftHandAddress">左手数据地址</param>
    /// <param name="resolutionAddress">分辨率控制地址</param>
    /// <param name="pauseAddress">暂停控制地址</param>
    public void Connect(string ipAddress, string rightHandAddress, string leftHandAddress, 
                       string resolutionAddress, string pauseAddress)
    {
        try
        {
            // 存储IP地址
            this.ipAddress = ipAddress;
            
            // 关闭任何现有的套接字
            CloseAllSockets();
            
            // 如果需要，初始化NetMQ
            if (!netMQInitialized)
            {
                InitializeNetMQ();
            }
            
            // 使用提供的完整地址创建套接字
            if (!string.IsNullOrEmpty(rightHandAddress) && rightHandAddress != "tcp://:")
                CreateSocket("RightHand", rightHandAddress);
            
            if (!string.IsNullOrEmpty(leftHandAddress) && leftHandAddress != "tcp://:")
                CreateSocket("LeftHand", leftHandAddress);
            
            if (!string.IsNullOrEmpty(resolutionAddress) && resolutionAddress != "tcp://:")
                CreateSocket("Resolution", resolutionAddress);
            
            if (!string.IsNullOrEmpty(pauseAddress) && pauseAddress != "tcp://:")
                CreateSocket("Pause", pauseAddress);
            
            // 记录套接字状态
            LogSocketStatus();
            
            // 测试连接
            PerformDiagnosticTests();
        }
        catch (Exception e)
        {
            Debug.LogError($"NetMQController: 连接错误 - {e.Message}");
            // 即使出现错误，也不抛出异常，允许应用继续运行
        }
    }

    /// <summary>
    /// 检查所有必需的套接字是否已连接
    /// </summary>
    /// <returns>如果所有必需的套接字已连接则返回true，否则返回false</returns>
    public bool AreSocketsConnected()
    {
        // 如果IP未定义，我们未连接
        if (string.IsNullOrEmpty(ipAddress) || ipAddress == "undefined")
            return false;
        
        // 检查我们是否有最低要求的套接字
        bool hasRightHand = sockets.ContainsKey("RightHand") && sockets["RightHand"] != null;
        bool hasLeftHand = sockets.ContainsKey("LeftHand") && sockets["LeftHand"] != null;
        
        return hasRightHand && hasLeftHand;
    }

    /// <summary>
    /// 尝试重新连接特定套接字
    /// </summary>
    /// <param name="socketName">套接字名称</param>
    private void ReconnectSocket(string socketName)
    {
        try
        {
            Debug.Log($"尝试重新连接套接字: {socketName}");
            
            // 关闭现有的套接字
            if (sockets.ContainsKey(socketName) && sockets[socketName] != null)
            {
                sockets[socketName].Close();
                sockets[socketName].Dispose();
            }
            
            // 根据套接字类型确定地址
            string address = "";
            switch (socketName)
            {
                case "RightHand":
                    address = $"tcp://{ipAddress}:{rightKeypointPort}";
                    break;
                case "LeftHand":
                    address = $"tcp://{ipAddress}:{leftKeypointPort}";
                    break;
                case "Resolution":
                    address = $"tcp://{ipAddress}:{resolutionPort}";
                    break;
                case "Pause":
                    address = $"tcp://{ipAddress}:{pausePort}";
                    break;
                default:
                    Debug.LogError($"未知的套接字类型: {socketName}");
                    return;
            }
            
            // 创建新套接字
            var socket = new PushSocket();
            socket.Options.SendHighWatermark = 1000;
            socket.Options.Linger = TimeSpan.FromMilliseconds(100);
            socket.Connect(address);
            
            // 替换字典中的套接字
            sockets[socketName] = socket;
            
            Debug.Log($"套接字 {socketName} 已重新连接到 {address}");
        }
        catch (Exception e)
        {
            Debug.LogError($"重新连接套接字 {socketName} 错误: {e.Message}");
            // 标记为损坏但不抛出异常
            if (sockets.ContainsKey(socketName))
            {
                sockets[socketName] = null;
            }
        }
    }
}

/// <summary>
/// 用于从JSON反序列化网络设置的类
/// </summary>
[Serializable]
public class NetworkSettings
{
    public string IPAddress;
    public string rightkeyptPortNum;
    public string leftkeyptPortNum;
    public string camPortNum;
    public string graphPortNum;
    public string resolutionPortNum;
    public string PausePortNum;
    public string LeftPausePortNum;
    public string RightPausePortNum;
} 