using System;
using UnityEngine;
using TMPro;

/// <summary>
/// 网络配置类，用于存储和管理网络连接的配置信息。
/// </summary>
[System.Serializable]
public class NetworkConfiguration
{
    /// <summary>服务器IP地址</summary>
    public string IPAddress;
    /// <summary>右手关键点数据端口</summary>
    public string rightkeyptPortNum;
    /// <summary>左手关键点数据端口</summary>
    public string leftkeyptPortNum;
    /// <summary>摄像头数据端口</summary>
    public string camPortNum;
    /// <summary>图形反馈端口</summary>
    public string graphPortNum;
    /// <summary>分辨率控制端口</summary>
    public string resolutionPortNum;
    /// <summary>暂停控制端口</summary>
    public string PausePortNum;
    /// <summary>左手暂停状态端口</summary>
    public string LeftPausePortNum;
    /// <summary>右手暂停状态端口</summary>
    public string RightPausePortNum;

    /// <summary>
    /// 检查IP地址是否已分配
    /// </summary>
    /// <returns>如果IP地址已分配则返回true，否则返回false</returns>
    public bool isIPAllocated()
    {
        if (String.Equals(IPAddress, "undefined") || string.IsNullOrEmpty(IPAddress))
            return false;
        else
            return true;
    }
}

/// <summary>
/// 网络管理器类，负责管理网络配置和连接状态。
/// </summary>
public class NetworkManager : MonoBehaviour
{
    // 加载网络配置
    public NetworkConfiguration netConfig;

    // 菜单显示变量
    public TextMeshProUGUI IPDisplay;

    // 指示IP是否找到
    private bool IPNotFound;

    // 强制断开连接标志
    private bool _forceDisconnect = false;
    public bool ForceDisconnect 
    {
        get { return _forceDisconnect; }
        set 
        {
            _forceDisconnect = value;
            if (value) {
                // 通知所有组件断开连接
                BroadcastMessage("DisconnectNetMQ", SendMessageOptions.DontRequireReceiver);
            }
        }
    }

    /// <summary>
    /// 获取右手关键点数据的ZMQ地址
    /// </summary>
    /// <returns>右手关键点数据的ZMQ地址</returns>
    public string getRightKeypointAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.rightkeyptPortNum;
    }

    /// <summary>
    /// 获取左手关键点数据的ZMQ地址
    /// </summary>
    /// <returns>左手关键点数据的ZMQ地址</returns>
    public string getLeftKeypointAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.leftkeyptPortNum;
    }

    /// <summary>
    /// 获取摄像头数据的ZMQ地址
    /// </summary>
    /// <returns>摄像头数据的ZMQ地址</returns>
    public string getCamAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.camPortNum;
    }

    /// <summary>
    /// 获取图形反馈的ZMQ地址
    /// </summary>
    /// <returns>图形反馈的ZMQ地址</returns>
    public string getGraphAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.graphPortNum;
    }

    /// <summary>
    /// 获取分辨率控制的ZMQ地址
    /// </summary>
    /// <returns>分辨率控制的ZMQ地址</returns>
    public string getResolutionAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.resolutionPortNum;
    }

    /// <summary>
    /// 获取暂停控制的ZMQ地址
    /// </summary>
    /// <returns>暂停控制的ZMQ地址</returns>
    public string getPauseAddress()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.PausePortNum;
    }

    /// <summary>
    /// 获取左手暂停状态的ZMQ地址
    /// </summary>
    /// <returns>左手暂停状态的ZMQ地址</returns>
    public string getLeftPauseStatus()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.LeftPausePortNum;
    }

    /// <summary>
    /// 获取右手暂停状态的ZMQ地址
    /// </summary>
    /// <returns>右手暂停状态的ZMQ地址</returns>
    public string getRightPauseStatus()
    {
        if (IPNotFound)
            return "tcp://:";
        else
            return "tcp://" + netConfig.IPAddress + ":" + netConfig.RightPausePortNum;
    }

    // changeIPAddress 不再需要；IP 从 PlayerPrefs[ServerIP] 获取

    /// <summary>
    /// 初始化网络管理器
    /// </summary>
    void Start()
    {
        // 从Resources文件夹加载网络配置文件
        var jsonFile = Resources.Load<TextAsset>("Configurations/Network");
        netConfig = JsonUtility.FromJson<NetworkConfiguration>(jsonFile.text);

        // 从PlayerPrefs加载IP地址（由用户界面设置）
        if (PlayerPrefs.HasKey(SaveAndReturnIP.PlayerPrefsKey))
            netConfig.IPAddress = PlayerPrefs.GetString(SaveAndReturnIP.PlayerPrefsKey);

        // 检查IP地址是否已分配
        if (!netConfig.isIPAllocated())
            IPNotFound = true;
        else
            IPNotFound = false;        
    }

    /// <summary>
    /// 更新网络管理器状态
    /// </summary>
    void Update()
    {
        // 显示IP信息
        if (IPDisplay != null)
        {
            if (!IPNotFound)
                IPDisplay.text = "IP Address: " + netConfig.IPAddress;
            else
                IPDisplay.text = "IP Address: Not Specified";
        }
    }

    /// <summary>
    /// 更新连接反馈信息
    /// </summary>
    /// <param name="message">反馈信息</param>
    public void UpdateConnectionFeedback(string message)
    {
        GameObject fieldInputManager = GameObject.Find("FieldInputManager");
        if (fieldInputManager != null)
        {
            FieldInputManager inputManager = fieldInputManager.GetComponent<FieldInputManager>();
            if (inputManager != null && inputManager.feedbackText != null)
            {
                inputManager.feedbackText.text = message;
            }
        }
    }

    /// <summary>
    /// 连接所有网络组件
    /// </summary>
    public void ConnectAllNetworkComponents()
    {
        _forceDisconnect = false;
        BroadcastMessage("ConnectNetMQ", SendMessageOptions.DontRequireReceiver);
        UpdateConnectionFeedback("Attempting to connect...");
    }

    /// <summary>
    /// 断开所有网络组件的连接
    /// </summary>
    public void DisconnectAllNetworkComponents()
    {
        _forceDisconnect = true;
        BroadcastMessage("DisconnectNetMQ", SendMessageOptions.DontRequireReceiver);
        UpdateConnectionFeedback("Network connections closed");
    }
}