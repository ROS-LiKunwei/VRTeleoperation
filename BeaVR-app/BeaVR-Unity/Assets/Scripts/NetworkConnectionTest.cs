using TMPro; // 引入 TextMeshPro 的命名空间
using UnityEngine;
using UnityEngine.UI;
using System.Collections;

/// <summary>
/// 网络连接测试脚本，用于诊断PICO4设备上的连接问题
/// </summary>
public class NetworkConnectionTest : MonoBehaviour
{
    public TMP_Text statusText;
    public TMP_Text ipText;
    public TMP_Text portText;
    
    private NetworkManager networkManager;
    private NetMQController netMQController;
    
    void Start()
    {
        // 获取网络管理器
        GameObject netConfGameObject = GameObject.Find("NetworkConfigsLoader");
        if (netConfGameObject != null)
        {
            networkManager = netConfGameObject.GetComponent<NetworkManager>();
        }
        
        netMQController = NetMQController.Instance;
        
        // 开始测试
        StartCoroutine(TestNetworkConnection());
    }
    
    IEnumerator TestNetworkConnection()
    {
        ClearStatus();
        UpdateStatus("正在初始化网络连接测试...");
        
        yield return new WaitForSeconds(1f);
        
        // 检查网络管理器
        if (networkManager == null)
        {
            UpdateStatus("错误：NetworkManager未找到");
            yield break;
        }
        
        // 显示IP地址
        string ip = networkManager.netConfig.IPAddress;
        if (string.IsNullOrEmpty(ip) || ip == "undefined")
        {
            UpdateStatus("错误：IP地址未配置");
            yield break;
        }
        
        UpdateIPText(ip);
        UpdatePortText(networkManager.netConfig.rightkeyptPortNum);
        
        // 测试网络连接
        UpdateStatus("正在尝试连接到服务器...");
        
        netMQController.Connect(
            ip,
            networkManager.getRightKeypointAddress(),
            networkManager.getLeftKeypointAddress(),
            networkManager.getResolutionAddress(),
            networkManager.getPauseAddress()
        );
        
        yield return new WaitForSeconds(3f);
        
        // 检查连接状态
        bool connected = netMQController.AreSocketsConnected();
        
        if (connected)
        {
            UpdateStatus("✓ 网络连接成功！");
            UpdateStatusColor(Color.green);
        }
        else
        {
            UpdateStatus("✗ 网络连接失败");
            UpdateStatusColor(Color.red);
            UpdateStatus("请检查：");
            UpdateStatus("1. 后端程序是否正在运行");
            UpdateStatus("2. IP地址是否正确：" + ip);
            UpdateStatus("3. 网络连接是否正常");
        }
    }
    
    void UpdateStatus(string message)
    {
        if (statusText != null)
        {
            statusText.text += message + "\n";
            Debug.Log("[NetworkConnectionTest] " + message);
        }
    }
    
    void UpdateIPText(string ip)
    {
        if (ipText != null)
        {
            ipText.text = "IP: " + ip;
        }
    }
    
    void UpdatePortText(string port)
    {
        if (portText != null)
        {
            portText.text = "端口: " + port;
        }
    }
    
    void UpdateStatusColor(Color color)
    {
        if (statusText != null)
        {
            statusText.color = color;
        }
    }
    
    void ClearStatus()
    {
        if (statusText != null)
        {
            statusText.text = "";
        }
    }
}
