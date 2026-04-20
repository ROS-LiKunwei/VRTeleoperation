using TMPro;
using UnityEngine;
using System.Collections;

/// <summary>
/// 网络连接诊断脚本，用于详细检查网络连接问题
/// </summary>
public class NetworkDiagnosticTest : MonoBehaviour
{
    public TMP_Text diagnosticText;
    
    private NetworkManager networkManager;
    private NetMQController netMQController;
    
    void Start()
    {
        StartCoroutine(RunDiagnostic());
    }
    
    IEnumerator RunDiagnostic()
    {
        ClearDiagnostic();
        
        // 检查NetworkConfigsLoader
        LogDiagnostic("=== 网络连接诊断测试 ===");
        yield return new WaitForSeconds(0.5f);
        
        LogDiagnostic("1. 检查NetworkConfigsLoader游戏对象...");
        GameObject netConfGameObject = GameObject.Find("NetworkConfigsLoader");
        if (netConfGameObject != null)
        {
            LogDiagnostic("   ✓ NetworkConfigsLoader游戏对象存在");
            
            // 检查NetworkManager组件
            networkManager = netConfGameObject.GetComponent<NetworkManager>();
            if (networkManager != null)
            {
                LogDiagnostic("   ✓ NetworkManager组件存在");
                
                // 检查网络配置
                if (networkManager.netConfig != null)
                {
                    LogDiagnostic("   ✓ 网络配置已加载");
                    LogDiagnostic("   IP地址: " + networkManager.netConfig.IPAddress);
                    LogDiagnostic("   右手关键点端口: " + networkManager.netConfig.rightkeyptPortNum);
                    LogDiagnostic("   左手关键点端口: " + networkManager.netConfig.leftkeyptPortNum);
                }
                else
                {
                    LogDiagnostic("   ✗ 网络配置未加载");
                }
            }
            else
            {
                LogDiagnostic("   ✗ NetworkManager组件不存在");
            }
        }
        else
        {
            LogDiagnostic("   ✗ NetworkConfigsLoader游戏对象不存在");
        }
        
        yield return new WaitForSeconds(1f);
        
        // 检查NetMQController
        LogDiagnostic("2. 检查NetMQController...");
        netMQController = NetMQController.Instance;
        if (netMQController != null)
        {
            LogDiagnostic("   ✓ NetMQController实例存在");
            LogDiagnostic("   是否已初始化: " + netMQController.IsInitialized());
        }
        else
        {
            LogDiagnostic("   ✗ NetMQController实例不存在");
        }
        
        yield return new WaitForSeconds(1f);
        
        // 检查PlayerPrefs中的IP
        LogDiagnostic("3. 检查PlayerPrefs中的IP地址...");
        if (PlayerPrefs.HasKey(SaveAndReturnIP.PlayerPrefsKey))
        {
            string savedIP = PlayerPrefs.GetString(SaveAndReturnIP.PlayerPrefsKey);
            LogDiagnostic("   ✓ PlayerPrefs中存在IP地址: " + savedIP);
        }
        else
        {
            LogDiagnostic("   ✗ PlayerPrefs中不存在IP地址");
        }
        
        yield return new WaitForSeconds(1f);
        
        // 检查Network.json配置
        LogDiagnostic("4. 检查Network.json配置文件...");
        var jsonFile = Resources.Load<TextAsset>("Configurations/Network");
        if (jsonFile != null)
        {
            LogDiagnostic("   ✓ Network.json文件存在");
            LogDiagnostic("   文件内容: " + jsonFile.text);
        }
        else
        {
            LogDiagnostic("   ✗ Network.json文件不存在");
        }
        
        yield return new WaitForSeconds(1f);
        
        // 尝试连接测试
        if (networkManager != null && networkManager.netConfig != null && !string.IsNullOrEmpty(networkManager.netConfig.IPAddress))
        {
            LogDiagnostic("5. 尝试网络连接测试...");
            
            netMQController.Connect(
                networkManager.netConfig.IPAddress,
                networkManager.getRightKeypointAddress(),
                networkManager.getLeftKeypointAddress(),
                networkManager.getResolutionAddress(),
                networkManager.getPauseAddress()
            );
            
            yield return new WaitForSeconds(2f);
            
            bool connected = netMQController.AreSocketsConnected();
            if (connected)
            {
                LogDiagnostic("   ✓ 网络连接成功！");
            }
            else
            {
                LogDiagnostic("   ✗ 网络连接失败");
                LogDiagnostic("   请检查：");
                LogDiagnostic("   - 后端程序是否正在运行");
                LogDiagnostic("   - IP地址是否正确: " + networkManager.netConfig.IPAddress);
                LogDiagnostic("   - 网络连接是否正常");
            }
        }
        else
        {
            LogDiagnostic("5. 跳过连接测试：网络配置不完整");
        }
        
        LogDiagnostic("=== 诊断测试完成 ===");
    }
    
    void LogDiagnostic(string message)
    {
        if (diagnosticText != null)
        {
            diagnosticText.text += message + "\n";
        }
        Debug.Log("[NetworkDiagnostic] " + message);
    }
    
    void ClearDiagnostic()
    {
        if (diagnosticText != null)
        {
            diagnosticText.text = "";
        }
    }
}