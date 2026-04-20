using UnityEngine;
using TMPro;

/// <summary>
/// CurrentIPLabel 脚本用于在UI上显示当前的IP地址
/// 工作机制：
/// 1. 首先尝试从PlayerPrefs中获取IP地址（用户保存的设置）
/// 2. 如果PlayerPrefs中没有IP地址，从Network.json配置文件中获取
/// 3. 如果Network.json中也没有IP地址，使用默认值192.168.1.133
/// 4. 将获取到的IP地址显示在指定的TextMeshProUGUI组件上
/// </summary>
public class CurrentIPLabel : MonoBehaviour
{
    [Header("UI")]
    /// <summary>
    /// 显示IP地址的TextMeshProUGUI组件
    /// </summary>
    public TextMeshProUGUI target;

    [Header("Formatting")]
    /// <summary>
    /// 显示在IP地址前面的前缀文本
    /// </summary>
    public string prefix = "Current IP Address:";
    /// <summary>
    /// 当无法获取IP地址时显示的 fallback 文本
    /// </summary>
    public string emptyFallback = "Unknown";

    /// <summary>
    /// 当脚本启用时调用
    /// 负责获取IP地址并显示在UI上
    /// </summary>
    private void OnEnable()
    {
        Debug.Log("[CurrentIPLabel] OnEnable called");
        
        // 检查target是否设置，如果没有设置，尝试从自身获取TextMeshProUGUI组件
        if (target == null)
        {
            Debug.Log("[CurrentIPLabel] Target is null, trying to get from component");
            target = GetComponent<TextMeshProUGUI>();
        }
        
        // 如果仍然没有找到target，输出错误信息并返回
        if (target == null)
        {
            Debug.LogError("[CurrentIPLabel] Target TextMeshProUGUI not found");
            return;
        }
        else
        {
            Debug.Log("[CurrentIPLabel] Target found: " + target.gameObject.name);
        }

        // 存储IP地址的变量
        string ip = string.Empty;
        
        try
        {
            // 首先尝试从PlayerPrefs获取IP地址
            string playerPrefsKey = "ServerIP"; // 直接使用硬编码的键值
            Debug.Log("[CurrentIPLabel] Using PlayerPrefs key: " + playerPrefsKey);
            
            ip = PlayerPrefs.GetString(playerPrefsKey, string.Empty);
            Debug.Log("[CurrentIPLabel] IP from PlayerPrefs: '" + ip + "'");
        }
        catch (System.Exception e)
        {
            Debug.LogError("[CurrentIPLabel] Error accessing PlayerPrefs: " + e.Message);
        }
        
        // 如果PlayerPrefs中没有IP地址，从Network.json文件中获取默认值
        if (string.IsNullOrEmpty(ip))
        {
            Debug.Log("[CurrentIPLabel] IP from PlayerPrefs is empty, trying to load from Network.json");
            try
            {
                // 加载Network.json文件
                var jsonFile = Resources.Load<TextAsset>("Configurations/Network");
                if (jsonFile != null)
                {
                    Debug.Log("[CurrentIPLabel] Network.json found");
                    string jsonText = jsonFile.text;
                    Debug.Log("[CurrentIPLabel] Network.json content length: " + jsonText.Length);
                    
                    // 使用更可靠的方式解析JSON文件
                    // 寻找IPAddress字段的起始位置
                    int ipAddressIndex = jsonText.IndexOf("IPAddress");
                    if (ipAddressIndex != -1)
                    {
                        // 寻找冒号
                        int colonIndex = jsonText.IndexOf(":", ipAddressIndex);
                        if (colonIndex != -1)
                        {
                            // 寻找引号
                            int quoteIndex = jsonText.IndexOf("\"", colonIndex);
                            if (quoteIndex != -1)
                            {
                                // 寻找结束引号
                                int endQuoteIndex = jsonText.IndexOf("\"", quoteIndex + 1);
                                if (endQuoteIndex != -1)
                                {
                                    // 提取IP地址
                                    ip = jsonText.Substring(quoteIndex + 1, endQuoteIndex - quoteIndex - 1);
                                    Debug.Log("[CurrentIPLabel] IP from Network.json: '" + ip + "'");
                                }
                                else
                                {
                                    Debug.LogError("[CurrentIPLabel] End quote not found");
                                }
                            }
                            else
                            {
                                Debug.LogError("[CurrentIPLabel] Start quote not found");
                            }
                        }
                        else
                        {
                            Debug.LogError("[CurrentIPLabel] Colon not found");
                        }
                    }
                    else
                    {
                        Debug.LogError("[CurrentIPLabel] IPAddress field not found in Network.json");
                    }
                }
                else
                {
                    Debug.LogError("[CurrentIPLabel] Network.json not found");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError("[CurrentIPLabel] Error loading Network.json: " + e.Message);
            }
        }
        
        // 如果仍然没有IP地址，使用默认值
        if (string.IsNullOrEmpty(ip))
        {
            ip = "192.168.1.133";
            Debug.Log("[CurrentIPLabel] Using default IP address: " + ip);
        }
        
        // 显示最终获取到的IP地址
        Debug.Log("[CurrentIPLabel] Final IP: '" + ip + "'");
        target.text = string.IsNullOrEmpty(ip)
            ? $"{prefix} {emptyFallback}"
            : $"{prefix} {ip}";
        Debug.Log("[CurrentIPLabel] Text set to: '" + target.text + "'");
    }
}


