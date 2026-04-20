using System.Collections;
using UnityEngine;

public class StartTeleopButton : MonoBehaviour
{
    [Header("References")]
    public NetworkManager networkManager; // optional, will auto-find if null
    public CanvasSwitch canvasSwitch;     // target canvas switcher

    [Header("Behavior")]
    public float connectTimeoutSeconds = 2.0f;

    public void OnClick_StartTeleop()
    {
        try
        {
            if (networkManager == null)
            {
                var go = GameObject.Find("NetworkConfigsLoader");
                if (go != null) networkManager = go.GetComponent<NetworkManager>();
            }

            if (networkManager == null)
            {
                Debug.LogError("[StartTeleop] NetworkManager not found.");
                return;
            }

            // 优先从PlayerPrefs获取IP，如果没有则从NetworkManager获取
            string ip = PlayerPrefs.GetString("ServerIP", string.Empty);
            if (string.IsNullOrEmpty(ip) || ip == "undefined")
            {
                // 从NetworkManager的配置中获取IP
                ip = networkManager.netConfig.IPAddress;
                if (string.IsNullOrEmpty(ip) || ip == "undefined")
                {
                    Debug.LogWarning("[StartTeleop] No IP set in PlayerPrefs or Network.json.");
                    return;
                }
            }

            NetMQController.Instance.Connect(
                ip,
                networkManager.getRightKeypointAddress(),
                networkManager.getLeftKeypointAddress(),
                networkManager.getResolutionAddress(),
                networkManager.getPauseAddress()
            );

            StartCoroutine(WaitAndSwitch());
        }
        catch (System.Exception e)
        {
            Debug.LogError("[StartTeleop] Error starting teleop: " + e.Message);
            // 即使出现错误，也不抛出异常，允许应用继续运行
        }
    }

    private IEnumerator WaitAndSwitch()
    {
        float t = 0f;
        while (t < connectTimeoutSeconds)
        {
            yield return null;
            t += Time.unscaledDeltaTime;
        }

        try
        {
            bool ok = NetMQController.Instance.AreSocketsConnected();
            Debug.Log("[StartTeleop] Connection status after timeout: " + ok);

            if (ok && canvasSwitch != null)
            {
                canvasSwitch.Switch();

                // After switching, set streaming active (relative by default)
                var gd = FindFirstObjectByType<GestureDetectorXR>();
                if (gd != null)
                {
                    gd.ActivateStreaming("relative");
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError("[StartTeleop] Error in WaitAndSwitch: " + e.Message);
            // 即使出现错误，也不抛出异常，允许应用继续运行
        }
    }
}


