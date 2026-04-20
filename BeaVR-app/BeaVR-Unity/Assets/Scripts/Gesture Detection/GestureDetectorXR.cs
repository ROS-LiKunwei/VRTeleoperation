using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Management;
using Unity.XR.CoreUtils;

/// <summary>
/// PICO4手势探测器，使用XR手部追踪系统采集手部关节数据。
/// 支持左手和右手的手势识别，包括捏合手势检测。
/// </summary>
public class GestureDetectorXR : MonoBehaviour
{
	// XR / 手部追踪（使用XRHandSubsystem）
	private XRHandSubsystem _handSubsystem;

	// 关键点的节流遥测日志
	[Header("调试日志")]
	public bool EnableKeypointLogging = false;
	[Range(0.25f, 10f)] public float KeypointLogIntervalSeconds = 1.0f;
	private float _lastKeypointLogTime = 0f;
	private int _lastRightTrackedCount = -1;
	private int _lastLeftTrackedCount = -1;
	private string _lastModeLogged = "";

	// UI和辅助工具（保持与原始行为匹配）
	public GameObject MenuButton;
	public GameObject ResolutionButton;
	public GameObject HighResolutionButton;
	public GameObject LowResolutionButton;
	// WristTracker视觉已移除
	public RawImage StreamBorder;

	public HighResolutionButtonController HighResolutionButtonController;
	public LowResolutionButtonController LowResolutionButtonController;

	// 网络
	private NetworkManager netConfig;
	private bool connectionAttemptInProgress = false;

	// 模式
	bool StreamRelativeData = true;
	bool StreamAbsoluteData = false;
	bool StreamResolution = false;
	private bool ShouldContinueArmTeleop = false;

	// 关节顺序定义（26个关节）
	static readonly XRHandJointID[] k_JointOrder = new XRHandJointID[]
	{
		XRHandJointID.Wrist,
		XRHandJointID.Palm,
		XRHandJointID.ThumbMetacarpal,
		XRHandJointID.ThumbProximal,
		XRHandJointID.ThumbDistal,
		XRHandJointID.ThumbTip,
		XRHandJointID.IndexMetacarpal,
		XRHandJointID.IndexProximal,
		XRHandJointID.IndexIntermediate,
		XRHandJointID.IndexDistal,
		XRHandJointID.IndexTip,
		XRHandJointID.MiddleMetacarpal,
		XRHandJointID.MiddleProximal,
		XRHandJointID.MiddleIntermediate,
		XRHandJointID.MiddleDistal,
		XRHandJointID.MiddleTip,
		XRHandJointID.RingMetacarpal,
		XRHandJointID.RingProximal,
		XRHandJointID.RingIntermediate,
		XRHandJointID.RingDistal,
		XRHandJointID.RingTip,
		XRHandJointID.LittleMetacarpal,
		XRHandJointID.LittleProximal,
		XRHandJointID.LittleIntermediate,
		XRHandJointID.LittleDistal,
		XRHandJointID.LittleTip
	};

    /// <summary>
    /// 初始化手势探测器
    /// </summary>
    void Start()
    {
		// 网络配置
		GameObject netConfGameObject = GameObject.Find("NetworkConfigsLoader");
		if (netConfGameObject != null)
			netConfig = netConfGameObject.GetComponent<NetworkManager>();

		// 获取XR手部子系统
		TryResolveHandSubsystem();

		// 给OpenXR一点时间并运行NetMQController初始化
		StartCoroutine(InitializeNetMQAfterDelay());
	}

	/// <summary>
	/// 延迟初始化NetMQ控制器
	/// </summary>
	IEnumerator InitializeNetMQAfterDelay()
	{
		yield return new WaitForSeconds(2f);
		NetMQController.Instance.CreateStandardSockets();
		NetMQController.Instance.PerformDiagnosticTests();
	}

	/// <summary>
	/// 尝试解析手部子系统
	/// </summary>
	void TryResolveHandSubsystem()
	{
		if (_handSubsystem != null)
			return;
		var loader = XRGeneralSettings.Instance?.Manager?.activeLoader;
		if (loader != null)
		{
			_handSubsystem = loader.GetLoadedSubsystem<XRHandSubsystem>();
			if (_handSubsystem == null)
			{
				Debug.LogWarning("未找到XRHandSubsystem。请确保XR Hands包/功能已启用。");
			}
		}
	}

	/// <summary>
	/// 序列化Vector3列表为字符串
	/// </summary>
	/// <param name="gestureData">手势数据列表</param>
	/// <returns>序列化的字符串</returns>
	public static string SerializeVector3List(List<Vector3> gestureData)
	{
		string vectorString = "";
		foreach (Vector3 vec in gestureData)
			vectorString = vectorString + vec.x + "," + vec.y + "," + vec.z + "|";

		if (vectorString.Length > 0)
			vectorString = vectorString.Substring(0, vectorString.Length - 1) + ":";

		return vectorString;
	}

    /// <summary>
    /// 每帧更新手势探测器
    /// </summary>
    void Update()
    {
		// 如果需要，重新获取子系统（域重新加载等）
		if (_handSubsystem == null)
			TryResolveHandSubsystem();

		bool isConnected = NetMQController.Instance.AreSocketsConnected();
		if (!isConnected)
		{
			if (StreamBorder != null) StreamBorder.color = Color.red;
			string ipAddress = netConfig != null ? netConfig.netConfig.IPAddress : null;
			bool hasIP = !string.IsNullOrEmpty(ipAddress) && ipAddress != "undefined";
			if (!hasIP)
			{
				// 未配置IP：保持菜单可见，以便用户可以配置/连接
				ToggleMenuButton(true);
				return;
			}
			// 配置了IP：避免闪烁：仅在尝试结果后切换可见性
			if (!connectionAttemptInProgress)
			{
				connectionAttemptInProgress = true;
				StartCoroutine(AttemptConnection());
			}
			return;
		}

		connectionAttemptInProgress = false;

		// 处理手势（左手捏合）
		StreamPauser();

		// 发送辅助通道
		SendResolutionThroughController();
		SendPauseStatusThroughController();

		// 发送手部数据
		if (StreamAbsoluteData)
		{
			SendHandDataThroughController("absolute");
			ToggleResolutionButton(false);
		}
		else if (StreamRelativeData)
		{
			SendHandDataThroughController("relative");
			ToggleResolutionButton(false);
		}
		else if (StreamResolution)
		{
			ToggleHighResolutionButton(true);
			ToggleLowResolutionButton(true);
		}
	}

	/// <summary>
	/// 尝试连接到网络
	/// </summary>
	IEnumerator AttemptConnection()
	{
		NetMQController.Instance.Connect(
			netConfig.netConfig.IPAddress,
			netConfig.getRightKeypointAddress(),
			netConfig.getLeftKeypointAddress(),
			netConfig.getResolutionAddress(),
			netConfig.getPauseAddress()
		);

		yield return new WaitForSeconds(2f);

		bool success = NetMQController.Instance.AreSocketsConnected();
		if (StreamBorder != null) StreamBorder.color = success ? Color.green : Color.red;
		ToggleMenuButton(!success);
		connectionAttemptInProgress = false;
	}

	/// <summary>
	/// 使用XR手部进行手势切换（仅左手，以匹配原始）
	/// </summary>
	void StreamPauser()
	{
		bool pinchIndex = false;
		bool pinchMiddle = false;
		bool pinchRing = false;

		if (_handSubsystem == null)
			return;

		var left = _handSubsystem.leftHand;
		if (!left.isTracked)
			return;

		pinchIndex = IsPinching(left, XRHandJointID.IndexTip);
		pinchMiddle = IsPinching(left, XRHandJointID.MiddleTip);
		pinchRing = IsPinching(left, XRHandJointID.RingTip);

		if (pinchMiddle)
		{
			// 中指捏合：绝对数据模式
			StreamRelativeData = false;
			StreamAbsoluteData = true;
			if (StreamBorder != null) StreamBorder.color = Color.blue;
			ToggleMenuButton(false);
			ShouldContinueArmTeleop = true;
		}

		if (pinchIndex)
		{
			// 食指捏合：相对数据模式
			StreamRelativeData = true;
			StreamAbsoluteData = false;
			if (StreamBorder != null) StreamBorder.color = Color.green;
			ToggleMenuButton(false);
			ShouldContinueArmTeleop = true;
		}

		if (pinchRing)
		{
			// 无名指捏合：停止遥操作
			StreamRelativeData = false;
			StreamAbsoluteData = false;
			if (StreamBorder != null) StreamBorder.color = Color.red;
			ToggleMenuButton(true);
			ShouldContinueArmTeleop = false;
		}
	}

	/// <summary>
	/// 检测手指是否捏合
	/// </summary>
	/// <param name="hand">手部对象</param>
	/// <param name="fingerTip">手指尖关节ID</param>
	/// <param name="thresholdMeters">距离阈值（米）</param>
	/// <returns>如果捏合则返回true，否则返回false</returns>
	bool IsPinching(XRHand hand, XRHandJointID fingerTip, float thresholdMeters = 0.02f)
	{
		var thumb = hand.GetJoint(XRHandJointID.ThumbTip);
		var tip = hand.GetJoint(fingerTip);
		if (!thumb.TryGetPose(out var tPose) || !tip.TryGetPose(out var fPose))
			return false;
		Vector3 tp = ToWorldPosition(tPose.position);
		Vector3 fp = ToWorldPosition(fPose.position);
		return Vector3.Distance(tp, fp) < thresholdMeters;
	}

	/// <summary>
	/// 转换到世界坐标
	/// </summary>
	/// <param name="pos">本地坐标</param>
	/// <returns>世界坐标</returns>
	Vector3 ToWorldPosition(Vector3 pos)
	{
		// 使用XR手部追踪：世界空间位置
		return pos;
	}

	/// <summary>
	/// 通过控制器发送手部数据
	/// </summary>
	/// <param name="typeMarker">数据类型标记（"relative"或"absolute"）</param>
	void SendHandDataThroughController(string typeMarker)
	{
		try
		{
			if (_handSubsystem == null)
				return;

			// 右手
			List<Vector3> rightHandGestureData = new List<Vector3>();
			CollectHandJointPositions(_handSubsystem.rightHand, rightHandGestureData);
			string rightHandDataString = SerializeVector3List(rightHandGestureData);
			rightHandDataString = typeMarker + ":" + rightHandDataString;
			NetMQController.Instance.SendMessage("RightHand", rightHandDataString);

			// 左手
			List<Vector3> leftHandGestureData = new List<Vector3>();
			CollectHandJointPositions(_handSubsystem.leftHand, leftHandGestureData);
			string leftHandDataString = SerializeVector3List(leftHandGestureData);
			leftHandDataString = typeMarker + ":" + leftHandDataString;
			NetMQController.Instance.SendMessage("LeftHand", leftHandDataString);

			// 节流的设备日志，以便您可以通过adb验证我们发送的内容
			if (EnableKeypointLogging)
			{
				int rTotal = rightHandGestureData.Count;
				int lTotal = leftHandGestureData.Count;
				int rTracked = CountNonZeroJoints(rightHandGestureData);
				int lTracked = CountNonZeroJoints(leftHandGestureData);
				bool countsChanged = rTracked != _lastRightTrackedCount || lTracked != _lastLeftTrackedCount;
				bool modeChanged = _lastModeLogged != typeMarker;
				bool intervalElapsed = Time.time - _lastKeypointLogTime > Mathf.Max(0.1f, KeypointLogIntervalSeconds);
				if (countsChanged || modeChanged || intervalElapsed)
				{
					int sampleIndex = Mathf.Min(10, Mathf.Max(0, rTotal - 1)); // 如果可用，优先使用IndexTip
					Vector3 rSample = rTotal > 0 ? rightHandGestureData[sampleIndex] : Vector3.zero;
					Vector3 lSample = lTotal > 0 ? leftHandGestureData[sampleIndex] : Vector3.zero;
					Debug.Log(
						$"GestureDetectorXR: 发送 {typeMarker} | 右手 关节={rTotal} 追踪={rTracked} 示例={FormatVec(rSample)} | 左手 关节={lTotal} 追踪={lTracked} 示例={FormatVec(lSample)}");
					_lastKeypointLogTime = Time.time;
					_lastRightTrackedCount = rTracked;
					_lastLeftTrackedCount = lTracked;
					_lastModeLogged = typeMarker;
				}
			}
		}
		catch (Exception e)
		{
			Debug.LogError("发送手部数据错误 (XR): " + e.Message);
		}
	}

	/// <summary>
	/// 采集手部关节位置
	/// </summary>
	/// <param name="hand">手部对象</param>
	/// <param name="outPositions">输出位置列表</param>
	void CollectHandJointPositions(XRHand hand, List<Vector3> outPositions)
	{
		outPositions.Clear();
		for (int i = 0; i < k_JointOrder.Length; i++)
		{
			var joint = hand.GetJoint(k_JointOrder[i]);
			if (joint.TryGetPose(out Pose pose))
			{
				outPositions.Add(ToWorldPosition(pose.position));
			}
			else
			{
				outPositions.Add(Vector3.zero);
			}
		}
	}

	/// <summary>
	/// 计算非零关节数量
	/// </summary>
	/// <param name="positions">位置列表</param>
	/// <returns>非零关节数量</returns>
	int CountNonZeroJoints(List<Vector3> positions)
	{
		int count = 0;
		for (int i = 0; i < positions.Count; i++)
		{
			if (positions[i] != Vector3.zero) count++;
		}
		return count;
	}

	/// <summary>
	/// 格式化Vector3为字符串
	/// </summary>
	/// <param name="v">Vector3对象</param>
	/// <returns>格式化的字符串</returns>
	string FormatVec(Vector3 v)
	{
		return $"({v.x:F3},{v.y:F3},{v.z:F3})";
	}

	/// <summary>
	/// 通过控制器发送分辨率状态
	/// </summary>
	void SendResolutionThroughController()
	{
		try
		{
			string state = "None";
			if (HighResolutionButtonController != null && HighResolutionButtonController.HighResolution)
			{
				state = "High";
			}
			else if (LowResolutionButtonController != null && LowResolutionButtonController.LowResolution)
			{
				state = "Low";
			}
			NetMQController.Instance.SendMessage("Resolution", state);
		}
		catch (Exception e)
		{
			Debug.LogError("发送分辨率数据错误: " + e.Message);
		}
	}

	/// <summary>
	/// 通过控制器发送暂停状态
	/// </summary>
	void SendPauseStatusThroughController()
	{
		try
		{
			string pauseState = ShouldContinueArmTeleop ? "High" : "Low";
			NetMQController.Instance.SendMessage("Pause", pauseState);
		}
		catch (Exception e)
		{
			Debug.LogError("发送暂停状态错误: " + e.Message);
		}
	}

	/// <summary>
	/// 切换菜单按钮可见性
	/// </summary>
	/// <param name="toggle">是否显示</param>
	public void ToggleMenuButton(bool toggle)
	{
		try
		{
			if (MenuButton != null)
				MenuButton.SetActive(toggle);
		}
		catch (Exception e)
		{
			Debug.LogError("ToggleMenuButton错误: " + e.Message);
		}
	}

	/// <summary>
	/// 切换分辨率按钮可见性
	/// </summary>
	/// <param name="toggle">是否显示</param>
	public void ToggleResolutionButton(bool toggle)
	{
		try
		{
			if (ResolutionButton != null)
				ResolutionButton.SetActive(toggle);
		}
		catch (Exception e)
		{
			Debug.LogError("ToggleResolutionButton错误: " + e.Message);
		}
	}

	/// <summary>
	/// 切换高分辨率按钮
	/// </summary>
	/// <param name="toggle">是否激活</param>
	public void ToggleHighResolutionButton(bool toggle)
	{
		Debug.Log("HighResolutionButton切换 (XR): " + toggle);
	}

	/// <summary>
	/// 切换低分辨率按钮
	/// </summary>
	/// <param name="toggle">是否激活</param>
	public void ToggleLowResolutionButton(bool toggle)
	{
		Debug.Log("LowResolutionButton切换 (XR): " + toggle);
	}

	/// <summary>
	/// 激活流传输
	/// </summary>
	/// <param name="mode">数据模式（"relative"或"absolute"）</param>
	public void ActivateStreaming(string mode = "relative")
	{
		try
		{
			string normalized = (mode ?? "relative").ToLowerInvariant();
			StreamResolution = false;
			if (normalized == "absolute")
			{
				StreamRelativeData = false;
				StreamAbsoluteData = true;
				if (StreamBorder != null) StreamBorder.color = Color.blue;
			}
			else
			{
				StreamRelativeData = true;
				StreamAbsoluteData = false;
				if (StreamBorder != null) StreamBorder.color = Color.green;
			}
			ToggleMenuButton(false);
			ShouldContinueArmTeleop = true;
		}
		catch (Exception e)
		{
			Debug.LogError("ActivateStreaming错误: " + e.Message);
		}
	}

	// 公开的辅助工具用于保持活动连接
	/// <summary>
	/// 检查所有连接是否已建立
	/// </summary>
	/// <returns>如果所有连接已建立则返回true，否则返回false</returns>
	public bool AreAllConnectionsEstablished()
	{
		return NetMQController.Instance != null && NetMQController.Instance.AreSocketsConnected();
	}

	/// <summary>
	/// 发送保持活动ping
	/// </summary>
	public void SendKeepAlivePing()
	{
		try
		{
			NetMQController.Instance.SendMessage("Pause", "KEEPALIVE");
		}
		catch (Exception e)
		{
			Debug.LogError("保持活动ping失败: " + e.Message);
		}
	}

	/// <summary>
	/// 应用程序退出时调用
	/// </summary>
	void OnApplicationQuit()
	{
	}

	/// <summary>
	/// 销毁时调用
	/// </summary>
	void OnDestroy()
	{
	}
}
