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

	// PICO手部发送频率统计
	[Header("PICO发送频率统计")]
	public bool EnableSendFrequencyLogging = true;
	private int _rightHandSendCount = 0;
	private int _leftHandSendCount = 0;
	private float _lastRightSendFreqLogTime = 0f;
	private float _lastLeftSendFreqLogTime = 0f;
	private float _rightSendFrequency = 0f;
	private float _leftSendFrequency = 0f;
	private const float FREQ_CALC_INTERVAL = 1.0f;

	// PICO手腕部数据打印
	[Header("PICO手腕部数据打印")]
	public bool EnableWristDataLogging = true;
	private float _lastWristLogTime = 0f;
	private const float WRIST_LOG_INTERVAL = 2.0f;

	// 手势发送过滤：避免发送手丢失时的旧姿态，以及毫米级抖动
	[Header("手势发送过滤")]
	[Tooltip("两次已发送手势帧之间的最大关节位移小于该值时跳过发送。单位：米，0.001 = 1毫米。")]
	public float MinGestureSendDeltaMeters = 0.001f;
	private readonly Dictionary<string, List<Vector3>> _lastSentHandFrames = new Dictionary<string, List<Vector3>>();
	private readonly Dictionary<string, string> _lastSentHandModes = new Dictionary<string, string>();

	// PICO 26个坐标系数据打印
    [Header("PICO 26个坐标系数据打印")]
    public bool EnableFullJointLogging = true;
    private float _lastFullJointLogTime = 0f;
    private const float FULL_JOINT_LOG_INTERVAL = 5.0f;

    // 帧索引，用于匹配三个环节的数据
    private int _frameIndex = 0;

	// UI和辅助工具（保持与原始行为匹配）
	public GameObject MenuButton;
	public GameObject ResolutionButton;
	public GameObject HighResolutionButton;
	public GameObject LowResolutionButton;
	// WristTracker视觉已移除
	public RawImage StreamBorder;

	public HighResolutionButtonController HighResolutionButtonController;
	public LowResolutionButtonController LowResolutionButtonController;

	// 头部手势控制：点头开始，摇头结束
	[Header("头部手势控制")]
	public bool EnableHeadGestureControl = true;
	public Button NodStartButton;
	public Button ShakeEndButton;
	[Range(5f, 35f)] public float HeadGestureAngleThresholdDegrees = 12f;
	[Range(0.2f, 2.0f)] public float HeadGestureWindowSeconds = 0.8f;
	[Range(0.2f, 3.0f)] public float HeadGestureCooldownSeconds = 1.2f;
	[Range(0.1f, 5.0f)] public float HeadNeutralFollowSpeed = 1.5f;
	private bool _hasHeadNeutral = false;
	private float _neutralHeadPitch = 0f;
	private float _neutralHeadYaw = 0f;
	private int _nodStage = 0;
	private int _nodDirection = 0;
	private float _nodStageStartTime = 0f;
	private int _shakeStage = 0;
	private int _shakeDirection = 0;
	private float _shakeStageStartTime = 0f;
	private float _headGestureRangeStartTime = 0f;
	private float _minPitchDeltaInWindow = 0f;
	private float _maxPitchDeltaInWindow = 0f;
	private float _minYawDeltaInWindow = 0f;
	private float _maxYawDeltaInWindow = 0f;
	private float _lastHeadGestureTriggerTime = -999f;

	// 网络
	private NetworkManager netConfig;
	private bool connectionAttemptInProgress = false;

	// 模式
	bool StreamRelativeData = false;
	bool StreamAbsoluteData = false;
	bool StreamResolution = false;
	private bool ShouldContinueArmTeleop = false;
	[Header("手势模式切换")]
	public bool EnablePinchStreamingControl = false;

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

		if (NodStartButton == null && MenuButton != null)
			NodStartButton = MenuButton.GetComponentInChildren<Button>(true);
		if (NodStartButton == null)
			NodStartButton = FindSceneButtonByName("start");
		if (ShakeEndButton == null)
			ShakeEndButton = FindSceneButtonByName("end");

		// 给OpenXR一点时间并运行NetMQController初始化
		StartCoroutine(InitializeNetMQAfterDelay());
	}

	/// <summary>
	/// 按场景对象名查找按钮，便于自动绑定start/end按钮。
	/// </summary>
	Button FindSceneButtonByName(string buttonName)
	{
		Button[] buttons = Resources.FindObjectsOfTypeAll<Button>();
		for (int i = 0; i < buttons.Length; i++)
		{
			Button button = buttons[i];
			if (button == null || button.gameObject == null || !button.gameObject.scene.IsValid())
				continue;
			if (string.Equals(button.gameObject.name, buttonName, StringComparison.OrdinalIgnoreCase))
				return button;
		}
		return null;
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

		ProcessHeadGestureControl();

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
		if (EnablePinchStreamingControl)
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
	/// 识别头部点头/摇头动作。点头触发开始，摇头触发结束。
	/// </summary>
	void ProcessHeadGestureControl()
	{
		if (!EnableHeadGestureControl)
			return;

		InputDevice headDevice = InputDevices.GetDeviceAtXRNode(XRNode.Head);
		if (!headDevice.isValid || !headDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion headRotation))
			return;

		GetHeadPitchYaw(headRotation, out float pitch, out float yaw);

		if (!_hasHeadNeutral)
		{
			_neutralHeadPitch = pitch;
			_neutralHeadYaw = yaw;
			_hasHeadNeutral = true;
			return;
		}

		float pitchDelta = Mathf.DeltaAngle(_neutralHeadPitch, pitch);
		float yawDelta = Mathf.DeltaAngle(_neutralHeadYaw, yaw);
		float threshold = Mathf.Max(1f, HeadGestureAngleThresholdDegrees);
		float currentTime = Time.time;

		if (Mathf.Abs(pitchDelta) < threshold * 0.5f && Mathf.Abs(yawDelta) < threshold * 0.5f)
		{
			float follow = Mathf.Clamp01(Time.deltaTime * HeadNeutralFollowSpeed);
			_neutralHeadPitch = Mathf.LerpAngle(_neutralHeadPitch, pitch, follow);
			_neutralHeadYaw = Mathf.LerpAngle(_neutralHeadYaw, yaw, follow);
		}

		if (currentTime - _lastHeadGestureTriggerTime < HeadGestureCooldownSeconds)
			return;

		UpdateHeadGestureRange(pitchDelta, yawDelta, currentTime);
		float pitchRange = _maxPitchDeltaInWindow - _minPitchDeltaInWindow;
		float yawRange = _maxYawDeltaInWindow - _minYawDeltaInWindow;
		bool nodTriggered = pitchRange >= threshold * 1.6f && pitchRange >= yawRange;
		bool shakeTriggered = yawRange >= threshold * 1.4f && yawRange > pitchRange * 1.1f;

		if (shakeTriggered && yawRange >= pitchRange)
		{
			_lastHeadGestureTriggerTime = currentTime;
			ResetHeadGestureState();
			TriggerEndFromHeadGesture();
		}
		else if (nodTriggered)
		{
			_lastHeadGestureTriggerTime = currentTime;
			ResetHeadGestureState();
			TriggerStartFromHeadGesture();
		}
	}

	/// <summary>
	/// 跟踪短时间窗口内的俯仰/左右转头范围，用于更稳定地识别点头和摇头。
	/// </summary>
	void UpdateHeadGestureRange(float pitchDelta, float yawDelta, float currentTime)
	{
		if (_headGestureRangeStartTime <= 0f || currentTime - _headGestureRangeStartTime > HeadGestureWindowSeconds)
		{
			_headGestureRangeStartTime = currentTime;
			_minPitchDeltaInWindow = pitchDelta;
			_maxPitchDeltaInWindow = pitchDelta;
			_minYawDeltaInWindow = yawDelta;
			_maxYawDeltaInWindow = yawDelta;
			return;
		}

		_minPitchDeltaInWindow = Mathf.Min(_minPitchDeltaInWindow, pitchDelta);
		_maxPitchDeltaInWindow = Mathf.Max(_maxPitchDeltaInWindow, pitchDelta);
		_minYawDeltaInWindow = Mathf.Min(_minYawDeltaInWindow, yawDelta);
		_maxYawDeltaInWindow = Mathf.Max(_maxYawDeltaInWindow, yawDelta);
	}

	/// <summary>
	/// 从头显朝向向量计算俯仰和左右转头角，避免直接使用欧拉角导致轴不稳定。
	/// </summary>
	void GetHeadPitchYaw(Quaternion headRotation, out float pitch, out float yaw)
	{
		Vector3 forward = headRotation * Vector3.forward;
		if (forward.sqrMagnitude < 0.0001f)
		{
			pitch = 0f;
			yaw = 0f;
			return;
		}

		forward.Normalize();
		pitch = Mathf.Asin(Mathf.Clamp(forward.y, -1f, 1f)) * Mathf.Rad2Deg;
		yaw = Mathf.Atan2(forward.x, forward.z) * Mathf.Rad2Deg;
	}

	/// <summary>
	/// 检测某个角度轴是否完成一次正反向摆动。
	/// </summary>
	bool UpdateAlternatingHeadGesture(float delta, ref int stage, ref int direction, ref float stageStartTime, float threshold, float currentTime)
	{
		if (stage != 0 && currentTime - stageStartTime > HeadGestureWindowSeconds)
		{
			stage = 0;
			direction = 0;
		}

		if (stage == 0)
		{
			if (Mathf.Abs(delta) >= threshold)
			{
				stage = 1;
				direction = delta > 0f ? 1 : -1;
				stageStartTime = currentTime;
			}
			return false;
		}

		if (stage == 1 && direction != 0 && delta * direction <= -threshold)
			return true;

		return false;
	}

	/// <summary>
	/// 重置头部手势检测状态。
	/// </summary>
	void ResetHeadGestureState()
	{
		_nodStage = 0;
		_nodDirection = 0;
		_shakeStage = 0;
		_shakeDirection = 0;
		_headGestureRangeStartTime = 0f;
	}

	/// <summary>
	/// 将Unity的0-360欧拉角规整到-180到180。
	/// </summary>
	float NormalizeAngle(float angle)
	{
		return Mathf.Repeat(angle + 180f, 360f) - 180f;
	}

	/// <summary>
	/// 点头时闪烁开始按钮，并允许向后端发送手部数据。
	/// </summary>
	void TriggerStartFromHeadGesture()
	{
		Debug.Log("头部手势：点头开始发送手部数据");
		if (NodStartButton != null)
			StartCoroutine(FlashButton(NodStartButton));

		SetHeadGestureStreaming(true);
	}

	/// <summary>
	/// 摇头时闪烁结束按钮，并停止向后端发送手部数据。
	/// </summary>
	void TriggerEndFromHeadGesture()
	{
		Debug.Log("头部手势：摇头停止发送手部数据");
		if (ShakeEndButton != null)
			StartCoroutine(FlashButton(ShakeEndButton));

		SetHeadGestureStreaming(false);
	}

	/// <summary>
	/// 头部手势专用发送开关：只控制数据流，不触发按钮OnClick，也不隐藏UI。
	/// </summary>
	void SetHeadGestureStreaming(bool shouldStream)
	{
		if (shouldStream)
		{
			StreamResolution = false;
			StreamRelativeData = true;
			StreamAbsoluteData = false;
			ShouldContinueArmTeleop = true;
			if (StreamBorder != null) StreamBorder.color = Color.green;
			return;
		}

		StreamRelativeData = false;
		StreamAbsoluteData = false;
		StreamResolution = false;
		ShouldContinueArmTeleop = false;
		ResetLastSentHandFrame("RightHand");
		ResetLastSentHandFrame("LeftHand");
		if (StreamBorder != null) StreamBorder.color = Color.red;
		NetMQController.Instance.SendMessage("Pause", "Low");
	}

	/// <summary>
	/// 显示按钮闪烁提示，但不触发按钮自身的OnClick事件。
	/// </summary>
	IEnumerator FlashButton(Button button)
	{
		if (button == null)
			yield break;

		button.Select();
		if (button.targetGraphic != null)
		{
			Color originalColor = button.targetGraphic.color;
			button.targetGraphic.color = button.colors.pressedColor;
			yield return new WaitForSecondsRealtime(0.12f);
			button.targetGraphic.color = originalColor;
		}
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

			bool sentRight = false;
			bool sentLeft = false;

			// 右手
			List<Vector3> rightHandGestureData = new List<Vector3>();
			if (!CollectHandJointPositions(_handSubsystem.rightHand, rightHandGestureData))
			{
				ResetLastSentHandFrame("RightHand");
			}
			else if (ShouldSendHandFrame("RightHand", rightHandGestureData, typeMarker))
			{
				string rightHandDataString = SerializeVector3List(rightHandGestureData);
				rightHandDataString = typeMarker + ":" + rightHandDataString;
				sentRight = NetMQController.Instance.SendMessage("RightHand", rightHandDataString);
				if (sentRight)
					StoreLastSentHandFrame("RightHand", rightHandGestureData, typeMarker);
			}

			// 左手
			List<Vector3> leftHandGestureData = new List<Vector3>();
			if (!CollectHandJointPositions(_handSubsystem.leftHand, leftHandGestureData))
			{
				ResetLastSentHandFrame("LeftHand");
			}
			else if (ShouldSendHandFrame("LeftHand", leftHandGestureData, typeMarker))
			{
				string leftHandDataString = SerializeVector3List(leftHandGestureData);
				leftHandDataString = typeMarker + ":" + leftHandDataString;
				sentLeft = NetMQController.Instance.SendMessage("LeftHand", leftHandDataString);
				if (sentLeft)
					StoreLastSentHandFrame("LeftHand", leftHandGestureData, typeMarker);
			}

			// PICO发送频率统计
			if (EnableSendFrequencyLogging)
			{
				if (sentRight)
					_rightHandSendCount++;
				if (sentLeft)
					_leftHandSendCount++;

				float currentTime = Time.time;
				if (currentTime - _lastRightSendFreqLogTime >= FREQ_CALC_INTERVAL)
				{
					_rightSendFrequency = _rightHandSendCount / (currentTime - _lastRightSendFreqLogTime);
					_rightHandSendCount = 0;
					_lastRightSendFreqLogTime = currentTime;
					Debug.Log($"[PICO→App] 右手实际发送频率: {_rightSendFrequency:F1} Hz");
				}

				if (currentTime - _lastLeftSendFreqLogTime >= FREQ_CALC_INTERVAL)
				{
					_leftSendFrequency = _leftHandSendCount / (currentTime - _lastLeftSendFreqLogTime);
					_leftHandSendCount = 0;
					_lastLeftSendFreqLogTime = currentTime;
					Debug.Log($"[PICO→App] 左手实际发送频率: {_leftSendFrequency:F1} Hz");
				}
			}

			// PICO手腕部数据打印
			if (EnableWristDataLogging)
			{
				float currentTime = Time.time;
				if (currentTime - _lastWristLogTime >= WRIST_LOG_INTERVAL)
				{
					_lastWristLogTime = currentTime;
					Vector3 rWrist = rightHandGestureData.Count > 0 ? rightHandGestureData[0] : Vector3.zero;
					Vector3 rPalm = rightHandGestureData.Count > 1 ? rightHandGestureData[1] : Vector3.zero;
					Vector3 lWrist = leftHandGestureData.Count > 0 ? leftHandGestureData[0] : Vector3.zero;
					Vector3 lPalm = leftHandGestureData.Count > 1 ? leftHandGestureData[1] : Vector3.zero;
					Debug.Log($"[PICO获取] index={_frameIndex} 右手腕={FormatVec(rWrist)} 右手掌={FormatVec(rPalm)} | 左手腕={FormatVec(lWrist)} 左手掌={FormatVec(lPalm)}");
					_frameIndex++;
				}
			}

			// PICO 26个坐标系数据打印
			if (EnableFullJointLogging)
			{
				float currentTime = Time.time;
				if (currentTime - _lastFullJointLogTime >= FULL_JOINT_LOG_INTERVAL)
				{
					_lastFullJointLogTime = currentTime;

					// 右手26个关节数据
					string rightJoints = "";
					for (int i = 0; i < Mathf.Min(26, rightHandGestureData.Count); i++)
					{
						rightJoints += $"{i}:{FormatVec(rightHandGestureData[i])}" + (i < 25 ? " " : "");
					}

					// 左手26个关节数据
					string leftJoints = "";
					for (int i = 0; i < Mathf.Min(26, leftHandGestureData.Count); i++)
					{
						leftJoints += $"{i}:{FormatVec(leftHandGestureData[i])}" + (i < 25 ? " " : "");
					}

					Debug.Log($"[PICO获取] index={_frameIndex} 右手26关节: {rightJoints}");
					Debug.Log($"[PICO获取] index={_frameIndex} 左手26关节: {leftJoints}");
					_frameIndex++;
				}
			}

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
	bool CollectHandJointPositions(XRHand hand, List<Vector3> outPositions)
	{
		outPositions.Clear();
		if (!hand.isTracked)
			return false;

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

		return true;
	}

	/// <summary>
	/// 判断当前手部帧是否应该发送。与上一次成功发送的帧相比，最大关节位移小于阈值时跳过。
	/// </summary>
	bool ShouldSendHandFrame(string handKey, List<Vector3> currentFrame, string typeMarker)
	{
		if (currentFrame == null || currentFrame.Count == 0)
			return false;

		if (!_lastSentHandModes.TryGetValue(handKey, out var lastMode) ||
			lastMode != typeMarker)
		{
			return true;
		}

		if (!_lastSentHandFrames.TryGetValue(handKey, out var lastFrame) ||
			lastFrame == null ||
			lastFrame.Count != currentFrame.Count)
		{
			return true;
		}

		float threshold = Mathf.Max(0f, MinGestureSendDeltaMeters);
		if (threshold <= 0f)
			return true;

		for (int i = 0; i < currentFrame.Count; i++)
		{
			if (Vector3.Distance(currentFrame[i], lastFrame[i]) >= threshold)
				return true;
		}

		return false;
	}

	/// <summary>
	/// 保存上一次成功发送的手部帧，避免后续被列表复用影响比较。
	/// </summary>
	void StoreLastSentHandFrame(string handKey, List<Vector3> frame, string typeMarker)
	{
		_lastSentHandFrames[handKey] = new List<Vector3>(frame);
		_lastSentHandModes[handKey] = typeMarker;
	}

	/// <summary>
	/// 手部丢失追踪时清除上一帧，避免重新追踪后继续沿用旧姿态比较。
	/// </summary>
	void ResetLastSentHandFrame(string handKey)
	{
		if (_lastSentHandFrames.ContainsKey(handKey))
			_lastSentHandFrames.Remove(handKey);
		if (_lastSentHandModes.ContainsKey(handKey))
			_lastSentHandModes.Remove(handKey);
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

	/// <summary>
	/// 停止遥操作并显示菜单。
	/// </summary>
	public void DeactivateStreaming()
	{
		try
		{
			StreamRelativeData = false;
			StreamAbsoluteData = false;
			StreamResolution = false;
			ShouldContinueArmTeleop = false;
			ResetLastSentHandFrame("RightHand");
			ResetLastSentHandFrame("LeftHand");
			if (StreamBorder != null) StreamBorder.color = Color.red;
			ToggleMenuButton(true);
			NetMQController.Instance.SendMessage("Pause", "Low");
		}
		catch (Exception e)
		{
			Debug.LogError("DeactivateStreaming错误: " + e.Message);
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
