using System;
using UnityEngine;

public static class AndroidPerformanceMode
{
	private const string LogPrefix = "[AndroidPerformanceMode]";
#if UNITY_ANDROID && !UNITY_EDITOR
	private static AndroidJavaObject _wifiLock;
	private static AndroidJavaObject _wakeLock;
#endif
	private static bool _acquired;
	private static bool _batteryOptimizationRequestStarted;
	private static int _previousSleepTimeout = SleepTimeout.SystemSetting;
	private static bool _previousRunInBackground;

	public static void Acquire()
	{
		if (_acquired)
			return;

		_previousSleepTimeout = Screen.sleepTimeout;
		_previousRunInBackground = Application.runInBackground;
		Screen.sleepTimeout = SleepTimeout.NeverSleep;
		Application.runInBackground = true;

#if UNITY_ANDROID && !UNITY_EDITOR
		AcquireAndroidLocks();
#endif
		_acquired = true;
		Debug.Log(LogPrefix + " enabled: screen never sleep, runInBackground=true");
	}

	public static void Release()
	{
		if (!_acquired)
			return;

#if UNITY_ANDROID && !UNITY_EDITOR
		ReleaseAndroidLocks();
#endif
		Screen.sleepTimeout = _previousSleepTimeout;
		Application.runInBackground = _previousRunInBackground;
		_acquired = false;
		Debug.Log(LogPrefix + " released");
	}

	public static void RequestIgnoreBatteryOptimizationsOnce()
	{
#if UNITY_ANDROID && !UNITY_EDITOR
		if (_batteryOptimizationRequestStarted)
			return;

		_batteryOptimizationRequestStarted = true;
		try
		{
			using (AndroidJavaObject activity = GetCurrentActivity())
			using (AndroidJavaObject context = activity.Call<AndroidJavaObject>("getApplicationContext"))
			using (AndroidJavaObject powerManager = GetSystemService(context, "power"))
			{
				string packageName = context.Call<string>("getPackageName");
				bool isIgnoring = powerManager.Call<bool>("isIgnoringBatteryOptimizations", packageName);
				if (isIgnoring)
				{
					Debug.Log(LogPrefix + " already ignoring battery optimizations");
					return;
				}

				using (AndroidJavaClass uriClass = new AndroidJavaClass("android.net.Uri"))
				using (AndroidJavaObject uri = uriClass.CallStatic<AndroidJavaObject>("parse", "package:" + packageName))
				using (AndroidJavaObject intent = new AndroidJavaObject("android.content.Intent", "android.settings.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS", uri))
				{
					activity.Call("startActivity", intent);
					Debug.Log(LogPrefix + " requested ignore battery optimizations");
				}
			}
		}
		catch (Exception e)
		{
			Debug.LogWarning(LogPrefix + " request ignore battery optimizations failed: " + e.Message);
		}
#else
		Debug.Log(LogPrefix + " battery optimization request skipped outside Android player");
#endif
	}

#if UNITY_ANDROID && !UNITY_EDITOR
	private static void AcquireAndroidLocks()
	{
		try
		{
			using (AndroidJavaObject activity = GetCurrentActivity())
			using (AndroidJavaObject context = activity.Call<AndroidJavaObject>("getApplicationContext"))
			{
				AcquireWifiLock(context);
				AcquireWakeLock(context);
			}
		}
		catch (Exception e)
		{
			Debug.LogWarning(LogPrefix + " acquire Android locks failed: " + e.Message);
		}
	}

	private static void AcquireWifiLock(AndroidJavaObject context)
	{
		if (_wifiLock != null)
			return;

		try
		{
			using (AndroidJavaObject wifiManager = GetSystemService(context, "wifi"))
			{
				int lockMode = ResolveWifiLockMode();
				_wifiLock = wifiManager.Call<AndroidJavaObject>("createWifiLock", lockMode, "BeaVRLowLatencyWifiLock");
				_wifiLock.Call("setReferenceCounted", false);
				_wifiLock.Call("acquire");
				Debug.Log(LogPrefix + " WiFi lock acquired, mode=" + lockMode);
			}
		}
		catch (Exception e)
		{
			Debug.LogWarning(LogPrefix + " WiFi lock acquire failed: " + e.Message);
			_wifiLock = null;
		}
	}

	private static void AcquireWakeLock(AndroidJavaObject context)
	{
		if (_wakeLock != null)
			return;

		try
		{
			using (AndroidJavaObject powerManager = GetSystemService(context, "power"))
			{
				int partialWakeLock = 1;
				_wakeLock = powerManager.Call<AndroidJavaObject>("newWakeLock", partialWakeLock, "BeaVR:PicoTeleopWakeLock");
				_wakeLock.Call("setReferenceCounted", false);
				_wakeLock.Call("acquire");
				Debug.Log(LogPrefix + " partial wake lock acquired");
			}
		}
		catch (Exception e)
		{
			Debug.LogWarning(LogPrefix + " wake lock acquire failed: " + e.Message);
			_wakeLock = null;
		}
	}

	private static void ReleaseAndroidLocks()
	{
		ReleaseLock(ref _wifiLock, "WiFi lock");
		ReleaseLock(ref _wakeLock, "wake lock");
	}

	private static void ReleaseLock(ref AndroidJavaObject javaLock, string lockName)
	{
		if (javaLock == null)
			return;

		try
		{
			bool held = javaLock.Call<bool>("isHeld");
			if (held)
				javaLock.Call("release");
			Debug.Log(LogPrefix + " " + lockName + " released");
		}
		catch (Exception e)
		{
			Debug.LogWarning(LogPrefix + " " + lockName + " release failed: " + e.Message);
		}
		finally
		{
			javaLock.Dispose();
			javaLock = null;
		}
	}

	private static AndroidJavaObject GetCurrentActivity()
	{
		using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
		{
			return unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
		}
	}

	private static AndroidJavaObject GetSystemService(AndroidJavaObject context, string serviceName)
	{
		return context.Call<AndroidJavaObject>("getSystemService", serviceName);
	}

	private static int ResolveWifiLockMode()
	{
		using (AndroidJavaClass wifiManager = new AndroidJavaClass("android.net.wifi.WifiManager"))
		{
			try
			{
				return wifiManager.GetStatic<int>("WIFI_MODE_FULL_LOW_LATENCY");
			}
			catch
			{
				try
				{
					return wifiManager.GetStatic<int>("WIFI_MODE_FULL_HIGH_PERF");
				}
				catch
				{
					return 1;
				}
			}
		}
	}
#endif
}
