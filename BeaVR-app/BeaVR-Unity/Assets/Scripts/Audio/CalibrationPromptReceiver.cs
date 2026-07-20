using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;

public class CalibrationPromptReceiver : MonoBehaviour
{
    private static CalibrationPromptReceiver _instance;
    public static CalibrationPromptReceiver Instance
    {
        get
        {
            if (_instance == null)
            {
                GameObject go = new GameObject("CalibrationPromptReceiver");
                _instance = go.AddComponent<CalibrationPromptReceiver>();
                DontDestroyOnLoad(go);
            }
            return _instance;
        }
    }

    public bool EnableCalibrationPrompts = true;
    public string PromptTopic = "fa_calibration_prompt";
    public string ResourcePrefix = "Audio/FaCalibration/";
    [Range(0.2f, 10.0f)] public float RepeatedPromptCooldownSeconds = 2.5f;

    private readonly ConcurrentQueue<string> _pendingPromptKeys = new ConcurrentQueue<string>();
    private readonly Dictionary<string, float> _lastPromptTimes = new Dictionary<string, float>();
    private SubscriberSocket _subscriber;
    private Thread _receiverThread;
    private volatile bool _running;
    private string _address;

    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }

        _instance = this;
        DontDestroyOnLoad(gameObject);
    }

    public void StartReceiver(string address)
    {
        if (string.IsNullOrEmpty(address) || address == "tcp://:")
        {
            Debug.LogWarning("[FA Calibration Prompt] invalid address: " + address);
            return;
        }

        if (_running && string.Equals(_address, address, StringComparison.Ordinal))
            return;

        StopReceiver();
        _address = address;
        _running = true;
        _receiverThread = new Thread(ReceiveLoop)
        {
            IsBackground = true,
            Name = "FaCalibrationPromptReceiver"
        };
        _receiverThread.Start();
        Debug.Log("[FA Calibration Prompt] receiver started: " + address);
    }

    public void StopReceiver()
    {
        _running = false;
        try
        {
            _subscriber?.Close();
            _subscriber?.Dispose();
        }
        catch (Exception e)
        {
            Debug.LogWarning("[FA Calibration Prompt] close failed: " + e.Message);
        }
        _subscriber = null;

        if (_receiverThread != null && _receiverThread.IsAlive)
            _receiverThread.Join(300);
        _receiverThread = null;
    }

    private void ReceiveLoop()
    {
        try
        {
            AsyncIO.ForceDotNet.Force();
            _subscriber = new SubscriberSocket();
            _subscriber.Options.ReceiveHighWatermark = 5;
            _subscriber.Options.Linger = TimeSpan.Zero;
            _subscriber.Connect(_address);
            _subscriber.Subscribe(PromptTopic);

            while (_running)
            {
                NetMQMessage message = new NetMQMessage();
                if (!_subscriber.TryReceiveMultipartMessage(TimeSpan.FromMilliseconds(100), ref message))
                    continue;
                if (message.FrameCount < 2)
                    continue;

                string topic = message[0].ConvertToString();
                if (!string.Equals(topic, PromptTopic, StringComparison.Ordinal))
                    continue;

                string promptKey = message[1].ConvertToString();
                if (!string.IsNullOrEmpty(promptKey))
                    _pendingPromptKeys.Enqueue(promptKey);
            }
        }
        catch (Exception e)
        {
            if (_running)
                Debug.LogWarning("[FA Calibration Prompt] receive loop failed: " + e.Message);
        }
    }

    private void Update()
    {
        if (!EnableCalibrationPrompts)
            return;

        while (_pendingPromptKeys.TryDequeue(out string promptKey))
        {
            float now = Time.time;
            if (_lastPromptTimes.TryGetValue(promptKey, out float lastTime) &&
                now - lastTime < RepeatedPromptCooldownSeconds)
            {
                continue;
            }

            _lastPromptTimes[promptKey] = now;
            string resourcePath = ResourcePrefix + promptKey;
            if (!AndroidTextToSpeech.Instance.PlayVoiceClipResource(resourcePath))
                Debug.LogWarning("[FA Calibration Prompt] missing voice resource: Resources/" + resourcePath);
        }
    }

    private void OnDestroy()
    {
        StopReceiver();
    }

    private void OnApplicationQuit()
    {
        StopReceiver();
    }
}
