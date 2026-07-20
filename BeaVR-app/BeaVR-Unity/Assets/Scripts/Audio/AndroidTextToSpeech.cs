using System;
using UnityEngine;

public class AndroidTextToSpeech : MonoBehaviour
{
    private static AndroidTextToSpeech _instance;
    public static AndroidTextToSpeech Instance
    {
        get
        {
            if (_instance == null)
            {
                GameObject go = new GameObject("AndroidTextToSpeech");
                _instance = go.AddComponent<AndroidTextToSpeech>();
                DontDestroyOnLoad(go);
            }
            return _instance;
        }
    }

    [Range(0.2f, 2.0f)] public float SpeechRate = 1.0f;
    [Range(0.2f, 2.0f)] public float Pitch = 1.0f;
    public bool PreferBundledVoicePrompt = true;
    public string StartedVoiceClipResource = "Audio/teleop_started";
    public string PausedVoiceClipResource = "Audio/teleop_paused";
    public bool EnableToneFallback = true;
    public bool AlwaysPlayTonePrompt = true;
    [Range(0.0f, 1.0f)] public float ToneVolume = 0.25f;
    [Range(0.0f, 1.0f)] public float VoiceClipVolume = 1.0f;

#if UNITY_ANDROID && !UNITY_EDITOR
    private AndroidJavaObject _tts;
    private AndroidJavaObject _activity;
    private bool _engineCreated;
    private bool _ready;
    private string _pendingText;
    private string _queuedText;
    private string _toneFallbackText;
#endif
    private AudioSource _audioSource;

    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }

        _instance = this;
        DontDestroyOnLoad(gameObject);
        _audioSource = gameObject.AddComponent<AudioSource>();
        _audioSource.playOnAwake = false;
        _audioSource.spatialBlend = 0.0f;
        Initialize();
    }

    public void Speak(string text)
    {
        if (string.IsNullOrEmpty(text))
            return;

        if (PreferBundledVoicePrompt && TryPlayBundledVoicePrompt(text))
            return;

        if (AlwaysPlayTonePrompt)
            PlayToneFallback(text);

#if UNITY_ANDROID && !UNITY_EDITOR
        Debug.Log("[TTS] request: " + text);
        Initialize();
        if (_tts == null)
        {
            _pendingText = text;
            Debug.LogWarning("[TTS] engine not ready yet, queued text and playing tone fallback");
            if (!AlwaysPlayTonePrompt)
                PlayToneFallback(text);
            return;
        }
        if (!_ready)
        {
            _pendingText = text;
            if (!AlwaysPlayTonePrompt)
                PlayToneFallback(text);
            return;
        }

        QueueSpeak(text);
#else
        Debug.Log("[TTS] " + text);
#endif
    }

    public bool PlayVoiceClipResource(string resourcePath)
    {
        if (string.IsNullOrEmpty(resourcePath))
            return false;

        return TryPlayVoiceClipResource(resourcePath, logMissing: true);
    }

    private void Update()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!string.IsNullOrEmpty(_toneFallbackText))
        {
            string toneText = _toneFallbackText;
            _toneFallbackText = null;
            PlayToneFallback(toneText);
        }

        if (!_ready || _tts == null || string.IsNullOrEmpty(_queuedText))
            return;

        string text = _queuedText;
        _queuedText = null;
        SpeakNowWithVisualFallback(text);
#endif
    }

    private void Initialize()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (_engineCreated)
            return;
        _engineCreated = true;

        try
        {
            using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            {
                _activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
                _activity.Call(
                    "runOnUiThread",
                    new AndroidJavaRunnable(() =>
                    {
                        try
                        {
                            _tts = new AndroidJavaObject(
                                "android.speech.tts.TextToSpeech",
                                _activity,
                                new TtsInitListener(this));
                            Debug.Log("[TTS] engine create requested");
                        }
                        catch (Exception e)
                        {
                            Debug.LogWarning("[TTS] engine create failed on UI thread: " + e.Message);
                            _tts = null;
                            _engineCreated = false;
                        }
                    }));
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("AndroidTextToSpeech init failed: " + e.Message);
            _tts = null;
            _engineCreated = false;
        }
#endif
    }

#if UNITY_ANDROID && !UNITY_EDITOR
    private void OnTtsReady(int status)
    {
        if (_tts == null)
            return;

        if (status != 0)
        {
            Debug.LogWarning("AndroidTextToSpeech init returned status: " + status);
            return;
        }

        try
        {
            using (AndroidJavaClass localeClass = new AndroidJavaClass("java.util.Locale"))
            {
                AndroidJavaObject chineseLocale = localeClass.GetStatic<AndroidJavaObject>("CHINESE");
                int languageResult = _tts.Call<int>("setLanguage", chineseLocale);
                Debug.Log("[TTS] setLanguage(CHINESE) result=" + languageResult);
            }
            _tts.Call<int>("setSpeechRate", SpeechRate);
            _tts.Call<int>("setPitch", Pitch);
            _ready = true;
            Debug.Log("[TTS] ready");
            if (!string.IsNullOrEmpty(_pendingText))
            {
                string text = _pendingText;
                _pendingText = null;
                QueueSpeak(text);
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("AndroidTextToSpeech ready setup failed: " + e.Message);
        }
    }

    private void QueueSpeak(string text)
    {
        _queuedText = text;
    }

    private void SpeakNow(string text)
    {
        try
        {
            if (_activity == null)
            {
                RequestToneFallback(text);
                return;
            }
            _activity.Call(
                "runOnUiThread",
                new AndroidJavaRunnable(() =>
                {
                    try
                    {
                        using (AndroidJavaObject bundle = new AndroidJavaObject("android.os.Bundle"))
                        {
                            int result = _tts.Call<int>("speak", text, 0, bundle, Guid.NewGuid().ToString());
                            Debug.Log("[TTS] speak result=" + result + " text=" + text);
                            if (result < 0)
                                RequestToneFallback(text);
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning("AndroidTextToSpeech speak failed on UI thread: " + e.Message);
                        RequestToneFallback(text);
                    }
                }));
        }
        catch (Exception e)
        {
            Debug.LogWarning("AndroidTextToSpeech speak failed: " + e.Message);
            RequestToneFallback(text);
        }
    }

    private void RequestToneFallback(string text)
    {
        _toneFallbackText = text;
    }
#endif

    private void PlayToneFallback(string text)
    {
        if (!EnableToneFallback)
            return;
        if (_audioSource == null)
            return;

        bool started = !string.IsNullOrEmpty(text) && text.Contains("开启");
        AudioClip clip = BuildToneClip(started);
        _audioSource.Stop();
        _audioSource.volume = ToneVolume;
        _audioSource.PlayOneShot(clip);
    }

    private bool TryPlayBundledVoicePrompt(string text)
    {
        bool started = !string.IsNullOrEmpty(text) && text.Contains("开启");
        string resourcePath = started ? StartedVoiceClipResource : PausedVoiceClipResource;
        return TryPlayVoiceClipResource(resourcePath, logMissing: true);
    }

    private bool TryPlayVoiceClipResource(string resourcePath, bool logMissing)
    {
        if (_audioSource == null || string.IsNullOrEmpty(resourcePath))
            return false;

        AudioClip clip = Resources.Load<AudioClip>(resourcePath);
        if (clip == null)
        {
            if (logMissing)
                Debug.LogWarning("[TTS] bundled voice clip not found: Resources/" + resourcePath);
            return false;
        }

        _audioSource.Stop();
        _audioSource.PlayOneShot(clip, VoiceClipVolume);
        Debug.Log("[TTS] bundled voice clip played: " + resourcePath);
        return true;
    }

    private AudioClip BuildToneClip(bool started)
    {
        const int sampleRate = 24000;
        float duration = started ? 0.28f : 0.36f;
        int sampleCount = Mathf.CeilToInt(sampleRate * duration);
        float frequency = started ? 880.0f : 440.0f;
        AudioClip clip = AudioClip.Create(started ? "teleop_on_tone" : "teleop_off_tone", sampleCount, 1, sampleRate, false);
        float[] samples = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            float t = (float)i / sampleRate;
            float envelope = Mathf.Clamp01(Mathf.Min(t / 0.03f, (duration - t) / 0.04f));
            samples[i] = Mathf.Sin(2.0f * Mathf.PI * frequency * t) * envelope;
        }
        clip.SetData(samples, 0);
        return clip;
    }

#if UNITY_ANDROID && !UNITY_EDITOR
    private void ShowToast(string text)
    {
        if (_activity == null || string.IsNullOrEmpty(text))
            return;
        try
        {
            _activity.Call(
                "runOnUiThread",
                new AndroidJavaRunnable(() =>
                {
                    using (AndroidJavaClass toastClass = new AndroidJavaClass("android.widget.Toast"))
                    {
                        AndroidJavaObject toast = toastClass.CallStatic<AndroidJavaObject>(
                            "makeText",
                            _activity,
                            text,
                            0);
                        toast.Call("show");
                    }
                }));
        }
        catch (Exception e)
        {
            Debug.LogWarning("AndroidTextToSpeech toast failed: " + e.Message);
        }
    }

    private void SpeakNowWithVisualFallback(string text)
    {
        ShowToast(text);
        SpeakNow(text);
    }
#endif

#if UNITY_ANDROID && !UNITY_EDITOR
    private class TtsInitListener : AndroidJavaProxy
    {
        private readonly AndroidTextToSpeech _owner;

        public TtsInitListener(AndroidTextToSpeech owner)
            : base("android.speech.tts.TextToSpeech$OnInitListener")
        {
            _owner = owner;
        }

        public void onInit(int status)
        {
            _owner.OnTtsReady(status);
        }
    }
#endif

    private void OnDestroy()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (_tts != null)
        {
            try
            {
                _tts.Call("stop");
                _tts.Call("shutdown");
            }
            catch (Exception e)
            {
                Debug.LogWarning("AndroidTextToSpeech shutdown failed: " + e.Message);
            }
            _tts = null;
        }
#endif
    }
}
