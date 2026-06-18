using UnityEngine;
using UnityEngine.UI;

using NetMQ;
using NetMQ.Sockets;

using System;
using System.Collections.Generic;
using System.Threading;

public class CameraOneStreamer : MonoBehaviour
{
    private Thread imageStreamer;
    private Thread projectionControlStreamer;
    private readonly object imageLock = new object();
    private readonly object projectionControlLock = new object();
    private byte[] latestImageBytes;
    private bool hasNewImage;
    private string latestProjectionCommand;
    private bool hasProjectionCommand;

    public RawImage image;
    public RawImage statusBackground;
    private Texture2D texture;

    [Header("VR Display")]
    public bool keepInViewCenter = true;
    public Transform viewTransform;
    public float viewDistance = 1.6f;
    public Vector2 maxDisplaySizeMeters = new Vector2(1.25f, 0.8f);
    public Vector3 viewOffsetMeters = Vector3.zero;
    public Vector2 statusBackgroundPaddingPixels = new Vector2(180f, 180f);

    //public NetworkConfigs netConf;
    private bool connectionEstablished = false;
    private volatile bool imageThreadRunning;
    private volatile bool projectionControlThreadRunning;
    private bool projectionVisible = true;
    private string communicationAddress;
    private string projectionControlAddress;
    private NetworkManager netConfig;
    private SubscriberSocket socket;
    private SubscriberSocket projectionControlSocket;

    private void StartImageThread()
    {
        try
        {
            // Check if communication address is available and not forced to disconnect
            communicationAddress = netConfig.getCamAddress();
            bool AddressAvailable = !String.Equals(communicationAddress, "tcp://:");

            if (AddressAvailable && !netConfig.ForceDisconnect)
            {
                StartConnection();
                if (!connectionEstablished)
                {
                    return;
                }

                imageThreadRunning = true;
                imageStreamer = new Thread(getRobotImage);
                imageStreamer.IsBackground = true;
                imageStreamer.Start();
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error starting camera thread: " + e.Message);
        }
    }

    private void StartProjectionControlThread()
    {
        try
        {
            projectionControlAddress = netConfig.getCameraProjectionControlAddress();
            bool AddressAvailable = !String.Equals(projectionControlAddress, "tcp://:");

            if (AddressAvailable && !netConfig.ForceDisconnect)
            {
                StartProjectionControlConnection();
                if (projectionControlSocket == null)
                {
                    return;
                }

                projectionControlThreadRunning = true;
                projectionControlStreamer = new Thread(getProjectionControl);
                projectionControlStreamer.IsBackground = true;
                projectionControlStreamer.Start();
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Error starting camera projection control thread: " + e.Message);
        }
    }

    public void StartConnection()
    {
        try
        {
            // Clean up any existing socket first
            if (socket != null)
            {
                socket.Close();
                socket.Dispose();
            }

            // Initiate Subscriber Socket
            socket = new SubscriberSocket();
            socket.Options.ReceiveHighWatermark = 1000;
            socket.Connect(communicationAddress);
            socket.Subscribe("");
            connectionEstablished = true;
            Debug.Log("Camera connection established to: " + communicationAddress);
        }
        catch (Exception e)
        {
            Debug.LogError("Error establishing camera connection: " + e.Message);
            connectionEstablished = false;
        }
    }

    private void StartProjectionControlConnection()
    {
        try
        {
            if (projectionControlSocket != null)
            {
                projectionControlSocket.Close();
                projectionControlSocket.Dispose();
            }

            projectionControlSocket = new SubscriberSocket();
            projectionControlSocket.Options.ReceiveHighWatermark = 1;
            projectionControlSocket.Connect(projectionControlAddress);
            projectionControlSocket.Subscribe("camera_projection_toggle");
            Debug.Log("Camera projection control connection established to: " + projectionControlAddress);
        }
        catch (Exception e)
        {
            Debug.LogError("Error establishing camera projection control connection: " + e.Message);
            projectionControlSocket = null;
        }
    }

    private void getRobotImage()
    {
        try
        {
            while (true)
            {
                if (!imageThreadRunning)
                {
                    break;
                }

                SubscriberSocket currentSocket = socket;
                if (currentSocket == null)
                {
                    break;
                }

                // The bot camera stream publishes multipart messages as:
                // [topic="image", jpeg_bytes]. Legacy single-frame JPEG streams
                // are still accepted by using the last frame in the message.
                List<byte[]> frames = null;
                if (!currentSocket.TryReceiveMultipartBytes(TimeSpan.FromMilliseconds(100), ref frames, 2))
                {
                    continue;
                }

                if (frames == null || frames.Count == 0)
                {
                    continue;
                }

                byte[] imageBytes = frames[frames.Count - 1];
                if (imageBytes == null || imageBytes.Length == 0)
                {
                    continue;
                }

                lock (imageLock)
                {
                    latestImageBytes = imageBytes;
                    hasNewImage = true;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Camera thread error: " + e.Message);
        }
    }

    private void getProjectionControl()
    {
        try
        {
            while (true)
            {
                if (!projectionControlThreadRunning)
                {
                    break;
                }

                SubscriberSocket currentSocket = projectionControlSocket;
                if (currentSocket == null)
                {
                    break;
                }

                List<string> frames = null;
                if (!currentSocket.TryReceiveMultipartStrings(TimeSpan.FromMilliseconds(100), ref frames, 2))
                {
                    continue;
                }

                if (frames == null || frames.Count == 0)
                {
                    continue;
                }

                string command = frames[frames.Count - 1];
                if (string.IsNullOrEmpty(command))
                {
                    continue;
                }

                lock (projectionControlLock)
                {
                    latestProjectionCommand = command.Trim();
                    hasProjectionCommand = true;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Camera projection control thread error: " + e.Message);
        }
    }

    public void Start()
    {
        // Getting the Network Config Updater gameobject
        GameObject netConfGame = GameObject.Find("NetworkConfigsLoader");
        if (netConfGame != null)
        {
            netConfig = netConfGame.GetComponent<NetworkManager>();
        }
        else
        {
            Debug.LogError("NetworkConfigsLoader not found!");
            return;
        }

        if (image == null)
        {
            Debug.LogError("CameraOneStreamer requires a RawImage reference.");
            enabled = false;
            return;
        }

        // Initializing the image texture
        texture = new Texture2D(2, 2, TextureFormat.RGB24, false);
        image.texture = texture;
        image.raycastTarget = false;

        ResolveStatusBackground();
        ConfigureDisplayRect(16f / 9f);
        ResolveViewTransform();
        PlaceDisplayInViewCenter();
        ApplyProjectionVisibility();
        StartProjectionControlThread();
    }

    public void Update()
    {
        if (connectionEstablished)
        {
            // Check if network manager is forcing disconnect
            if (netConfig.ForceDisconnect)
            {
                DisconnectNetMQ();
                return;
            }

            // To check if the same IP is being used
            if (String.Equals(communicationAddress, netConfig.getCamAddress()))
            {
                // Check if the list has any elements before trying to access them
                byte[] imageBytes = null;
                lock (imageLock)
                {
                    if (hasNewImage)
                    {
                        imageBytes = latestImageBytes;
                        hasNewImage = false;
                    }
                }

                if (imageBytes != null)
                {
                    try
                    {
                        if (texture.LoadImage(imageBytes))
                        {
                            ConfigureDisplayRect((float)texture.width / texture.height);
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogError("Error updating camera texture: " + e.Message);
                    }
                }
            }
            else
            {
                // Address changed, disconnect and reconnect
                DisconnectNetMQ();
            }
        }
        else if (!netConfig.ForceDisconnect)
        {
            StartImageThread();
        }

        UpdateProjectionControl();
    }

    private void LateUpdate()
    {
        PlaceDisplayInViewCenter();
    }

    private void ResolveViewTransform()
    {
        if (viewTransform != null)
        {
            return;
        }

        Camera mainCamera = Camera.main;
        if (mainCamera != null)
        {
            viewTransform = mainCamera.transform;
        }
    }

    private void PlaceDisplayInViewCenter()
    {
        if (!keepInViewCenter)
        {
            return;
        }

        ResolveViewTransform();
        if (viewTransform == null)
        {
            return;
        }

        float distance = Mathf.Max(0.1f, viewDistance);
        transform.position =
            viewTransform.position +
            viewTransform.forward * distance +
            viewTransform.right * viewOffsetMeters.x +
            viewTransform.up * viewOffsetMeters.y +
            viewTransform.forward * viewOffsetMeters.z;
        transform.rotation = viewTransform.rotation;
    }

    private void UpdateProjectionControl()
    {
        string command = null;
        lock (projectionControlLock)
        {
            if (hasProjectionCommand)
            {
                command = latestProjectionCommand;
                hasProjectionCommand = false;
            }
        }

        if (string.IsNullOrEmpty(command))
        {
            return;
        }

        bool previous = projectionVisible;
        if (command == "camera_projection_on")
        {
            projectionVisible = true;
        }
        else if (command == "camera_projection_off")
        {
            projectionVisible = false;
        }
        else if (command == "camera_projection_toggle")
        {
            projectionVisible = !projectionVisible;
        }
        else
        {
            return;
        }

        if (projectionVisible != previous)
        {
            ApplyProjectionVisibility();
            Debug.Log("Camera projection visibility: " + projectionVisible + " command=" + command);
        }
    }

    private void ApplyProjectionVisibility()
    {
        if (image != null)
        {
            image.enabled = projectionVisible;
        }
        ResolveStatusBackground();
        if (statusBackground != null)
        {
            statusBackground.enabled = projectionVisible;
        }
    }

    private void ConfigureDisplayRect(float aspectRatio)
    {
        if (image == null)
        {
            return;
        }

        RectTransform rectTransform = image.rectTransform;
        rectTransform.anchorMin = new Vector2(0.5f, 0.5f);
        rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
        rectTransform.pivot = new Vector2(0.5f, 0.5f);
        rectTransform.anchoredPosition = Vector2.zero;
        rectTransform.localPosition = Vector3.zero;
        rectTransform.localRotation = Quaternion.identity;

        float safeAspectRatio = Mathf.Max(0.01f, aspectRatio);
        float widthMeters = maxDisplaySizeMeters.x;
        float heightMeters = widthMeters / safeAspectRatio;

        if (heightMeters > maxDisplaySizeMeters.y)
        {
            heightMeters = maxDisplaySizeMeters.y;
            widthMeters = heightMeters * safeAspectRatio;
        }

        rectTransform.sizeDelta = new Vector2(1000f, 1000f / safeAspectRatio);
        rectTransform.localScale = new Vector3(
            widthMeters / rectTransform.sizeDelta.x,
            heightMeters / rectTransform.sizeDelta.y,
            1f
        );

        // ConfigureStatusBackground(rectTransform);
    }

    private void ResolveStatusBackground()
    {
        if (statusBackground != null)
        {
            return;
        }

        Transform parent = image != null ? image.transform.parent : transform;
        Transform candidate = parent != null ? parent.Find("ControlIndicatorRawImage") : null;
        if (candidate != null)
        {
            statusBackground = candidate.GetComponent<RawImage>();
        }
    }

    private void ConfigureStatusBackground(RectTransform cameraRect)
    {
        ResolveStatusBackground();
        if (statusBackground == null)
        {
            return;
        }

        RectTransform backgroundRect = statusBackground.rectTransform;
        backgroundRect.anchorMin = cameraRect.anchorMin;
        backgroundRect.anchorMax = cameraRect.anchorMax;
        backgroundRect.pivot = cameraRect.pivot;
        backgroundRect.anchoredPosition = cameraRect.anchoredPosition;
        backgroundRect.localRotation = cameraRect.localRotation;
        backgroundRect.sizeDelta = cameraRect.sizeDelta + statusBackgroundPaddingPixels;
        backgroundRect.localScale = cameraRect.localScale;
        statusBackground.raycastTarget = false;
        statusBackground.transform.SetAsFirstSibling();
    }

    // Add these methods for NetworkManager integration
    void OnDestroy()
    {
        DisconnectNetMQ();
    }

    void OnApplicationQuit()
    {
        DisconnectNetMQ();
    }

    public void DisconnectNetMQ()
    {
        // Safely stop the thread
        imageThreadRunning = false;
        if (imageStreamer != null && imageStreamer.IsAlive)
        {
            try
            {
                if (!imageStreamer.Join(300))
                {
                    imageStreamer.Abort();
                }
                imageStreamer = null;
            }
            catch (Exception e)
            {
                Debug.LogError("Error stopping camera thread: " + e.Message);
            }
        }

        // Close socket
        if (socket != null)
        {
            try
            {
                socket.Close();
                socket.Dispose();
                socket = null;
            }
            catch (Exception e)
            {
                Debug.LogError("Error closing camera socket: " + e.Message);
            }
        }

        projectionControlThreadRunning = false;
        if (projectionControlStreamer != null && projectionControlStreamer.IsAlive)
        {
            try
            {
                if (!projectionControlStreamer.Join(300))
                {
                    projectionControlStreamer.Abort();
                }
                projectionControlStreamer = null;
            }
            catch (Exception e)
            {
                Debug.LogError("Error stopping camera projection control thread: " + e.Message);
            }
        }

        if (projectionControlSocket != null)
        {
            try
            {
                projectionControlSocket.Close();
                projectionControlSocket.Dispose();
                projectionControlSocket = null;
            }
            catch (Exception e)
            {
                Debug.LogError("Error closing camera projection control socket: " + e.Message);
            }
        }

        connectionEstablished = false;
        Debug.Log("Camera connection closed");
    }

    public void ConnectNetMQ()
    {
        // Only reconnect if we're not already connected
        if (!connectionEstablished)
        {
            StartImageThread();
        }
        if (projectionControlSocket == null)
        {
            StartProjectionControlThread();
        }
    }
}
