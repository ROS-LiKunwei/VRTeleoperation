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
    private readonly object imageLock = new object();
    private byte[] latestImageBytes;
    private bool hasNewImage;

    public RawImage image;
    private Texture2D texture;

    [Header("VR Display")]
    public bool keepInViewCenter = true;
    public Transform viewTransform;
    public float viewDistance = 1.6f;
    public Vector2 maxDisplaySizeMeters = new Vector2(1.25f, 0.8f);
    public Vector3 viewOffsetMeters = Vector3.zero;

    //public NetworkConfigs netConf;
    private bool connectionEstablished = false;
    private volatile bool imageThreadRunning;
    private string communicationAddress;
    private NetworkManager netConfig;
    private SubscriberSocket socket;

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

        ConfigureDisplayRect(16f / 9f);
        ResolveViewTransform();
        PlaceDisplayInViewCenter();
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
    }
}