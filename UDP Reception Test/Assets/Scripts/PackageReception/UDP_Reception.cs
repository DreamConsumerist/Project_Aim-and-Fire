using Mono.Cecil.Cil;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class UDP_Reception : MonoBehaviour
{
    // --- Networking variables ---
    private UdpClient udpClient;
    private Thread receiveThread;
    public int port = 5555; // Port to listen on

    // --- Data Storage (Main Thread access only) ---
    private string lastReceivedMessage = "";
    private bool messageAvailable = false;

    // Public property to safely read the message from other scripts
    public string LastMessage
    {
        get { return lastReceivedMessage; }
    }

    void Start()
    {
        UnityMainThreadDispatcher.Instance(); // Ensure dispatcher is ready
        receiveThread = new Thread(ReceiveData);
        receiveThread.IsBackground = true;
        receiveThread.Start();
        Debug.Log("UDP Receiver started on port " + port);
    }

    private void ReceiveData()
    {
        try
        {
            udpClient = new UdpClient(port);
            while (true)
            {
                IPEndPoint remoteIpEndPoint = new IPEndPoint(IPAddress.Any, 0);
                byte[] receivedBytes = udpClient.Receive(ref remoteIpEndPoint);
                string message = System.Text.Encoding.ASCII.GetString(receivedBytes);

                // Use the dispatcher to update the main thread variable
                UnityMainThreadDispatcher.Instance().Enqueue(() =>
                {
                    // This runs on the Main Thread
                    lastReceivedMessage = message;
                    messageAvailable = true; // Flag that a new message is ready
                    Debug.Log("Updated local variable with new data.");
                });
            }
        }
        catch (SocketException ex)
        {
            Debug.LogError("UDP Receive Error: " + ex.Message);
        }
        finally
        {
            if (udpClient != null)
            {
                udpClient.Close();
            }
        }
    }

    // You can call this from another script or within this script's Update loop
    public string GetAndClearMessage()
    {
        if (messageAvailable)
        {
            messageAvailable = false;
            return lastReceivedMessage;
        }
        return null; // Return null if no new message has arrived
    }

    void OnDestroy()
    {
        if (udpClient != null) udpClient.Close();
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Abort();
    }
}