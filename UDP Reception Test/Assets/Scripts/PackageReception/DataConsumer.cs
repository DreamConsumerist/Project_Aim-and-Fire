using UnityEngine;

public class DataConsumer : MonoBehaviour
{
    public UDP_Reception udpReceiver; // Drag the GameObject with UDP_Reception script onto this field in the Inspector
    [HideInInspector]
    public string[] parser;

    void Update()
    {
        if (udpReceiver == null)
        {
            Debug.LogError("UDP Receiver reference not set!");
            return;
        }

        // Check for the new message using the safe method
        string newMessage = udpReceiver.GetAndClearMessage();

        if (newMessage != null)
        {
            // We are on the main thread now. We can safely update game state.
            //Debug.Log("Consumer Script processing new data: " + newMessage);

            parser = newMessage.Split(',');
            ReceptionEvents.MessageReceived(parser[0], float.Parse(parser[1]), float.Parse(parser[2]), float.Parse(parser[3]), float.Parse(parser[4]));
        }
    }
}