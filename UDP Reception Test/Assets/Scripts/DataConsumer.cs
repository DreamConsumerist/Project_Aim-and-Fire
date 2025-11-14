using UnityEngine;

public class DataConsumer : MonoBehaviour
{
    public UDP_Reception udpReceiver; // Drag the GameObject with UDP_Reception script onto this field in the Inspector

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
            Debug.Log("Consumer Script processing new data: " + newMessage);

            // Example: Parse data and update a local variable
            // Assume the message is just a number string, e.g., "15.4"
            if (float.TryParse(newMessage, out float value))
            {
                // Update a local variable
                float speed = value;
                Debug.Log("Updated local speed variable to: " + speed);
            }
        }
    }
}