using System;
using UnityEngine;

public static class ReceptionEvents
{
    public static Action<float, float> OnMessageReceived;

    public static void MessageReceived(float fingerX, float fingerY)
    {
        OnMessageReceived?.Invoke(fingerX, fingerY);
    }
}
