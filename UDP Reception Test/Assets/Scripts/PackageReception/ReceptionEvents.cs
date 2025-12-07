using System;
using UnityEngine;

public static class ReceptionEvents
{
    public static Action<float, float, float, float> OnMessageReceived;

    public static void MessageReceived(float wristX, float wristY, float indexX, float indexY)
    {
        OnMessageReceived?.Invoke(wristX, wristY, indexX, indexY);
    }
}
