using System;
using UnityEngine;

public static class ReceptionEvents
{
    public static Action<string, float, float, float, float> OnMessageReceived;

    public static void MessageReceived(string state, float wristX, float wristY, float indexX, float indexY)
    {
        OnMessageReceived?.Invoke(state, wristX, wristY, indexX, indexY);
    }
}
