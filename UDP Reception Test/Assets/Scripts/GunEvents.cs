using System;
using UnityEngine;

public static class GunEvents
{
    public static Action<Vector2> OnGunFired;

    public static void GunFired(Vector2 crosshairPos)
    {
        OnGunFired?.Invoke(crosshairPos);
        Debug.Log("Invoking gunfire");
    }
}
