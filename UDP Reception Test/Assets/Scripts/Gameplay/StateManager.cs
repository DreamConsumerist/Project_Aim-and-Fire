using UnityEngine;

public class StateManager : MonoBehaviour
{
    public string currState;
    public CrosshairController crosshair;
    public ShootInCanvas gun;

    void Start()
    {
        ReceptionEvents.OnMessageReceived += UpdateState;
        currState = "Idle";
    }

    void UpdateState(string state, float wx, float wy, float ix, float iy)
    {
        Debug.Log(state);
        if ((currState == ("Idle")) || (currState == "None"))
        {
            crosshair.UpdateCrosshair(wx, wy, ix, iy);
            return;
        }
        if (currState == "Aim")
        {
            crosshair.UpdateCrosshair(wx, wy, ix, iy);
        }
        if (currState == "Fire")
        {
            crosshair.UpdateCrosshair(wx, wy, ix, iy);
            gun.Shoot();
        }
    }

    private void OnDisable()
    {
        ReceptionEvents.OnMessageReceived -= UpdateState;
    }
}
