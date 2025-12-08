using UnityEngine;

public class StateManager : MonoBehaviour
{
    public string prevState;
    public CrosshairController crosshair;
    public ShootInCanvas gun;

    void Start()
    {
        ReceptionEvents.OnMessageReceived += UpdateState;
        prevState = "Idle";
    }

    void UpdateState(string state, float wx, float wy, float ix, float iy)
    {
        if ((state == ("Idle")) || (state == "None"))
        {
            crosshair.gameObject.SetActive(false);
            return;
        }
        if (state == "Aim")
        {
            crosshair.gameObject.SetActive(true);
        }
        if ((state == "Fire") && (prevState != "Fire"))
        {
            crosshair.gameObject.SetActive(true);
            gun.Shoot();
        }
        if (crosshair.gameObject.activeSelf)
        {
            crosshair.UpdateCrosshair(wx, wy, ix, iy); // Maybe make a LockCrosshair(time t) for firing so that the crosshair's not thrown off
        }
        prevState = state;
    }

    private void OnDisable()
    {
        ReceptionEvents.OnMessageReceived -= UpdateState;
    }
}
