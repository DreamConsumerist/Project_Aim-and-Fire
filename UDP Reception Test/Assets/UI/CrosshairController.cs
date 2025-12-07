using UnityEngine;

public class CrosshairController : MonoBehaviour
{
    public RectTransform crosshair;
    Vector2 smoothedCrosshair = Vector2.zero;

    private void OnEnable()
    {
        ReceptionEvents.OnMessageReceived += UpdateCrosshair;
    }

    private void OnDisable()
    {
        
    }

    private void UpdateCrosshair(float wristX, float wristY, float indexX, float indexY)
    {
        float mod = .6f;
        Vector2 wristScreen = new Vector2(wristX * Screen.width, (1 - wristY) * Screen.height);
        Vector2 indexScreen = new Vector2(indexX * Screen.width, (1 - indexY) * Screen.height);
        Vector2 indexDsplc = new Vector2(indexScreen.x - wristScreen.x, indexScreen.y - wristScreen.y);
        Vector2 dir = indexDsplc.normalized;
        float indexMagnitude = indexDsplc.magnitude;

        float expFactor = 1.2f;
        float scaledMagnitude = Mathf.Pow(indexMagnitude, expFactor);

        Vector2 aimPos = wristScreen + dir * scaledMagnitude * mod;
        aimPos.x = Mathf.Clamp(aimPos.x, 0, Screen.width);
        aimPos.y = Mathf.Clamp(aimPos.y, 0, Screen.height);

        if (smoothedCrosshair == Vector2.zero)
        {
            smoothedCrosshair = aimPos;
        }

        smoothedCrosshair = Vector2.Lerp(smoothedCrosshair, aimPos, 0.2f);
        crosshair.position = smoothedCrosshair;

        Debug.DrawLine(new Vector3(wristScreen.x, wristScreen.y, 0), new Vector3(aimPos.x, aimPos.y, 0), Color.red);
        //Debug.Log("aimPosX: " + aimPosX + ", aimPosY: " + aimPosY + ", wristPosX: " + wristPosX + ", wristPosY: " + wristPosY);
    }
}
