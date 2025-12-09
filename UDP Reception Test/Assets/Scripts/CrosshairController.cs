using UnityEngine;

public class CrosshairController : MonoBehaviour
{
    public RectTransform crosshair;
    public float sensitivity = .8f;
    public float expFactor = 1.2f;
    Vector2 smoothedCrosshair = Vector2.zero;

    public void UpdateCrosshair(float wristX, float wristY, float indexX, float indexY)
    {
        Vector2 wristScreen = new Vector2(wristX * Screen.width, (1 - wristY) * Screen.height);
        Vector2 indexScreen = new Vector2(indexX * Screen.width, (1 - indexY) * Screen.height);
        Vector2 indexDsplc = new Vector2(indexScreen.x - wristScreen.x, indexScreen.y - wristScreen.y);
        Vector2 dir = indexDsplc.normalized;
        float indexMagnitude = indexDsplc.magnitude;

        float scaledMagnitude = Mathf.Pow(indexMagnitude, expFactor);

        Vector2 aimPos = wristScreen + dir * scaledMagnitude * sensitivity;
        aimPos.x = Mathf.Clamp(aimPos.x, 0, Screen.width);
        aimPos.y = Mathf.Clamp(aimPos.y, 0, Screen.height);

        if (smoothedCrosshair == Vector2.zero)
        {
            smoothedCrosshair = aimPos;
        }

        smoothedCrosshair = Vector2.Lerp(smoothedCrosshair, aimPos, 0.2f);
        if (crosshair == null)
        {
            CrosshairController controller = FindAnyObjectByType<CrosshairController>();
            crosshair = controller.crosshair;
        }
        crosshair.position = smoothedCrosshair;

        Debug.DrawLine(new Vector3(wristScreen.x, wristScreen.y, 0), new Vector3(aimPos.x, aimPos.y, 0), Color.red);
    }
}
