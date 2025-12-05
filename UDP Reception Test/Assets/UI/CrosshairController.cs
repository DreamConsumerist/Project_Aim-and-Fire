using UnityEngine;
using UnityEngine.InputSystem;

public class CrosshairController : MonoBehaviour
{
    public RectTransform crosshair;

    private void OnEnable()
    {
        ReceptionEvents.OnMessageReceived += UpdateCrosshair;
    }

    private void OnDisable()
    {
        
    }

    private void UpdateCrosshair(float fingerX, float fingerY)
    {
        // Get mouse position
        Vector2 crosshairPos = new Vector2(fingerX*Screen.width, Screen.height - fingerY * Screen.height);

        // If using Screen Space - Overlay, no conversion is needed:
        crosshair.position = crosshairPos;
    }
}
