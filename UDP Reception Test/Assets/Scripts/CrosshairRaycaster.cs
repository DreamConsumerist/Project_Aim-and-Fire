using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class CrosshairRaycaster : MonoBehaviour
{
    public Canvas mainMenu;
    public RectTransform crosshair;
    public GraphicRaycaster raycaster;
    public EventSystem eventSystem;

    private void Start()
    {
        GunEvents.OnGunFired += FireUIRaycast;
    }
    private void FireUIRaycast(Vector2 crosshairPos)
    {
        Debug.Log("Attempting graphic raycast");
        PointerEventData data = new PointerEventData(eventSystem);
        data.position = crosshairPos;

        List<RaycastResult> results = new List<RaycastResult>();
        raycaster.Raycast(data, results);

        foreach (var result in results)
        {
            var btn = result.gameObject.GetComponent<Button>();
            if (btn != null)
            {
                btn.onClick.Invoke();
                break;
            }
        }
    }
    private void OnDisable()
    {
        GunEvents.OnGunFired -= FireUIRaycast;
    }
}
