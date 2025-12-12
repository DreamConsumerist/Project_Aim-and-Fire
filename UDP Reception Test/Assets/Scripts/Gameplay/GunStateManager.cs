using UnityEngine;

public class GunStateManager : MonoBehaviour
{
    public static GunStateManager Instance;
    
    public string prevState;
    [HideInInspector]
    public CrosshairController crosshairController;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        crosshairController = FindAnyObjectByType<CrosshairController>();
        ReceptionEvents.OnMessageReceived += UpdateState;
        prevState = "Idle";
    }

    void UpdateState(string state, float wx, float wy, float ix, float iy)
    {
        if (crosshairController == null)
        {
            crosshairController = FindAnyObjectByType<CrosshairController>();
        }
        if ((state == ("Idle")) || (state == "None"))
        {
            //crosshairController.gameObject.SetActive(false);
            //return;
            crosshairController.UpdateCrosshair(wx, wy, ix, iy);
        }
        if (state == "Aim")
        {
            crosshairController.gameObject.SetActive(true);
            crosshairController.UpdateCrosshair(wx, wy, ix, iy);
        }
        if ((state == "Fire") && (prevState != "Fire"))
        {
            crosshairController.gameObject.SetActive(true);
            GunEvents.GunFired(crosshairController.crosshair.position);
        }
        prevState = state;
    }

    private void OnDisable()
    {
        ReceptionEvents.OnMessageReceived -= UpdateState;
    }
}
