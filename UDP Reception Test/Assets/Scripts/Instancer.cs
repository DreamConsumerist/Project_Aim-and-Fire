using UnityEngine;

public class GameObjectInstancer : MonoBehaviour
{
    public static GameObject Instance;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this.gameObject;
            DontDestroyOnLoad(this.gameObject);
        }
        else
        {
            Destroy(this.gameObject);
        }
    }
}
