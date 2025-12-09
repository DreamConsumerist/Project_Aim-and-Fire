using UnityEngine;
using UnityEngine.SceneManagement;

public class StartButtonActions : MonoBehaviour
{
    public void StartGame()
    {
        SceneManager.LoadScene("GameplayScene");
    }

    public void TestClick()
    {
        Debug.Log("Button clicked!");
    }
}
