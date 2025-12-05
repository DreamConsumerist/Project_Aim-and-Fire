using UnityEngine;
using UnityEngine.InputSystem;

public class ShootInCanvas : MonoBehaviour
{
    public Camera cam;
    public GameObject projectilePrefab;
    public Vector3 muzzlePos;

    private void Start()
    {
        cam = Camera.main;
        muzzlePos = cam.transform.position + new Vector3(0, 0, .5f);
    }
    private void Update()
    {
        if (Mouse.current.leftButton.wasPressedThisFrame)
        {
            Shoot();
        }
    }

    private void Shoot()
    {
        Vector2 mousePos = Mouse.current.position.ReadValue(); //new Vector3(Screen.width / 2, Screen.height / 2)
        Ray ray = cam.ScreenPointToRay(mousePos);

        Vector3 targetPoint;

        if (Physics.Raycast(ray, out RaycastHit hit, 1000f))
        {
            targetPoint = hit.point;
        }
        else
        {
            targetPoint = ray.GetPoint(1000f);
        }

        Vector3 dir = (targetPoint - muzzlePos).normalized;

        Instantiate(projectilePrefab, muzzlePos, Quaternion.LookRotation(dir));
    }
}
