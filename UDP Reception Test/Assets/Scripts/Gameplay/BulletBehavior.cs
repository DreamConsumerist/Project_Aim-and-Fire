using UnityEngine;

public class BulletBehavior : MonoBehaviour
{
    float speed = 50f;
    float life = 5f;
    Rigidbody rb;
    bool gravityDamp = true;

    private void Start()
    {
        Destroy(gameObject, life);
        rb = GetComponent<Rigidbody>();
        rb.linearVelocity = transform.forward * speed;
    }

    public float gravityScale = 0.5f; // 50% of normal gravity

    void FixedUpdate()
    {
        if (gravityDamp)
        {
            rb.AddForce(Physics.gravity * (gravityScale - 1) * rb.mass);
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        gravityDamp = false;
    }
}
