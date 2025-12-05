using Mono.Cecil.Cil;
using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using Unity.VisualScripting;
using UnityEngine;

/// A utility class to dispatch actions to the main thread in Unity.
public class UnityMainThreadDispatcher : MonoBehaviour
{
    private static readonly ConcurrentQueue<Action> executionQueue = new ConcurrentQueue<Action>();
    private static UnityMainThreadDispatcher instance = null;

    /// Gets the singleton instance of the dispatcher.
    public static UnityMainThreadDispatcher Instance()
    {
        if (instance == null)
        {
            instance = FindAnyObjectByType<UnityMainThreadDispatcher>();
            if (instance == null)
            {
                GameObject dispatcherObject = new GameObject("UnityMainThreadDispatcher");
                instance = dispatcherObject.AddComponent<UnityMainThreadDispatcher>();
                DontDestroyOnLoad(dispatcherObject); // Keep the object alive across scenes
            }
        }
        return instance;
    }

    /// Enqueues an action to be executed on the main thread.
    public void Enqueue(Action action)
    {
        executionQueue.Enqueue(action);
    }

    // Update is called once per frame, on the main thread
    void Update()
    {
        while (!executionQueue.IsEmpty)
        {
            if (executionQueue.TryDequeue(out Action action))
            {
                action?.Invoke();
            }
        }
    }

    void OnDestroy()
    {
        if (instance == this)
        {
            instance = null;
        }
    }
}