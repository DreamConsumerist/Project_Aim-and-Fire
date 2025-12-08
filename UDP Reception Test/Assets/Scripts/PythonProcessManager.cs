using System.Diagnostics;
using Unity.VisualScripting.Antlr3.Runtime;
using UnityEngine;
using System.IO;
using Debug = UnityEngine.Debug;
public class PythonProcessManager : MonoBehaviour
{
    Process pythonProcess;
    public string pythonPath; // assign in inspector
    public string scriptPath;    // assign in inspector
    string[] parser;
    void Start()
    {
        StartPython();
    }

    void StartPython()
    {
        var psi = new ProcessStartInfo();

        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = $"\"{scriptPath}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = false
        };
        startInfo.Arguments = $"-u \"{scriptPath}\"";

        pythonProcess = new Process { StartInfo = startInfo };
        pythonProcess.OutputDataReceived += OnPythonOutput;
        pythonProcess.ErrorDataReceived += OnPythonError;
        pythonProcess.Start();
        pythonProcess.BeginOutputReadLine();
        pythonProcess.BeginErrorReadLine();
        Debug.Log("Started Python listener.");
    }

    void OnDisable()
    {
        CleanupPython();
    }

    private void OnPythonOutput(object sender, DataReceivedEventArgs e)
    {
        if (string.IsNullOrEmpty(e.Data))
            return;

        Debug.Log("[Python] " + e.Data);
    }

    private void OnPythonError(object sender, DataReceivedEventArgs e)
    {
        if (!string.IsNullOrEmpty(e.Data) &&
            !e.Data.StartsWith("INFO:") &&
            !e.Data.StartsWith("WARNING:") &&
            !e.Data.Contains("Feedback manager") &&
            !e.Data.Contains("SymbolDatabase.GetPrototype"))
        {
            Debug.LogError("[Python ERR] " + e.Data);
        }
    }


    void OnApplicationQuit()
    {
        CleanupPython();
    }
    void CleanupPython()
    {
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            pythonProcess.Kill();
            pythonProcess.WaitForExit(); // ensures it actually terminates
            pythonProcess.Dispose();
            pythonProcess = null;
        }
    }
}
