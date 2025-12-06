using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;
public class PythonProcessManager : MonoBehaviour
{
    Process pythonProcess;
    public string pythonPath; // assign in inspector
    public string scriptPath;    // assign in inspector
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
        if (pythonProcess != null)
        {
            pythonProcess.OutputDataReceived -= OnPythonOutput;
            pythonProcess.ErrorDataReceived -= OnPythonError;
        }
    }

    private void OnPythonOutput(object sender, DataReceivedEventArgs e)
    {
        if (!string.IsNullOrEmpty(e.Data))
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
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            pythonProcess.Kill();
            pythonProcess.Dispose();
        }
    }
}
