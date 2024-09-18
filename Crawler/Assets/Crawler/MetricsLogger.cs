using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class MetricsLogger : MonoBehaviour
{

    string filePath;

    public string folderPath;
    public string filename;

    public void Awake()
    {
        if (filename != "")
        {
            filePath = string.Format("{0}\\{1}.csv", folderPath, filename);
        }
        else
        {
            filePath = string.Format("{0}\\metrics_{1}.csv", folderPath, DateTime.Now.ToString("yyyy_MM_dd--hh_mm_ss"));
        }

    }

    public void Record(int episode, float steps, float successRate, float avgSpeed, int targetsCollected)
    {
        Debug.Log("EP: " + episode + ", total steps: " + steps + ", success rate sum: " + successRate + ", speed sum: " + avgSpeed + ", collected: " + targetsCollected);
        // It seems it is possible (and quite common) for a target to spawn on top of the agent.
        // Don't record such cases to prevent skewing the data.

        if (steps != 0)
        {
            //Debug.Log(steps);
            var line = string.Format("{0};{1};{2};{3};{4}\n", episode, steps, successRate, avgSpeed, targetsCollected);
            File.AppendAllText(filePath, line);
        }

    }
}
