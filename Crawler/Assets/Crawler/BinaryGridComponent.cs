using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;
using System.Linq;

public class BinaryGridComponent : GridSensorComponent
{
    List<BinaryGridSensor> binarySensors;
    protected override GridSensorBase[] GetGridSensors()
    {
        binarySensors = new List<BinaryGridSensor>();
        var sensor = new BinaryGridSensor(SensorName + "-Binary", CellScale, GridSize, DetectableTags, CompressionType);
        binarySensors.Add(sensor);
        return binarySensors.ToArray();
    }

    public List<BinaryGridSensor> BinarySensors
    {
        get { return binarySensors; }
    }

    public int GetObjectTag()
    {
        return binarySensors.First().LastObjectTag();
    }
}

