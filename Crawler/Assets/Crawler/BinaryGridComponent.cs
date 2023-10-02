using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace Unity.MLAgents.Sensors
{
    public class BinaryGridComponent : GridSensorComponent
    {
        protected override GridSensorBase[] GetGridSensors()
        {
            List<GridSensorBase> sensorList = new List<GridSensorBase>();
            var sensor = new BinaryGridSensor(SensorName + "-Binary", CellScale, GridSize, DetectableTags, CompressionType);
            sensorList.Add(sensor);
            return sensorList.ToArray();
        }
    }
}
