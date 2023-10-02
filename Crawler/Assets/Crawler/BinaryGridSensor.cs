using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    public class BinaryGridSensor : GridSensorBase
    {
        public BinaryGridSensor(
            string name,
            Vector3 cellScale,
            Vector3Int gridSize,
            string[] detectableTags,
            SensorCompressionType compression
        ) : base(name, cellScale, gridSize, detectableTags, compression)
        {
        }

        protected override int GetCellObservationSize()
        {
            return 0;
        }

        protected override void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)
        {
            dataBuffer[0] = 1;
        }


        //FIX DEFAULT ML-AGENTS FUNCTION
        // writer[...] is in wrong order in default implementation
        /* public new int Write(ObservationWriter writer)
        {
            using (TimerStack.Instance.Scoped("GridSensor.Write"))
            {
                int index = 0;
                for (var h = m_GridSize.z - 1; h >= 0; h--)
                {
                    for (var w = 0; w < m_GridSize.x; w++)
                    {
                        for (var d = 0; d < m_CellObservationSize; d++)
                        {
                            writer[w, h, d] = m_PerceptionBuffer[index];
                            index++;
                        }
                    }
                }
                return index;
            }
        } */
    }
}

