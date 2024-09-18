using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Handles = UnityEditor.Handles;
using Random = UnityEngine.Random;
using System;

[RequireComponent(typeof(JointDriveController))] // Required to set joint forces
public class CrawlerAgent : Agent
{

    [Header("Metrics")]
    public bool endEPOnGoal = false;
    public MetricsLogger metricsLogger;

    //Metrics
    int m_stepsToTarget;

    //Multi-target metrics
    int m_targetsCollected;
    float m_totalSteps;
    float m_speedSum;
    float m_successRateSum;
    int m_episode;



    [Header("Walk Speed")]
    [Range(0.1f, m_maxWalkingSpeed)]
    [SerializeField]
    [Tooltip(
        "The speed the agent will try to match.\n\n" +
        "TRAINING:\n" +
        "For VariableSpeed envs, this value will randomize at the start of each training episode.\n" +
        "Otherwise the agent will try to match the speed set here.\n\n" +
        "INFERENCE:\n" +
        "During inference, VariableSpeed agents will modify their behavior based on this value " +
        "whereas the CrawlerDynamic & CrawlerStatic agents will run at the speed specified during training "
    )]
    //The walking speed to try and achieve
    private float m_TargetWalkingSpeed = m_maxWalkingSpeed;

    const float m_maxWalkingSpeed = 20; //The max walking speed

    //The current target walking speed. Clamped because a value of zero will cause NaNs
    public float TargetWalkingSpeed
    {
        get { return m_TargetWalkingSpeed; }
        set { m_TargetWalkingSpeed = Mathf.Clamp(value, .1f, m_maxWalkingSpeed); }
    }

    [Header("Ray Height Sensors")]
    [Range(0, 100)]
    [SerializeField]
    private int m_raySensors = 10;

    [Range(0, 5.0f)]
    [SerializeField]
    private float m_rayDistance = 1.0f;

    [SerializeField]
    LayerMask m_layerMaskDown;

    [SerializeField]
    LayerMask m_layerMaskUp;


    [SerializeField]
    private bool drawRayGizmosDown = false;
    [SerializeField]
    private bool drawRayGizmosUp = false;

    float m_rayMaxLength = 3f;


    float m_initDistanceToTarget;
    float m_lastDistanceToTarget;



    //The direction an agent will walk during training.
    [Header("Target To Walk Towards")]
    public bool useDynamicTarget = false;
    public Transform TargetPrefab; //Target prefab to use in Dynamic envs
    private Transform m_Target; //Target the agent will walk towards during training.

    [Header("Body Parts")][Space(10)] public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;

    //This will be used as a stabilized model space reference point for observations
    //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
    OrientationCubeController m_OrientationCube;

    //The indicator graphic gameobject that points towards the target
    DirectionIndicator m_DirectionIndicator;
    JointDriveController m_JdController;

    [Header("Foot Grounded Visualization")]
    [Space(10)]
    public bool useFootGroundedVisualization;

    public MeshRenderer foot0;
    public MeshRenderer foot1;
    public MeshRenderer foot2;
    public MeshRenderer foot3;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    public override void Initialize()
    {
        if (useDynamicTarget)
        {
            SpawnTarget(TargetPrefab, transform.position);
        }
        else
        {
            m_Target = TargetPrefab;
        }
        ////spawn target
        //Use Static target, included with the environment rather than spawning a new one


        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();
        m_JdController = GetComponent<JointDriveController>();

        //Setup each body part
        m_JdController.SetupBodyPart(body);
        m_JdController.SetupBodyPart(leg0Upper);
        m_JdController.SetupBodyPart(leg0Lower);
        m_JdController.SetupBodyPart(leg1Upper);
        m_JdController.SetupBodyPart(leg1Lower);
        m_JdController.SetupBodyPart(leg2Upper);
        m_JdController.SetupBodyPart(leg2Lower);
        m_JdController.SetupBodyPart(leg3Upper);
        m_JdController.SetupBodyPart(leg3Lower);


        m_stepsToTarget = 0;
        m_episode = 0;
    }

    /// <summary>
    /// Spawns a target prefab at pos
    /// </summary>
    /// <param name="prefab"></param>
    /// <param name="pos"></param>
    void SpawnTarget(Transform prefab, Vector3 pos)
    {
        m_Target = Instantiate(prefab, pos, Quaternion.identity, transform.parent);
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        //Workaround for logging metrics at the end of episode due to lack of event for episode ending (e.g. OnEpisodeEnd()).
        //Will miss the last episode of training unfortunately.
        if (m_episode != 0)
        {
            LogMetrics();
        }
        m_episode += 1;


        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        //Random start rotation to help generalize
        body.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        UpdateOrientationObjects();

        //Set our goal walking speed
        TargetWalkingSpeed = 20.0f;//Random.Range(0.1f, m_maxWalkingSpeed);

        m_initDistanceToTarget = (body.position - m_Target.position).magnitude;
        m_lastDistanceToTarget = m_initDistanceToTarget;

        m_stepsToTarget = 0;
        m_targetsCollected = 0;
        m_totalSteps = 0;
        m_speedSum = 0;
        m_successRateSum = 0;
    }

    void OnDrawGizmos()
    {
        if (drawRayGizmosDown || drawRayGizmosUp)
        {
            Vector3 origin = body.position;
            RaycastHit hit;
            for (int i = 0; i < m_raySensors; i++)
            {
                Vector3 offset = body.forward;
                float mul = m_raySensors / 2 - i - (1 - m_raySensors % 2) * 0.5f;
                offset *= m_rayDistance * mul;
                //DOWN Raycasts
                if (drawRayGizmosDown)
                {
                    origin.y = m_rayMaxLength;
                    if (Physics.Raycast(origin + offset, Vector3.down, out hit, Mathf.Infinity, m_layerMaskDown))
                    {
                        Handles.color = Color.red;
                        Handles.DrawLine(origin + offset, origin + offset + Vector3.down * hit.distance, 3);
                    }
                    else
                    {
                        Handles.color = Color.white;
                        Handles.DrawLine(origin + offset, origin + offset + Vector3.down * 10, 3);
                    }
                }

                //UP Raycasts
                if (drawRayGizmosUp)
                {
                    origin.y = 0f;
                    if (Physics.Raycast(origin + offset, Vector3.up, out hit, Mathf.Infinity, m_layerMaskUp))
                    {
                        Handles.color = Color.red;
                        Handles.DrawLine(origin + offset, origin + offset + Vector3.up * hit.distance, 3);
                    }
                    else
                    {
                        Handles.color = Color.white;
                        Handles.DrawLine(origin + offset, origin + offset + Vector3.up * 10, 3);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        //GROUND CHECK
        sensor.AddObservation(bp.groundContact.touchingGround); // Is this bp touching the ground

        if (bp.rb.transform != body)
        {
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    public void CollectRayObservations(VectorSensor sensor)
    {
        Vector3 origin = body.position;
        RaycastHit hit;
        for (int i = 0; i < m_raySensors; i++)
        {
            Vector3 offset = body.forward;
            float mul = m_raySensors / 2 - i - (1 - m_raySensors % 2) * 0.5f;
            offset *= m_rayDistance * mul;
            //DOWN Raycasts
            origin.y = m_rayMaxLength;
            if (Physics.Raycast(origin + offset, Vector3.down, out hit, m_rayMaxLength, m_layerMaskDown))
            {
                sensor.AddObservation(hit.distance / m_rayMaxLength);
            }
            else
            {
                sensor.AddObservation(1);
            }

            //UP Raycasts
            origin.y = 0f;
            if (Physics.Raycast(origin + offset, Vector3.up, out hit, m_rayMaxLength, m_layerMaskDown))
            {
                sensor.AddObservation(hit.distance / m_rayMaxLength);
            }
            else
            {
                sensor.AddObservation(1);
            }
        }
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        var cubeForward = m_OrientationCube.transform.forward;

        //velocity we want to match
        var velGoal = cubeForward;
        //ragdoll's avg vel
        var avgVel = GetAvgVelocity();

        //current ragdoll velocity. normalized
        //sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        //avg body vel relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel)); //3
        //vel goal relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal)); //3
        //rotation delta
        sensor.AddObservation(Quaternion.FromToRotation(body.forward, cubeForward)); //4

        //Add pos of target relative to orientation cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(m_Target.transform.position)); //3

        RaycastHit hit;
        float maxRaycastDist = 10;
        if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist))
        {
            sensor.AddObservation(hit.distance / maxRaycastDist); //1
        }
        else
            sensor.AddObservation(1);

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }

        //31 Obs so far

        //Raycast sensors
        CollectRayObservations(sensor); //2*m_numRaySensors


    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // The dictionary with all the body parts in it are in the jdController
        var bpDict = m_JdController.bodyPartsDict;

        var continuousActions = actionBuffers.ContinuousActions;
        var i = -1;

        // Pick a new target joint rotation
        bpDict[leg0Upper].SetJointTargetRotation(Math.Clamp(continuousActions[++i], -1.0f, 1.0f), Math.Clamp(continuousActions[++i], -1.0f, 1.0f), 0);
        bpDict[leg1Upper].SetJointTargetRotation(Math.Clamp(continuousActions[++i], -1.0f, 1.0f), Math.Clamp(continuousActions[++i], -1.0f, 1.0f), 0);
        bpDict[leg2Upper].SetJointTargetRotation(Math.Clamp(continuousActions[++i], -1.0f, 1.0f), Math.Clamp(continuousActions[++i], -1.0f, 1.0f), 0);
        bpDict[leg3Upper].SetJointTargetRotation(Math.Clamp(continuousActions[++i], -1.0f, 1.0f), Math.Clamp(continuousActions[++i], -1.0f, 1.0f), 0);
        bpDict[leg0Lower].SetJointTargetRotation(Math.Clamp(continuousActions[++i], -1.0f, 1.0f), 0, 0);
        bpDict[leg1Lower].SetJointTargetRotation(Math.Clamp(continuousActions[++i], -1.0f, 1.0f), 0, 0);
        bpDict[leg2Lower].SetJointTargetRotation(Math.Clamp(continuousActions[++i], -1.0f, 1.0f), 0, 0);
        bpDict[leg3Lower].SetJointTargetRotation(Math.Clamp(continuousActions[++i], -1.0f, 1.0f), 0, 0);

        // Update joint strength
        bpDict[leg0Upper].SetJointStrength(Math.Clamp(continuousActions[++i], -1.0f, 1.0f));
        bpDict[leg1Upper].SetJointStrength(Math.Clamp(continuousActions[++i], -1.0f, 1.0f));
        bpDict[leg2Upper].SetJointStrength(Math.Clamp(continuousActions[++i], -1.0f, 1.0f));
        bpDict[leg3Upper].SetJointStrength(Math.Clamp(continuousActions[++i], -1.0f, 1.0f));
        bpDict[leg0Lower].SetJointStrength(Math.Clamp(continuousActions[++i], -1.0f, 1.0f));
        bpDict[leg1Lower].SetJointStrength(Math.Clamp(continuousActions[++i], -1.0f, 1.0f));
        bpDict[leg2Lower].SetJointStrength(Math.Clamp(continuousActions[++i], -1.0f, 1.0f));
        bpDict[leg3Lower].SetJointStrength(Math.Clamp(continuousActions[++i], -1.0f, 1.0f));


        /* bpDict[leg0Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg1Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg2Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg3Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg0Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg1Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg2Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg3Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);

        // Update joint strength
        bpDict[leg0Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg0Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Lower].SetJointStrength(continuousActions[++i]); */


    }

    void FixedUpdate()
    {
        UpdateOrientationObjects();

        // If enabled the feet will light up green when the foot is grounded.
        // This is just a visualization and isn't necessary for function
        if (useFootGroundedVisualization)
        {
            foot0.material = m_JdController.bodyPartsDict[leg0Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot1.material = m_JdController.bodyPartsDict[leg1Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot2.material = m_JdController.bodyPartsDict[leg2Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot3.material = m_JdController.bodyPartsDict[leg3Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
        }

        var cubeForward = m_OrientationCube.transform.forward;

        //Approaches 1 when looking at target, 0 when looking 180 degrees away
        var lookAtTargetReward = (Vector3.Dot(cubeForward, body.forward) + 1) * .5F;

        float distToTarget = (body.position - m_Target.position).magnitude;
        float deltaDistance = m_lastDistanceToTarget - distToTarget;
        float deltaDistanceNorm = deltaDistance / m_initDistanceToTarget;
        m_lastDistanceToTarget = distToTarget;


        //Debug.Log(deltaDistanceNorm);

        //Add reward for moving towards the target, normalized such that it adds up to 1 over the whole episode if target is touched
        //AddReward(deltaDistanceNorm);

        //Add reward for looking in the right direction. Requires movement so that the agent does not simply stand still 
        // Might have problem if the agent learns to look away, walk away from target and then walk in right direction? Maybe penalize more if moving backwards?
        /*if (deltaDistance < 0)
        {
            AddReward(deltaDistanceNorm);
        }
        else
        {
            AddReward(lookAtTargetReward * deltaDistanceNorm);
        }*/



        // Velocity of agent multiplied by cos of angle between target and velocity
        // i.e. reward 0 for moving 90 degrees from the target vector, -1*velocity for moving directly away and 1*velocity for moving towards target
        // Divide by 15 to scale the reward to a more appropriate value
        //var velReward = Vector3.Dot(cubeForward, GetAvgVelocity()) / 15.0f;
        //AddReward(velReward);

        //var velReward = GetAvgVelocity().magnitude / 50.0f;
        //AddReward(velReward * lookAtTargetReward);



        // Original agent reward
        var matchSpeedReward = GetMatchingVelocityReward(cubeForward * TargetWalkingSpeed, GetAvgVelocity());
        AddReward(matchSpeedReward * lookAtTargetReward);

        m_stepsToTarget += 1;
    }

    /// <summary>
    /// Update OrientationCube and DirectionIndicator
    /// </summary>
    void UpdateOrientationObjects()
    {
        m_OrientationCube.UpdateOrientation(body, m_Target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }

    /// <summary>
    ///Returns the average velocity of all of the body parts
    ///Using the velocity of the body only has shown to result in more erratic movement from the limbs
    ///Using the average helps prevent this erratic movement
    /// </summary>
    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;
        Vector3 avgVel = Vector3.zero;

        //ALL RBS
        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.velocity;
        }

        avgVel = velSum / numOfRb;
        return avgVel;
    }

    /// <summary>
    /// Normalized value of the difference in actual speed vs goal walking speed.
    /// </summary>
    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        //distance between our actual velocity and goal velocity
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetWalkingSpeed);

        //return the value on a declining sigmoid shaped curve that decays from 1 to 0
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / TargetWalkingSpeed, 2), 2);
    }

    /// <summary>
    /// Agent touched the target
    /// </summary>
    public void TouchedTarget()
    {
        if (!endEPOnGoal)
        {
            AddMetrics();
        }


        m_initDistanceToTarget = (body.position - m_Target.position).magnitude;
        m_lastDistanceToTarget = m_initDistanceToTarget;

        //AddReward(1f);

        if (endEPOnGoal)
        {
            EndEpisode();
        }

    }

    /// <summary>
    /// In case of episode not ending when touching the target.
    /// We want to keep track of each travel to the target and average it in the end.
    /// </summary>
    public void AddMetrics()
    {
        // Possible for the target to spawn under the agent -> no steps between targets
        if (m_stepsToTarget != 0)
        {
            m_targetsCollected += 1;
            float distTravelled = m_initDistanceToTarget;
            m_speedSum += distTravelled / m_stepsToTarget;
            m_successRateSum += 1;
            m_totalSteps += m_stepsToTarget;
        }

        m_stepsToTarget = 0;
    }

    public void LogMetrics()
    {
        //Metrics logging
        float distTravelled = m_initDistanceToTarget - m_lastDistanceToTarget;

        // Ignore last target metrics IF ends due to reaching max number of steps and episode doesnt end with touching the target
        // Could have some issues or bias?
        if (m_totalSteps == MaxStep)
        {
            // TODO?
        }

        m_speedSum += distTravelled / m_stepsToTarget;
        m_successRateSum += distTravelled / m_initDistanceToTarget;
        m_totalSteps += m_stepsToTarget;

        float avgSpeed = m_speedSum / (m_targetsCollected + 1);
        float avgSuccessRate = m_successRateSum / (m_targetsCollected + 1);
        float avgSteps = m_totalSteps / (m_targetsCollected + 1);

        metricsLogger.Record(m_episode, m_totalSteps, m_successRateSum, m_speedSum, m_targetsCollected);
    }
}
