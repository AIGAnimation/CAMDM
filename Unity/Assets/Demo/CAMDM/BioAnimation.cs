using UnityEngine;
using System;
using System.Collections.Generic;
using Unity.Sentis;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using System.Diagnostics;
using System.IO;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace CAMDM {
	[RequireComponent(typeof(Actor))]
	public class BioAnimation : MonoBehaviour
	{
		public ModelAsset modelPath;
		public TextAsset modelConfig;
		public TextAsset conditionConfig;
		public BackendType device = BackendType.GPUCompute;
		public DiffusionNetwork diffusionNetwork;
		
		private Quaternion traj_pose_quaterniaon;
		private float[] traj_pose_6D;
		public Vector3 base_trans = Vector3.zero;

		private string[] styleList;
		private int CurrentStyleIdx;
		
		private bool forward = true;//the mode of the orientation 

		public bool Inspect = false;
		public bool ShowInformation = true; // waring: May cause memory leaks if ture
		public bool ShowTrajectory = true;
		public bool Show_hand_trajectory=false;
		public bool Show_inference_time=false;

		private float speed; // m/s
		public float SprintTarget_speed = 2.5f; // the speed of runing which can be adjusted by hand
		public float WalkTarget_speed = 1f;     // the speed of walking which can be adjusted by hand
		public float Target_speed;              // target speed   if LeftShift  is pressed -> Target_speed = SprintTarget_speed 
										//                       else 				          Target_speed = WalkTarget_speed
		public float TargetGain = 0.112f;
		public float TargetDecay = 0.017f;

		public Controller Controller;
		private Actor Actor;
		// public BVHExporter BVHExporter; //保存实时推理时的运动数据为bvh格式，方便在blender软件中的观看和指标评估
		public InertiaAlgorithm InertiaAlgorithm;
		private Trajectory Trajectory;
		private Trajectory Trajectory_right_hand;
		private Trajectory Trajectory_root;
		private Trajectory Trajectory_left_hand;
		
		private Vector3 TargetDirection;
		private Vector3 TargetVelocity;
		private Vector3 TargetVelocity1;
    

		//Trajectory for 60 Hz framerate
		private const int Framerate = 30;
		private const int Points = 90;
		private const int PointSamples = 12;
		private const int PastPoints = 10;
		private const int FuturePoints = 45;
		private const int RootPointIndex = 44;
		
		private int PelvisIndex = 0;
		
		public bool Inertialize = true;
		
		private TensorFloat past_motion; 
        private TensorFloat traj_pose; 
		private TensorFloat traj_trans;
		private TensorFloat output;
		private TensorFloat styleFeature = TensorFloat.Zeros(new TensorShape(1));
		
        
		private Quaternion quat2 = new Quaternion(1, 0, 0, 0); // 创建一个四元数
		private Quaternion quat3 = new Quaternion(0, -1f, 0, 0); // 创建一个四元数 debug时发现unity的四元数如果等于这个值会发生跳变，变成quat2，造成不连续
		
		private bool control;
		private Vector3[] trajectory_positions_blend = new Vector3[(Points+FuturePoints-1)];
		private Vector3[] trajectory_positions_synthesis = new Vector3[Points];


		private int frame = 0; // current applied frame
		private Transform from;
		private Transform localAxisTransform;
		private Transform joint_transform;
		private Vector3 row1;
		private Vector3 row2;
		private Quaternion localrotation;
		private Matrix4x4 localMatrix;
		private float[] rotation6D;

		private Vector3 idle_position; 
		public float blendTime_rotation = 0.2f;
		public float blendTime_position = 0.2f;
		public int applyframes=15; // the prescribed frames to be applied
		private int applyframes_cache = 15;
		public float bias_pos =1.6f;
		public float bias_dir =2.2f;  
		public float bias_HFTE =0.4f;
		private float Max_angle =145f;
		public bool HFTE = true;

		public float CFG_weight =0.5f;
		public int CFGcount = 2; // 当风格发生变化时，执行CFG的持续时间。
		                         // 因为Pytorch训练模型时随机置空了past motion，所以网络不会过渡信赖（或者过拟合到)past motion。如果不这样做，styleFeature会失去作用，我们认为这是因为past motion的特征和future motion特征很像，容易造成过拟合。
								 // 推理时，令人意外的是，我们发现即使不使用CFG，网络在大部分风格间也能进行切换，只是切换的响应程度比较低，这可能与训练时随机置空past motion让其不过渡依赖past motion有关。					
								//  为了百分百地、快速地切换风格，我们默认在风格切换时开启CFG
		public int CFGcount_cache = 2;
		Stopwatch sw = new Stopwatch();//计时器
		private float inference_time =0f;
		// define the class of inertia blending 
		struct TransformState
        {

            public float3 PreviousPosition;
            public float3 CurrentPosition;
			public float3 TargetPosition;

            public Quaternion PreviousLocalRotation;
            public Quaternion CurrentLocalRotation;
			public Quaternion TargetLocalRotation;
			
            // public EvaluateSpace Space;
            public float BeginTime;
            public float EndTime;
        }

		private TransformState[] States = new TransformState[25];

		void Reset() {
			Controller = new Controller();
		}

		void Awake() {
			Actor = GetComponent<Actor>();
			InertiaAlgorithm = GetComponent<InertiaAlgorithm>();

			from= new GameObject("MyTransform").transform;// HFTE建立局部坐标系时需要用到
			localAxisTransform = new GameObject("Local Axis").transform;// HFTE建立局部坐标系时需要用到
			joint_transform = new GameObject("MyTransform1").transform;

			TargetDirection = new Vector3(transform.forward.x, 0f, transform.forward.z);
			traj_pose_quaterniaon = Quaternion.FromToRotation(Vector3.forward,transform.forward );
			traj_pose_6D = QuaternionToRotation6D(traj_pose_quaterniaon);

			TargetVelocity = Vector3.zero;
			TargetVelocity1 = Vector3.zero;

			Trajectory = new Trajectory(Points, Controller.GetNames(), transform.position, TargetDirection);

			Trajectory_right_hand = new Trajectory(100, Controller.GetNames(), transform.position, TargetDirection);
			Trajectory_left_hand = new Trajectory(100, Controller.GetNames(), transform.position, TargetDirection);
			// Trajectory_root = new Trajectory(100, Controller.GetNames(), transform.position, TargetDirection);
			
			if(Controller.Styles.Length > 0) {
				for(int i=0; i<Trajectory.Points.Length; i++) {
					Trajectory.Points[i].Styles[0] = 1f;
				}
			}
			
			past_motion = TensorFloat.Zeros(new TensorShape(1, Actor.Bones.Length, 6, PastPoints));
			traj_pose = TensorFloat.Zeros(new TensorShape(1, 6, FuturePoints)); 
			traj_trans = TensorFloat.Zeros(new TensorShape(1, 2,FuturePoints)); 
			output = TensorFloat.Zeros(new TensorShape(1, Actor.Bones.Length, 6, FuturePoints));
			
			diffusionNetwork = new DiffusionNetwork();
			diffusionNetwork.CreateSession(modelPath, device, modelConfig, conditionConfig);
			styleList = diffusionNetwork.styles;
			
			PelvisIndex = Actor.Bones.Length - 1;
			
			if(Controller.Styles.Length > 0) {
				for(int i=0; i<Trajectory.Points.Length; i++) {
					Trajectory.Points[i].Styles[0] = 0f;
					Trajectory.Points[i].Styles[1] = 1f;
				}
			}
			
			for(int i=0; i<Actor.Bones.Length-1; i++) {
				States[i].BeginTime = Time.time;
				States[i].EndTime = States[i].BeginTime + blendTime_rotation;
				if(i==0)
				{
					States[i].PreviousPosition = Actor.Bones[i].Transform.position;
					States[i].CurrentPosition = Actor.Bones[i].Transform.position;
					States[i].EndTime = States[i].BeginTime + blendTime_position;
				}
				States[i].PreviousLocalRotation = Actor.Bones[i].Transform.localRotation;
				States[i].CurrentLocalRotation = Actor.Bones[i].Transform.localRotation;
			}
		}
		
		void Start() {
			Utility.SetFPS(60);
			Time.fixedDeltaTime =1/35f;
		}

		private void Update()
		{
			Target_speed = WalkTarget_speed;
			if (Input.GetKey(KeyCode.LeftShift))
			{
				Target_speed = SprintTarget_speed;
			}
			
			if (Input.GetKey(KeyCode.LeftShift))
			{
				Target_speed = SprintTarget_speed;
			}
			
			if(Input.GetKeyDown(KeyCode.F))
			{
				forward = !forward ;
			}

			if(Input.GetKeyDown(KeyCode.N))
			{
				frame=0;
				base_trans= Actor.Bones[PelvisIndex].Transform.position;
				for(int i=0; i<Actor.Bones.Length-1; i++) 
				{
					if(frame == (applyframes-1))
					{
						States[i].BeginTime = Time.time;
						States[i].EndTime = States[i].BeginTime + blendTime_rotation;
						if(i==0)
						{
							States[i].EndTime = States[i].BeginTime + blendTime_position;
						}
						
					}
				}
				applyframes_cache +=5 ;
			}
			if(Input.GetKeyDown(KeyCode.M))
			{
				frame=0;
				base_trans= Actor.Bones[PelvisIndex].Transform.position;
				for(int i=0; i<Actor.Bones.Length-1; i++) 
				{
					if(frame == (applyframes-1))
					{
						States[i].BeginTime = Time.time;
						States[i].EndTime = States[i].BeginTime + blendTime_rotation;
						if(i==0)
						{
							States[i].EndTime = States[i].BeginTime + blendTime_position;
						}
					}
				}
				applyframes_cache -=5 ;
			}
			if(applyframes_cache>=30) //大于30的值可能会造成卡顿
			{
				applyframes_cache =30;
			}
			if(applyframes_cache<=3) // (3060GPU) Note!!!! if the applied frames<=3, the caculation of the speed will be wrong because of the frequent inference of diffusion model, leading to drift when standing.
			{
				applyframes_cache =4;
			}
			
			int availableNum = styleList.Length;
			if (Input.GetKeyDown(KeyCode.J))
			{
				CurrentStyleIdx =(CurrentStyleIdx - 1 + availableNum) % availableNum;
				print("Press J");
				CFGcount = CFGcount_cache;
			}
	            
			if (Input.GetKeyDown(KeyCode.L))
			{
				CurrentStyleIdx = (CurrentStyleIdx + 1) % availableNum;
				print("Press L");
				CFGcount = CFGcount_cache;
			}
			applyframes = applyframes_cache;
		}

		void FixedUpdate() {
			PredictTrajectory();
			Animate();
		}
		
		private void PredictTrajectory() {
			
			float turn = Controller.QueryTurn();
			Vector3 move = Controller.QueryMove();
			
			control = turn != 0f || move != Vector3.zero;
			speed = Utility.Interpolate(speed, (Trajectory.Points[RootPointIndex].GetPosition()-Trajectory.Points[RootPointIndex-1].GetPosition()).magnitude / Time.fixedDeltaTime,control?0.05f:0.5f); // (3060GPU) Note!!!! if the applied frames<=3, the caculation of the speed will be wrong because of the frequent inference of diffusion model, leading to drift when standing.
			TargetVelocity = Vector3.Lerp(TargetVelocity, Target_speed * (Quaternion.LookRotation( Vector3.forward, Vector3.up) * move).normalized, control ? TargetGain : TargetDecay);   //Quaternion.LookRotation(TargetDirection, Vector3.up) * move是为了从世界坐标系的move方向旋转到自身坐标系的move方向

			if(forward ==true )
			{
				float angle = Vector3.SignedAngle(TargetDirection,TargetVelocity ,Vector3.up);
				if (control == true)
				{
					TargetDirection = Trajectory.Points[RootPointIndex].GetDirection();

					if(angle>Max_angle)
					{
						TargetVelocity = Target_speed*   (Quaternion.AngleAxis(Max_angle, Vector3.up) * TargetDirection);
					}
					else if(angle<-Max_angle)
					{
						TargetVelocity = Target_speed* (Quaternion.AngleAxis(-Max_angle, Vector3.up) * TargetDirection);
					}
				}
			}
			else
			{
				TargetDirection = Vector3.Lerp(TargetDirection, Quaternion.AngleAxis(turn * 60f, Vector3.up) * TargetDirection, control ? TargetGain : TargetDecay).normalized;
			}
			
	
			if (control == true)
			{
				TargetVelocity1=TargetVelocity;
			}
			else
			{
				TargetVelocity1 = TargetDirection*0.01f;
			}
			
			trajectory_positions_blend[RootPointIndex] = Trajectory.Points[RootPointIndex].GetPosition();
			trajectory_positions_synthesis[RootPointIndex] = Trajectory.Points[RootPointIndex].GetPosition();
			if(speed>0.1f) // 若速度大于这个经验值，则时刻更新idle position,但请注意如果GPU性能不行时，且应用的帧数(applyframes)很少的话，speed算出来的值是不准确的，这会导致这个经验值失去作用而出现静止时原地漂移的情况   (3060GPU) Note!!!! if the applied frames<=3, the caculation of the speed will be wrong because of the frequent inference of diffusion model, leading to drift when standing.
			{
				idle_position = trajectory_positions_blend[RootPointIndex];
			}

			
			if(control==false && speed<=0.1)
			{
				TargetVelocity1=(idle_position-Trajectory.Points[RootPointIndex].GetPosition());
			}
				
			float scale=  (1f / Framerate);
			
			for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {

				float Target_speed_vel = 1.50f;
				float weight = (float)(i - RootPointIndex) / (float)FuturePoints;
				float scale_pos = 1.0f - Mathf.Pow(1.0f - weight, bias_pos);   
				float scale_pos1 = 1.0f - Mathf.Pow(1.0f - weight, bias_HFTE);  
				float scale_dir = 1.0f - Mathf.Pow(1.0f - weight, bias_dir);
				
				trajectory_positions_synthesis[i] = trajectory_positions_synthesis[i-1]  + scale * TargetVelocity1;
				trajectory_positions_blend[i] =
						Vector3.Lerp(
							Trajectory.Points[i].GetPosition(),  
							trajectory_positions_synthesis[i],
							HFTE?scale_pos1:1f
						);

				if (control==true )
				{
					Trajectory.Points[i].SetPosition(trajectory_positions_blend[i]);
				}
				else
				{
					if(speed<0.1) //速度小于这个经验值说明这时应该是idle状态，即原地静止不同状态
						Trajectory.Points[i].SetPosition(Vector3.Lerp(trajectory_positions_blend[i],idle_position,scale_pos)); 
					else
						Trajectory.Points[i].SetPosition(trajectory_positions_blend[i]);
				}


				if (forward == true && control == true)
				{
					Trajectory.Points[i].SetDirection(Vector3.Lerp(Trajectory.Points[i].GetDirection(),
							(trajectory_positions_synthesis[i] - trajectory_positions_synthesis[i - 1]).normalized,
							scale_dir));
				}
				else
				{
					Trajectory.Points[i].SetDirection(Vector3.Lerp(Trajectory.Points[i].GetDirection(),  TargetDirection.normalized,HFTE?scale_dir:1f).normalized);
				}

				traj_trans[0,0,i-RootPointIndex-1]=-(Trajectory.Points[i].GetPosition().x-base_trans.x);
				traj_trans[0,1,i-RootPointIndex-1]=(Trajectory.Points[i].GetPosition().z-base_trans.z);
				traj_pose_quaterniaon = Quaternion.FromToRotation(Vector3.forward, Trajectory.Points[i].GetDirection().normalized).normalized;
				traj_pose_quaterniaon.y=-traj_pose_quaterniaon.y;

				if (traj_pose_quaterniaon.Equals(quat2))  // 四元数突变检测
				{
					traj_pose_quaterniaon = quat3;
				}

				traj_pose_6D = QuaternionToRotation6D(traj_pose_quaterniaon);
				
				for (int j = 0; j < 6; j++)
				{
					traj_pose[0, j, i-RootPointIndex-1] = traj_pose_6D[j];
				}

			}
		}
	
		private void Animate() {

			Vector3 a1;
			Vector3 a2;
			Matrix4x4 Matrix;
			
			styleFeature[0] = (float)CurrentStyleIdx;
			if(frame==0)
			{
				output.Dispose();
				if(Show_inference_time)
				{
					sw.Start();
				}
				if (CFGcount > 0)
				{
					output = diffusionNetwork.Inference(past_motion, traj_pose, traj_trans, styleFeature, CFG_weight); 
					CFGcount--;
				}
				else
				{
				    output = diffusionNetwork.Inference(past_motion, traj_pose, traj_trans,styleFeature, 1f);
				}
				if(Show_inference_time)
				{
					inference_time= sw.ElapsedMilliseconds;
					sw.Reset();
				}
				output.MakeReadable();
				localAxisTransform.position =  new Vector3(-output[0,PelvisIndex,0,FuturePoints-1], 0f, output[0,PelvisIndex,2,FuturePoints-1]) + base_trans;
				Vector3 unitVector = new Vector3(-(output[0,PelvisIndex,0,FuturePoints-1]-output[0,PelvisIndex,0,FuturePoints-2]), 0f, output[0,PelvisIndex,2,0]-output[0,PelvisIndex,2,FuturePoints-2]).normalized;
				localAxisTransform.rotation = Quaternion.LookRotation(unitVector, Vector3.up);
				Vector3 localPosition;
				Vector3 symmetricLocalPositionxz;
				Vector3 symmetricWorldPositionxz;
				Vector3 symmetricLocalPositionx;
				for(int i=Points; i<Points+FuturePoints-1; i++)
				{
					localPosition = localAxisTransform.InverseTransformPoint(new Vector3(-output[0,PelvisIndex,0,FuturePoints-(i-Points+2)], 0f, output[0,PelvisIndex,2,FuturePoints-(i-Points+2)]) + base_trans);
					symmetricLocalPositionxz = new Vector3(-localPosition.x, localPosition.y, -localPosition.z);
					symmetricWorldPositionxz = localAxisTransform.TransformPoint(symmetricLocalPositionxz);
					trajectory_positions_blend[i] = symmetricWorldPositionxz;
				}
				
				localAxisTransform.position = trajectory_positions_blend[Points+((FuturePoints-1)/2)];
				unitVector = (trajectory_positions_blend[Points+((FuturePoints-1)/2)]- trajectory_positions_blend[Points+((FuturePoints-1)/2)-1]).normalized;
				localAxisTransform.rotation = Quaternion.LookRotation(unitVector, Vector3.up);
				for(int i=Points+((FuturePoints-1)/2)+1; i<Points+FuturePoints-1; i++)
				{
						localPosition = localAxisTransform.InverseTransformPoint(trajectory_positions_blend[Points+((FuturePoints-1)/2) - (i-Points-((FuturePoints-1)/2))]);
						symmetricLocalPositionxz = new Vector3(-localPosition.x, localPosition.y, -localPosition.z);
						symmetricWorldPositionxz = localAxisTransform.TransformPoint(symmetricLocalPositionxz);
						trajectory_positions_blend[i] = symmetricWorldPositionxz;
				}
			}
			// Update the past Trajectory
			for(int i=0; i<RootPointIndex; i++) {
				Trajectory.Points[i].SetPosition(Trajectory.Points[i+1].GetPosition());
				Trajectory.Points[i].SetDirection(Trajectory.Points[i+1].GetDirection());
			}
			if(Show_hand_trajectory)
			{
				for(int i=0; i<99; i++) {
					Trajectory_right_hand.Points[i].SetPosition(Trajectory_right_hand.Points[i+1].GetPosition());
					Trajectory_left_hand.Points[i].SetPosition(Trajectory_left_hand.Points[i+1].GetPosition());
					// Trajectory_root.Points[i].SetPosition(Trajectory_root.Points[i+1].GetPosition());
				}
			}
			//载入HFTE构造出来的轨迹和朝向    load the extended trajetory and orientation caculated by HFTE
			for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {

				if(i-RootPointIndex+frame<FuturePoints)
				{
					Trajectory.Points[i].SetPosition( new Vector3(-(output[0,PelvisIndex,0,i-RootPointIndex+frame]), 0f, output[0,PelvisIndex,2,i-RootPointIndex+frame]) + base_trans);   //第15帧预测的结果之后要替换当前帧，因此第15帧的轨迹应该是第16，以此类推，否则第14帧和第15帧的轨迹会重合
					a1 = new Vector3(output[0,0,0,i-RootPointIndex+frame], output[0,0,1,i-RootPointIndex+frame], output[0,0,2,i-RootPointIndex+frame]); 
					a2 = new Vector3(output[0,0,3,i-RootPointIndex+frame], output[0,0,4,i-RootPointIndex+frame], output[0,0,5,i-RootPointIndex+frame]);
				
					Matrix= Convert(a1, a2);
					from.rotation = MatrixToQuaternion(Matrix);
					Trajectory.Points[i].SetDirection(new Vector3(from.forward.x,0,from.forward.z).normalized);
				}
				else
				{
					Trajectory.Points[i].SetPosition( trajectory_positions_blend[(i-RootPointIndex+frame-FuturePoints)+Points]);
					Trajectory.Points[i].SetDirection(Trajectory.Points[i-1].GetDirection());
				}
			}
			//载入网络推理出来的运动数据到untiy的角色骨架上   load the model-predicted motion data to the character skelecton
			float dt =  Time.deltaTime;
			float factor =  dt /(1f / Framerate) ;
			for(int i=0; i<Actor.Bones.Length-1; i++) 
			{		
				for (int k=0; k<PastPoints-1;k++)
				{			
					for (int j = 0; j < 6; j++)
					{							
							past_motion[0, i, j, k] = past_motion[0, i, j, k+1];
					}
								
				}
				float tf = max(0.0001f, States[i].EndTime - Time.time);
				// inertialization
				if(i==0)
				{
					if(Inertialize)
					{
						States[0].TargetPosition = new Vector3(-output[0,PelvisIndex,0,frame], output[0,PelvisIndex,1,frame], output[0,PelvisIndex,2,frame]) + base_trans;
						Actor.Bones[0].Transform.position = InertiaAlgorithm.Inertialize_Position(States[0].PreviousPosition,States[0].CurrentPosition, States[0].TargetPosition, dt, tf, dt);
						States[0].PreviousPosition = States[0].CurrentPosition;
						States[0].CurrentPosition = Actor.Bones[0].Transform.position;
					}
					else
					{
						Vector3 pelvis_position = Actor.Bones[0].Transform.position;

						Actor.Bones[0].Transform.position=new Vector3(-output[0,PelvisIndex,0,frame], output[0,PelvisIndex,1,frame], output[0,PelvisIndex,2,frame]) + base_trans;
						Actor.Bones[0].Transform.position = Vector3.Lerp(pelvis_position,Actor.Bones[0].Transform.position,0.5f);
						States[0].PreviousPosition = States[0].CurrentPosition;
						States[0].CurrentPosition = Actor.Bones[0].Transform.position;
					}
				}

				a1 = new Vector3(output[0,i,0,frame], output[0,i,1,frame], output[0,i,2,frame]);
				a2 = new Vector3(output[0,i,3,frame], output[0,i,4,frame], output[0,i,5,frame]);
				
				Matrix= Convert(a1, a2);

				if(Inertialize)
				{

					States[i].TargetLocalRotation = MatrixToQuaternion(Matrix);
			
					Actor.Bones[i].Transform.localRotation = InertiaAlgorithm.Inertialize_Rotation(States[i].PreviousLocalRotation,States[i].CurrentLocalRotation, States[i].TargetLocalRotation, dt, tf, dt);
					States[i].PreviousLocalRotation = States[i].CurrentLocalRotation;
					States[i].CurrentLocalRotation = Actor.Bones[i].Transform.localRotation;
				}
				else
				{
					joint_transform.localRotation = Actor.Bones[i].Transform.localRotation;
					Actor.Bones[i].Transform.localRotation = MatrixToQuaternion(Matrix);
					if(Quaternion.Dot(joint_transform.localRotation, Actor.Bones[i].Transform.localRotation)<0)
					{
						Actor.Bones[i].Transform.localRotation=new Quaternion(-Actor.Bones[i].Transform.localRotation.x,-Actor.Bones[i].Transform.localRotation.y,-Actor.Bones[i].Transform.localRotation.z,-Actor.Bones[i].Transform.localRotation.w);
					}

					Actor.Bones[i].Transform.localRotation = Quaternion.Slerp(joint_transform.localRotation, Actor.Bones[i].Transform.localRotation, 0.5f);
					States[i].PreviousLocalRotation = States[i].CurrentLocalRotation;
					States[i].CurrentLocalRotation = Actor.Bones[i].Transform.localRotation;
				}

				localrotation = Actor.Bones[i].Transform.localRotation;
				localrotation.y =-localrotation.y;
				localrotation.z =-localrotation.z;
				localMatrix = Matrix4x4.Rotate(localrotation);	
				row1 = localMatrix.GetRow(0);
				row2 = localMatrix.GetRow(1);

				rotation6D = new float[6] {row1.x, row1.y, row1.z, row2.x, row2.y, row2.z};

				for (int j = 0; j < 6; j++)
				{
						past_motion[0, i, j, PastPoints-1] = rotation6D[j];
				}

				if(frame == (applyframes-1) && Inertialize==true)
				{
					States[i].BeginTime = Time.time;
					States[i].EndTime = States[i].BeginTime + blendTime_rotation;
					if(i==0)
					{
						States[i].EndTime = States[i].BeginTime + blendTime_position;
					}
					
				}
			}

			for(int i=0; i<PastPoints-1; i++) 
			{
				past_motion[0, PelvisIndex, 1, i] = past_motion[0,PelvisIndex,1,i+1];
			}

			Actor.Bones[PelvisIndex].Transform.position= new Vector3(Actor.Bones[0].Transform.position.x, 0f, Actor.Bones[0].Transform.position.z); // 更新alpha_surface的三维坐标
		
			Trajectory.Points[RootPointIndex].SetPosition(Actor.Bones[PelvisIndex].Transform.position);   //更新当前帧轨迹点的坐标

			if(Show_hand_trajectory)
			{
			Trajectory_right_hand.Points[99].SetPosition(Actor.Bones[10].Transform.position);
			Trajectory_left_hand.Points[99].SetPosition(Actor.Bones[14].Transform.position);
			Trajectory_root.Points[99].SetPosition(Actor.Bones[23].Transform.position);
			}

			Trajectory.Points[RootPointIndex].SetDirection(new Vector3(Actor.Bones[0].Transform.forward.x,0,Actor.Bones[0].Transform.forward.z).normalized);
			Trajectory.Points[RootPointIndex].SetDirection(Vector3.Lerp(Trajectory.Points[RootPointIndex].GetDirection(),  TargetDirection.normalized, 0f).normalized);
			
			past_motion[0, PelvisIndex, 0, PastPoints-1] = 0f;
			past_motion[0, PelvisIndex, 1, PastPoints-1] = Actor.Bones[0].Transform.position.y;  //当前帧的pelvis坐标只用更新y，因为x,z一直都是0
			past_motion[0, PelvisIndex, 2, PastPoints-1] = 0f;
			
			if( frame ==(applyframes-1))
			{
				base_trans= Actor.Bones[PelvisIndex].Transform.position;
			}
			
			for(int i=RootPointIndex-PastPoints+1; i<RootPointIndex; i++) 
			{

				past_motion[0, PelvisIndex, 0, i-(RootPointIndex-PastPoints+1)] = -(Trajectory.Points[i].GetPosition().x - base_trans.x);
				past_motion[0, PelvisIndex, 2, i-(RootPointIndex-PastPoints+1)] = (Trajectory.Points[i].GetPosition().z - base_trans.z);
			}

			frame++;
			if(frame>(applyframes-1))
			{
				frame=0;
			}

		}
		
		private Quaternion MatrixToQuaternion(Matrix4x4 m) //pytroch训练数据是opengl 右手坐标系，但是unity是左手坐标系，所以得对网络预测出来的四元数进行坐标系转换
		{
			Quaternion q = new Quaternion();
			q = Quaternion.LookRotation(m.GetColumn(2).normalized,m.GetColumn(1).normalized);
			q.y = -q.y;
			q.z = -q.z;
			q = q.normalized;

			return q;
		}

		private Matrix4x4 Convert(Vector3 a1, Vector3 a2)
		{
			Vector3 b1 = a1.normalized;
			Vector3 b2 = a2 - Vector3.Dot(a1, a2) * b1;
			b2=b2.normalized;
			Vector3 b3 = Vector3.Cross(b1, b2);
			b3=b3.normalized;

			Matrix4x4 rotationMatrix = new Matrix4x4();
			rotationMatrix.SetRow(0, b1);
			rotationMatrix.SetRow(1, b2);
			rotationMatrix.SetRow(2, b3);
			rotationMatrix[3, 3] = 1f;

			return rotationMatrix;
		}
		private float[] QuaternionToRotation6D(Quaternion rotation)
		{
			// 将四元数转换为旋转矩阵
			Matrix4x4 matrix = Matrix4x4.Rotate(rotation);
			
			// 提取前两行，得到一个2x3矩阵
			Vector3 row1 = matrix.GetRow(0);
			Vector3 row2 = matrix.GetRow(1);

			// 将2x3矩阵转换为6D向量
			float[] rotation6D = new float[6] {row1.x, row1.y, row1.z, row2.x, row2.y, row2.z};
			return rotation6D;
		}

		void OnRenderObject() {
			if(Application.isPlaying) {
				if(ShowTrajectory) {
					UltiDraw.Begin();
					UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetDirection, 0.05f, 0f, UltiDraw.Red.Transparent(0.75f));  //Target orientation
					UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetVelocity, 0.05f, 0f, UltiDraw.Green.Transparent(0.75f)); //Target trajetory direction(user control)
					for(int i=RootPointIndex+1; i<Trajectory.Points.Length+FuturePoints-1; i++) {
						if(i<Trajectory.Points.Length)
						{
							UltiDraw.DrawCircle(Trajectory.Points[i].GetPosition() , 0.02f, UltiDraw.Red); //model-predicted trajetory
							UltiDraw.DrawCircle(trajectory_positions_synthesis[i] , 0.02f, UltiDraw.Green); //synthesis trajetory (user control)	
							UltiDraw.DrawCircle(trajectory_positions_blend[i] , 0.02f, UltiDraw.Black); //blended trajetory (the input of the neural network)									
						}
						else
						{
							UltiDraw.DrawCircle(trajectory_positions_blend[i] , 0.02f, UltiDraw.Blue); // HFTE trajectory
							
						}				
					}	
					if(Show_hand_trajectory)
					{
						for(int i=0; i<Trajectory_right_hand.Points.Length-1; i++) {
							UltiDraw.DrawLine(Trajectory_right_hand.Points[i].GetPosition(), Trajectory_right_hand.Points[i+1].GetPosition(), 0.02f, UltiDraw.Green);
							UltiDraw.DrawLine(Trajectory_left_hand.Points[i].GetPosition(), Trajectory_left_hand.Points[i+1].GetPosition(), 0.02f, UltiDraw.Red);
						}
					}
					UltiDraw.End();
					Trajectory.Draw(5);
				}
			}
		}


		private void OnGUI()
		{
			UltiDraw.Begin();
			if(ShowInformation)
			{
				UltiDraw.OnGUILabel(new Vector2(0.5f, 0.15f), new Vector2(0.1f, 0.05f), 0.0225f, "Style: " + styleList[CurrentStyleIdx], Color.black);
				UltiDraw.OnGUILabel(new Vector2(0.5f, 0.1f), new Vector2(0.1f, 0.05f), 0.0225f, "Apply Frames: " +  applyframes, Color.black);
				UltiDraw.OnGUILabel(new Vector2(0.6f, 0.05f), new Vector2(0.1f, 0.05f), 0.0225f, "Speed: "+string.Format("{0:F2}",speed)+" m/s", Color.green);
				UltiDraw.OnGUILabel(new Vector2(0.4f, 0.05f), new Vector2(0.1f, 0.05f), 0.0225f, "Target: "+string.Format("{0:F2}",Target_speed)+" m/s", Color.red);
				if(Show_inference_time)
				{
					UltiDraw.OnGUILabel(new Vector2(0.9f, 0.05f), new Vector2(0.1f, 0.05f), 0.0225f, "Inf.Time: "+string.Format("{0:F2}",inference_time)+" ms", Color.red);
				}
			}
			UltiDraw.End();
		}
		#if UNITY_EDITOR
		[CustomEditor(typeof(BioAnimation))]
		public class BioAnimation_Editor : Editor {

			public BioAnimation Target;

			void Awake() {
				Target = (BioAnimation)target;
			}

			public override void OnInspectorGUI() {
				Undo.RecordObject(Target, Target.name);

				Inspector();
				Target.Controller.Inspector();

				if(GUI.changed) {
					EditorUtility.SetDirty(Target);
				}
			}

			private void Inspector() {
				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					
					if(Utility.GUIButton("Animation", UltiDraw.DarkGrey, UltiDraw.White)) {
						Target.Inspect = !Target.Inspect;
					}

					if(Target.Inspect) {
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Target.modelPath = (ModelAsset)EditorGUILayout.ObjectField("Model Path", Target.modelPath, typeof(ModelAsset), false);
							Target.modelConfig = (TextAsset)EditorGUILayout.ObjectField("Model Config", Target.modelConfig, typeof(TextAsset), false);
							Target.conditionConfig = (TextAsset)EditorGUILayout.ObjectField("Condition Config", Target.conditionConfig, typeof(TextAsset), false);
							Target.device = (BackendType)EditorGUILayout.EnumPopup("Device Type", Target.device);
							Target.CurrentStyleIdx = EditorGUILayout.IntField("Current Style Index", Target.CurrentStyleIdx);
							Target.ShowTrajectory = EditorGUILayout.Toggle("Show Trajectory", Target.ShowTrajectory);
							Target.Show_hand_trajectory = EditorGUILayout.Toggle("Show_hand_trajectory", Target.Show_hand_trajectory);
							Target.ShowInformation = EditorGUILayout.Toggle("ShowInformation", Target.ShowInformation);
							Target.Show_inference_time = EditorGUILayout.Toggle("Show_inference_time", Target.Show_inference_time);
							Target.CFG_weight = EditorGUILayout.Slider("CFG_weight", Target.CFG_weight, 0f, 2.5f);
							Target.CFGcount_cache =EditorGUILayout.IntField("CFG_count", Target.CFGcount_cache);
							Target.SprintTarget_speed = EditorGUILayout.Slider("Target sprint speed", Target.SprintTarget_speed, 0f, 4f);
							Target.WalkTarget_speed = EditorGUILayout.Slider("Target walk speed", Target.WalkTarget_speed, 0f, 4f);
							Target.TargetGain = EditorGUILayout.Slider("Target Gain", Target.TargetGain, 0f, 1f);
							Target.TargetDecay = EditorGUILayout.Slider("Target Decay", Target.TargetDecay, 0f, 1f);
							Target.Inertialize = EditorGUILayout.Toggle("Inertialize", Target.Inertialize);
							Target.applyframes_cache = EditorGUILayout.IntField("applyframes", Target.applyframes_cache);
							Target.blendTime_rotation = EditorGUILayout.Slider("blendTime_rotation_rotation", Target.blendTime_rotation, 0f, 0.2f);
							Target.blendTime_position = EditorGUILayout.Slider("blendTime_rotation_position", Target.blendTime_position, 0f, 0.2f);
							Target.bias_pos = EditorGUILayout.Slider("bias_pos", Target.bias_pos, 0.1f, 2.0f);
							Target.bias_dir = EditorGUILayout.Slider("bias_dir", Target.bias_dir, 0.1f, 3.0f);
							Target.bias_HFTE = EditorGUILayout.Slider("bias_HFTE", Target.bias_HFTE, 0.1f, 3.0f);
							Target.HFTE = EditorGUILayout.Toggle("HFTE", Target.HFTE);
							Target.Max_angle =EditorGUILayout.Slider("Max_angle", Target.Max_angle, 100f, 179f);
							
						}
					}
				}
			}
		}
		#endif
		
		public void OnDestroy()
		{
			diffusionNetwork.Dispose();
			past_motion.Dispose();
			traj_pose.Dispose();
			traj_trans.Dispose();
			output.Dispose();
			// BVHExporter.Save(Frames, Actor);
		}
		
		void OnDisable()
		{
			// writer.Close();
			// saveWriter.Close();
			// loadReader.Close();
		}
		public void OnExitGame()//退出程序
		{
			#if UNITY_EDITOR
					UnityEditor.EditorApplication.isPlaying = false;//在unity编译器中，退出运行状态
			#else
					Application.Quit();//在打包文件中，则退出整个程序
			#endif
		}

	}
}
