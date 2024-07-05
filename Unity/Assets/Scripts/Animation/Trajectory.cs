﻿using UnityEngine;
using System.Collections.Generic;

public class Trajectory {

	public bool Inspect = false;
	public Point[] Points = new Point[0];
	public string[] Styles = new string[0];
	private const int RootPointIndex = 44;
	private static float Width = 0.5f;

	public Trajectory(int size, string[] styles) {
		Inspect = false;
		Points = new Point[size];
		Styles = styles;
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles.Length);
			Points[i].SetTransformation(Matrix4x4.identity);
		}
	}

	public Trajectory(int size, string[] styles, Vector3 seedPosition, Vector3 seedDirection) {
		Inspect = false;
		Points = new Point[size];
		Styles = styles;
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles.Length);
			Points[i].SetTransformation(Matrix4x4.TRS(seedPosition, Quaternion.LookRotation(seedDirection, Vector3.up), Vector3.one));
		}
	}

	public Trajectory(int size, string[] styles, Vector3[] positions, Vector3[] directions) {
		Inspect = false;
		Points = new Point[size];
		Styles = styles;
		for(int i=0; i<Points.Length; i++) {
			Points[i] = new Point(i, styles.Length);
			Points[i].SetTransformation(Matrix4x4.TRS(positions[i], Quaternion.LookRotation(directions[i], Vector3.up), Vector3.one));
		}
	}

	public Point GetFirst() {
		return Points[0];
	}

	public Point GetLast() {
		return Points[Points.Length-1];
	}

	public float GetLength() {
		float length = 0f;
		for(int i=1; i<Points.Length; i++) {
			length += Vector3.Distance(Points[i-1].GetPosition(), Points[i].GetPosition());
		}
		return length;
	}

	public float GetLength(int start, int end, int step) {
		float length = 0f;
		for(int i=0; i<end-step; i+=step) {
			length += Vector3.Distance(Points[i+step].GetPosition(), Points[i].GetPosition());
		}
		return length;
	}
	
	public float GetCurvature(int start, int end, int step) {
		float curvature = 0f;
		for(int i=step; i<end-step; i+=step) {
			curvature += Vector3.SignedAngle(Points[i].GetPosition() - Points[i-step].GetPosition(), Points[i+step].GetPosition() - Points[i].GetPosition(), Vector3.up);
		}
		curvature = Mathf.Abs(curvature);
		curvature = Mathf.Clamp(curvature / 180f, 0f, 1f);
		return curvature;
	}

	public void Postprocess() {
		for(int i=0; i<Points.Length; i++) {
			Points[i].Postprocess();
		}
	}

	public class Point {
		private int Index;
		private Matrix4x4 Transformation;
		private Vector3 Velocity;
		private float Speed;
		private Vector3 LeftSample;
		private Vector3 RightSample;
		private float Slope;
		public float Phase;
		public float[] Signals = new float[0];
		public float[] Styles = new float[0];
		public float[] StyleUpdate = new float[0];

		public Point(int index, int styles) {
			Index = index;
			Transformation = Matrix4x4.identity;
			Velocity = Vector3.zero;
			LeftSample = Vector3.zero;
			RightSample = Vector3.zero;
			Slope = 0f;
			Signals = new float[styles];
			Styles = new float[styles];
			StyleUpdate = new float[styles];
		}

		public void SetIndex(int index) {
			Index = index;
		}

		public int GetIndex() {
			return Index;
		}

		public void SetTransformation(Matrix4x4 matrix) {
			Transformation = matrix;
		}

		public Matrix4x4 GetTransformation() {
			return Transformation;
		}

		public void SetPosition(Vector3 position) {
			Matrix4x4Extensions.SetPosition(ref Transformation, position);
		}

		public Vector3 GetPosition() {
			return Transformation.GetPosition();
		}

		public void SetRotation(Quaternion rotation) {
			Matrix4x4Extensions.SetRotation(ref Transformation, rotation);
		}

		public Quaternion GetRotation() {
			return Transformation.GetRotation();
		}

		public void SetDirection(Vector3 direction) {
			SetRotation(Quaternion.LookRotation(direction == Vector3.zero ? Vector3.forward : direction, Vector3.up));
		}

		public Vector3 GetDirection() {
			return Transformation.GetForward();
		}

		public void SetVelocity(Vector3 velocity) {
			Velocity = velocity;
		}

		public Vector3 GetVelocity() {
			return Velocity;
		}

		public void SetSpeed(float speed) {
			Speed = speed;
		}

		public float GetSpeed() {
			return Speed;
		}

		public void SetPhase(float value) {
			Phase = value;
		}

		public float GetPhase() {
			return Phase;
		}

		public void SetLeftsample(Vector3 position) {
			LeftSample = position;
		}

		public Vector3 GetLeftSample() {
			return LeftSample;
		}

		public void SetRightSample(Vector3 position) {
			RightSample = position;
		}

		public Vector3 GetRightSample() {
			return RightSample;
		}

		public void SetSlope(float slope) {
			Slope = slope;
		}

		public float GetSlope() {
			return Slope;
		}

		public void Postprocess() {
			LayerMask mask = LayerMask.GetMask("Ground");
			Vector3 position = Transformation.GetPosition();
			Vector3 direction = Transformation.GetForward();

			position.y = Utility.GetHeight(Transformation.GetPosition(), mask);
			SetPosition(position);

			Slope = Utility.GetSlope(position, mask);

			Vector3 ortho = Quaternion.Euler(0f, 90f, 0f) * direction;
			RightSample = position + Trajectory.Width * ortho.normalized;
			RightSample.y = Utility.GetHeight(RightSample, mask);
			LeftSample = position - Trajectory.Width * ortho.normalized;
			LeftSample.y = Utility.GetHeight(LeftSample, mask);
		}
	}

	public void Draw(int step=1) {
		UltiDraw.Begin();

		Color[] colors = UltiDraw.GetRainbowColors(Styles.Length);
		
		// orientation
		for(int i=RootPointIndex; i<Points.Length; i+=step) {
			if(i==RootPointIndex)
			{
				UltiDraw.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 0.4f*Points[i].GetDirection(), 0.025f, 0f, UltiDraw.Blue.Transparent(0.75f));
			}
			else
			{
				UltiDraw.DrawLine(Points[i].GetPosition(), Points[i].GetPosition() + 0.4f*Points[i].GetDirection(), 0.025f, 0f, UltiDraw.Orange.Transparent(0.75f));
			}
		}
		UltiDraw.End();
	}

}
