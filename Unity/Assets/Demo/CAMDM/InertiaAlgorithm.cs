using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Sentis;

using Unity.Mathematics;
using static Unity.Mathematics.math;

// modify from https://github.com/portalmk2/InertializationForUnity
public class InertiaAlgorithm : MonoBehaviour
{
    public float4 toAxisAngle(quaternion quat)
    {
        float4 q1 = quat.value;

        if (q1.w > 1)
            normalize(q1);
        float angle = 2 * acos(q1.w);
        float s = sqrt(1 - q1.w * q1.w);
        float3 axis;
        if (s < 0.001)
        {
            axis.x = q1.x;
            axis.y = q1.y;
            axis.z = q1.z;
        }
        else
        {
            axis.x = q1.x / s; // normalise axis
            axis.y = q1.y / s;
            axis.z = q1.z / s;
        }
        return float4(axis, angle);
    }

    public float Inertialize(float x0, float v0, float dt, float tf, float t)
    {
        float tf1 = -5 * x0 / v0;
        if (tf1 > 0)
            tf = min(tf, tf1);

        t = min(t, tf);

        float tf2 = tf * tf;
        float tf3 = tf2 * tf;
        float tf4 = tf3 * tf;
        float tf5 = tf4 * tf;

        float a0 = (-8 * v0 * tf - 20 * x0) / (tf * tf);

        float A = -(a0 * tf2 + 6 * v0 * tf + 12 * x0) / (2 * tf5);
        float B = (3 * a0 * tf2 + 16 * v0 * tf + 30 * x0) / (2 * tf4);
        float C = -(3 * a0 * tf2 + 12 * v0 * tf + 20 * x0) / (2 * tf3);

        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t3 * t;
        float t5 = t4 * t;

        float xt = A * t5 + B * t4 + C * t3 + (a0 / 2) * t2 + v0 * t + x0;


        if (tf < 0.00001f)
            xt = 0;

        return xt;

    }

    public float Inertialize(float prev, float curr, float target, float dt, float tf, float t)
    {
        float x0 = curr - target;
        float v0 = (curr - prev) / dt;

        return Inertialize(x0, v0, dt, tf, t);
    }

    public float3 Inertialize_Position(float3 prev, float3 curr, float3 target, float dt, float tf, float t)
    {
        float3 vx0 = curr - target;
        float3 vxn1 = prev - target;
// 
        float x0 = length(vx0); //幅值？？？？

        float3 vx0_dir = x0 > 0.00001f ? (vx0 / x0) : length(vxn1) > 0.00001f ? normalize(vxn1) : float3(1, 0, 0);

        float xn1 = dot(vxn1, vx0_dir);
        float v0 = (x0 - xn1) / dt;

        float xt = Inertialize(x0, v0, dt, tf, t);
        // print(xt);

        float3 vxt = xt * vx0_dir + target;

        return vxt;
    }


    public quaternion Inertialize_Rotation(Quaternion prev, Quaternion curr, Quaternion target, float dt, float tf, float t)
    {
        // curr 是当前帧的四元数（blend以后的)
        // For quaternions, we usually want them to be continuous when performing animation interpolation. 
        // However, quaternions have a property that they can represent two versions of the same rotation, i.e. a positive version and a negative version. 
        // These two versions are mathematically equivalent, but may cause discontinuity issues during interpolation.
        // To ensure the continuity of quaternions, we usually choose the version closest to the previous quaternion. 
        // This can be achieved by checking the dot product of the new quaternion with the previous one. If the dot product is negative, we take the negative version of the new quaternion.

        // The dot product of quaternions (also known as scalar product or inner product) can be used to measure the angle between two quaternions. 
        // If the dot product is negative, it means that the angle between the two quaternions is greater than 90 degrees, which is to say, they are in the "opposite" direction in quaternion space.
        if(Quaternion.Dot(curr,target)<0)
        {
            target=new Quaternion(-target.x,-target.y,-target.z,-target.w);
        }

        quaternion q0 = normalize(mul(curr, inverse(target)));  //offet rotation 计算q0和qn1，它们表示当前帧和前一帧的四元数相对于目标四元数的偏移旋转。这是通过将当前帧和前一帧的四元数与目标四元数的逆相乘并归一化得到的。
        quaternion qn1 = normalize(mul(prev, inverse(target)));


        float4 q0_aa = toAxisAngle(q0); //将q0四元数转换为轴角表示。将轴存储在vx0中，将角度存储在x0中。
        float3 vx0 = q0_aa.xyz;  //轴
        float x0 = q0_aa.w;    // 角度


        float xn1 = 2 * atan(dot(qn1.value.xyz, vx0) / qn1.value.w);

        float v0 = (x0 - xn1) / dt;  //角速度, 标量值   x0 和 xn1 都是以弧度表示的角度。x0 是当前帧相对于目标四元数的旋转角度，xn1 是前一帧相对于目标四元数的旋转角度。dt 是时间步长，即两帧之间的时间差。

        float xt = Inertialize(x0, v0, dt, tf, t);
        quaternion qt = mul(Unity.Mathematics.quaternion.AxisAngle(vx0, xt), target);

        return normalize(qt);
    }

}
