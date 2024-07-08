using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class BVHExporter : MonoBehaviour
{
    public string HIERARCHY_filepath = "Assets/Resources/rest.bvh";
    public string output_filepath = "Assets/Resources/example.bvh";
    public string MOTION_filepath = "Assets/Resources/motion_temp.bvh";
    public string condition_filepath = "Assets/Resources/condition.txt";
    public string speed_filepath = "Assets/Resources/speed.txt";
    public void Save(int Frames,Actor Actor, List<string> names = null, double frametime = 1.0 / 30.0f, string order = "zyx", bool positions = false, bool orients = true)
    {
        if (names == null)
        {
            names = new List<string>();
            for (int i = 0; i < Actor.Bones.Length; i++)
                names.Add(Actor.Bones[i].Transform.name);
        }
        StreamReader reader = new StreamReader(HIERARCHY_filepath);
        string content_HIERARCHY = reader.ReadToEnd();
        reader.Close();

        StreamReader reader1 = new StreamReader(MOTION_filepath);
        string content_MOTION = reader1.ReadToEnd();
        reader1.Close();
        using (StreamWriter f = new StreamWriter(output_filepath))
        {
            f.Write(content_HIERARCHY);  
            f.WriteLine();
            f.WriteLine("MOTION");
            f.WriteLine("Frames: {0}", Frames);
            f.WriteLine("Frame Time: {0}", frametime);
            f.Write(content_MOTION);
        }
    }
}
