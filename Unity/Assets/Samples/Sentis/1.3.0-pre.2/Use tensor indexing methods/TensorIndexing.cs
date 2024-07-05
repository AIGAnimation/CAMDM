using UnityEngine;
using Unity.Sentis;
using UnityEngine.Assertions;

public class TensorIndexing : MonoBehaviour
{
    [SerializeField]
    Texture2D textureInput;

    void Start()
    {
        // Texture to Tensor and vice-versa.
        using var textureAsATensor = TextureConverter.ToTensor(textureInput);

        // Notice that the tensor axes are [Batch, Channels, Height, Width], with a channel shape of 3 because the texture is RGB.
        Debug.Assert(textureAsATensor.shape == new TensorShape(1, 3, textureInput.height, textureInput.width));

        var renderTexture = TextureConverter.ToTexture(textureAsATensor);

        // Clean up the render texture after use.
        renderTexture.Release();

        // Declares a tensor of rank 3 of shape (1,2,3).
        using var tensorA = new TensorInt(shape: new TensorShape(1, 2, 3), srcData: new int[] { 1, 2, 3, 4, 5, 6 });

        // You can access the tensor shape with the .shape accessor.
        Assert.AreEqual(3, tensorA.shape.rank);
        Assert.AreEqual(1 * 2 * 3, tensorA.shape.length);

        // Shapes can be manipulated like a int[], it supports negative indexing.
        Assert.AreEqual(1, tensorA.shape[0]);
        Assert.AreEqual(2, tensorA.shape[1]);
        Assert.AreEqual(3, tensorA.shape[2]);
        Assert.AreEqual(3, tensorA.shape[-1]);
        Assert.AreEqual(2, tensorA.shape[-2]);
        Assert.AreEqual(1, tensorA.shape[-3]);

        // Shapes can be manipulated in different ways.
        TensorShape shapeB = TensorShape.Ones(rank: 4); // (1,1,1,1)
        shapeB[1] = 2;
        shapeB[2] = 3;
        shapeB[3] = 4;
        Assert.AreEqual(1 * 2 * 3 * 4, shapeB.length);

        // Tensor zero-filled of shape (1,2,3,4).
        var tensorB = TensorFloat.Zeros(shape: shapeB);

        // You can access tensors via their accessors.
        // If your tensor data is on the GPU you need to call MakeReadable before accessing with indexes.
        tensorA.MakeReadable();
        Assert.AreEqual(1, tensorA[0, 0, 0]);
        Assert.AreEqual(2, tensorA[0, 0, 1]);
        Assert.AreEqual(3, tensorA[0, 0, 2]);
        Assert.AreEqual(4, tensorA[0, 1, 0]);
        Assert.AreEqual(5, tensorA[0, 1, 1]);
        Assert.AreEqual(6, tensorA[0, 1, 2]);

        // Each accessors internally flattens the index and uses that to access a flattened representation of the array.
        Assert.AreEqual(1, tensorA[0]); // [0,0,0] = 0*2*3+0*3+0 = 0
        Assert.AreEqual(2, tensorA[1]); // [0,0,1] = 0*2*3+0*3+1 = 1
        Assert.AreEqual(3, tensorA[2]); // [0,0,2] = 0*2*3+0*3+2 = 2
        Assert.AreEqual(4, tensorA[3]); // [0,1,1] = 0*2*3+1*3+0 = 3
        Assert.AreEqual(5, tensorA[4]); // [0,1,2] = 0*2*3+1*3+1 = 4
        Assert.AreEqual(6, tensorA[5]); // [0,1,3] = 0*2*3+1*3+2 = 5

        // You can also compute the 1D index from a multi dim values this way.
        Assert.AreEqual(4, tensorA.shape.RavelIndex(0, 1, 1)); // [0,1,1] = 0*2*3+1*3+1 = 4
        Assert.AreEqual(5, tensorA.shape.RavelIndex(0, 1, 2)); // [0,1,2] = 0*2*3+1*3+2 = 5

        // Accessors can be used to set values in the tensor.
        // If your tensor data is on the GPU you need to call MakeReadable before accessing with indexes.
        tensorB.MakeReadable();
        tensorB[0, 0, 0, 0] = 2.0f;
        tensorB[0, 1, 1, 1] = 3.0f;
        Assert.AreEqual(2.0f, tensorB[0, 0, 0, 0]);
        Assert.AreEqual(3.0f, tensorB[0, 1, 1, 1]);

        // To get the tensor as a flattened array, call ToReadOnlyArray.
        // Tensors can also be created from a slice of a bigger array
        // If your tensor data is on the GPU you need to call MakeReadable before calling ToReadOnlyArray.
        tensorA.MakeReadable();
        var arrayA = tensorA.ToReadOnlyArray();
        var tensorC = new TensorInt(shape: new TensorShape(2), srcData: arrayA, dataStartIndex: 3);
        tensorC.MakeReadable();
        Assert.AreEqual(4, tensorC[0]);
        Assert.AreEqual(5, tensorC[1]);

        // Scalars are declared as follows.
        var tensorD = new TensorFloat(srcData: 5.0f);
        Assert.AreEqual(0, tensorD.shape.rank);
        Assert.AreEqual(new TensorShape(), tensorD.shape);
        tensorD.MakeReadable();
        Assert.AreEqual(5.0f, tensorD[0]);

        // Tensors can also have 0-dim shape, in this case they are empty.
        var tensorE = TensorFloat.Zeros(shape: new TensorShape(0, 4, 5));
        Assert.AreEqual(3, tensorE.shape.rank);
        Assert.AreEqual(0, tensorE.shape.length);
        Assert.IsTrue(tensorE.shape.HasZeroDims());

        // Array.Empty<float>()
        tensorE.MakeReadable();
        var arrayE = tensorE.ToReadOnlyArray();
        Assert.AreEqual(0, arrayE.Length);
    }
}
