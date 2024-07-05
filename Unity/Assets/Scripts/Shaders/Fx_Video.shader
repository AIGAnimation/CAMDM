Shader "测试/Fx_Video" {
	Properties{
	
		//[Header(TexBase)]
		
		_BaseTex("BaseTex", 2D) = "white" {}
		

			//[Header(TexSec)]
		//[Toggle]_SecondTexture("启用TexSec", float) = 0

		_SecTex("SecTex", 2D) = "white" {}
		

			//[Header(MaskTexture)]
		//[Toggle] _MaskTexture("启用Mask", float) = 0
		_MaskTex("MaskTex", 2D) = "white" {}
		

			//[Header(DissTex)]
		//[Toggle] _DissTexture("启用Dissolve", float) = 0
		
		_DissTex("DissTex", 2D) = "white" {}
		

			//[Header(TwistTex)]
		//[Toggle] _TwistTexture("启用Twist", float) = 0
		_TwistTex("TwistTex", 2D) = "white" {}
		

			//[Header(VertexAnimationTex)]
		//[Toggle] _VertexAnimation("启用VertexAnimation", float) = 0
		_VerTex("VertexAnimationTexture", 2D) = "white" {}
		

			//[Header(FresnelAlpha Properties)]
		//[Toggle] _FresnelAlpha("启用FresnelAlpha", float) = 0
		


		_BaseColor("Base颜色", Color) = (1,1,1,1)
		_BaseParameter("Base参数", Vector) = (0,0,0,1)
		_SecColor("Sec颜色", Color) = (1,1,1,1)
		_SecParameter("Sec参数", Vector) = (0,0,0,1)
		_MaskParameter("Mask参数", Vector) = (0,0,0,0)
		_DissColor("Diss颜色", Color) = (1,0,0,1)
		_DissParameter("Diss参数", Vector) = (0,0,0,0)
		_DissControl("Diss溶解", Range(0, 1)) = 0.75
		_DissWidth("Diss硬边", Range(1, 50)) = 5
		_DissIntensity("Diss边亮度", Range(0, 10)) = 5
		_TwistParameter("Twist参数", Vector) = (0,0,0,0.02)
		_VerTexParameter("VeAnim参数", Vector) = (0,0,0,0.5)
		_VertAnimMode("VertAnim模式", Range(0, 1)) = 0
		_FreParameter("Fresnel参数", Vector) = (1,3,0,0)


			//[Toggle]_ZWrite("ZWrite On-----------------------------------------------------------------------", float) = 0.0
	[Toggle]_ParticleAndSlider("溶解", float) = 0
	[Toggle]_ParticleCustomData("自定义数据", float) = 0.0//DissTex没有CustomDataUV控制
	_BlendMode("叠加方式", Range(0, 1)) = 0

			//[Enum(UnityEngine.Rendering.CullMode)] _Cull("Culling", float) = 0

	}

	SubShader{
		Tags { "Queue" = "Transparent+100" "IgnoreProjector" = "True" "RenderType" = "Transparent" }
		Fog { Mode Off }
		LOD 100

		Pass {
			Cull[_Cull]
			ZWrite[_ZWrite]
			Lighting Off
			Blend One OneMinusSrcAlpha

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#pragma shader_feature __ _PARTICLECUSTOMDATA_ON
			#pragma shader_feature __ _SECONDTEXTURE_ON
			#pragma shader_feature __ _MASKTEXTURE_ON
			#pragma shader_feature __ _DISSTEXTURE_ON
			#pragma shader_feature __ _TWISTTEXTURE_ON
			#pragma shader_feature __ _VERTEXANIMATION_ON
			#pragma shader_feature __ _PARTICLEANDSLIDER_ON
			#pragma shader_feature __ _FRESNELALPHA_ON
			#pragma target 3.0
			#include "UnityCG.cginc"


			float _BlendMode;
			sampler2D _BaseTex;		float4 _BaseTex_ST;		float4 _BaseColor;		float4 _BaseParameter;		float _DissControl;
							
		#ifdef 	_SECONDTEXTURE_ON
			sampler2D _SecTex;		float4 _SecColor;		float4 _SecTex_ST;		float4 _SecParameter;
		#endif
			
		#ifdef _MASKTEXTURE_ON
			sampler2D _MaskTex;		float4 _MaskTex_ST;		float4 _MaskParameter;
		#endif
			
		#ifdef 	_DISSTEXTURE_ON
			sampler2D _DissTex;		float4 _DissColor;		float _DissWidth;		float _DissIntensity;		float4 _DissTex_ST;		float4 _DissParameter;
		#endif
			
		#ifdef 	_TWISTTEXTURE_ON
			sampler2D _TwistTex;	float4 _TwistTex_ST;	float4 _TwistParameter;
		#endif

		#ifdef 	_VERTEXANIMATION_ON
			sampler2D _VerTex;	float4 _VerTex_ST;	float4 _VerTexParameter;	float _VertAnimMode;
		#endif

		#ifdef _FRESNELALPHA_ON
			float4 _FreParameter;
		#endif

			float2 RotateUV(float2 texcoord, float theta)
			{
				float2 sc;
				sincos((theta * 3.141592653) / 180.0, sc.x, sc.y);//除以了180,那么属性面板的180就代表180度。没有除以180,属性面板的1代表180度。
				float2 uv = texcoord - 0.5;
				float2 rotateduv;
				rotateduv.x = dot(uv, float2(sc.y, -sc.x));
				rotateduv.y = dot(uv, sc.xy);
				rotateduv += 0.5;
				return rotateduv;
			}
			//平移贴图<函数>
			float2 OffsetUV(float speedX, float speedY)
			{
				return frac(float2(speedX, speedY) * _Time.y);
			}


			struct a2v {
				float4 vertex : POSITION;
				float4 vertexColor : COLOR;
				float2 texcoord0 : TEXCOORD0;
				float4 texcoord1 : TEXCOORD1;
				float4 texcoord2 : TEXCOORD2;
				float3 normal : NORMAL;
			};
			struct v2f {
				float4 pos : SV_POSITION;
				float4 vertexColor : COLOR;
				float2 uv0 : TEXCOORD0;
			#ifdef _PARTICLECUSTOMDATA_ON
				float4 uv1 : TEXCOORD1;
				float4 uv2 : TEXCOORD2;
			#endif

			#ifdef _FRESNELALPHA_ON
				float4 posWorld : TEXCOORD3;
				float3 normalDir : TEXCOORD4;
			#endif
			};
//------------------------------------------------------------------------------------------------------------------------------
			v2f vert(a2v v) {
				v2f o;
				o.vertexColor = v.vertexColor;
				
				o.uv0 = v.texcoord0;
			#ifdef _PARTICLECUSTOMDATA_ON
				o.uv1 = v.texcoord1;
				o.uv2 = v.texcoord2;
			#endif

			#ifdef _PARTICLECUSTOMDATA_ON	
				float CustomData_VerUV = o.uv2.b;	//uv2的z通道控制顶点动画的UV。
				float CustomData_VerStrength = o.uv2.a; 	//uv2的w通道控制顶点动画的强度。
			#else
				float CustomData_VerUV = 0;
				float CustomData_VerStrength = 1;
			#endif

			#ifdef 	_VERTEXANIMATION_ON
				float2 verTex_Rotate = RotateUV(o.uv0, _VerTexParameter.z);
				float2 verTex_Offset = OffsetUV(_VerTexParameter.x, _VerTexParameter.y);
				float2 verTexUV = TRANSFORM_TEX(verTex_Rotate, _VerTex) + verTex_Offset - (0.5f * _VerTex_ST.xy - 0.5f) + float2(CustomData_VerUV, 0);

				float4 _vertexAni = tex2Dlod(_VerTex, float4(verTexUV, 0, 0));
				v.vertex.xyz += (lerp(_vertexAni.r, (_vertexAni.r*v.normal), _VertAnimMode) * _VerTexParameter.w * CustomData_VerStrength);
			#endif

			#ifdef _FRESNELALPHA_ON
				o.normalDir = UnityObjectToWorldNormal(v.normal);
				o.posWorld = mul(unity_ObjectToWorld, v.vertex);
			#endif
				

				o.pos = UnityObjectToClipPos(v.vertex);
				return o;
			}
//------------------------------------------------------------------------------------------------------------------------------
			float4 frag(v2f i, float facing : VFACE) : SV_Target{

			#ifdef _PARTICLECUSTOMDATA_ON	//Diss贴图没有用到CustomDataUV控制
				float2 CustomDataUV_Base = float2(i.uv1.r, i.uv1.g);
				float2 CustomDataUV_Sec = float2(i.uv1.b, i.uv1.a);
				float2 CustomDataUV_Mask = float2(i.uv2.r, i.uv2.g);
			#else
				float2 CustomDataUV_Base = float2(0, 0);
				float2 CustomDataUV_Sec = float2(0, 0);
				float2 CustomDataUV_Mask = float2(0, 0);
			#endif
				float2 base_Rotate = RotateUV(i.uv0, _BaseParameter.z);
				float2 base_Offset = OffsetUV(_BaseParameter.x, _BaseParameter.y);
				float2 baseUV = TRANSFORM_TEX(base_Rotate, _BaseTex) + base_Offset - (0.5f * _BaseTex_ST.xy - 0.5f) + CustomDataUV_Base;
				
			#ifdef _TWISTTEXTURE_ON//如果激活了Twist扭曲贴图,就怎么样
				float2 twist_Rotate = RotateUV(i.uv0, _TwistParameter.z);
				float2 twist_Offset = OffsetUV(_TwistParameter.x, _TwistParameter.y);
				float2 twistUV = TRANSFORM_TEX(twist_Rotate, _TwistTex) + twist_Offset - (0.5f * _TwistTex_ST.xy - 0.5f);
				float4 _TwistTex_var = tex2D(_TwistTex, twistUV);
				float TwistEffect = _TwistTex_var.r*_TwistParameter.a;

				float2 baseTwist = baseUV + TwistEffect;//激活Base扭曲后的BaseUV
			#else
				float2 baseTwist = baseUV;//没激活Twist扭曲的TwistUV

				float TwistEffect = float2(0, 0);
			#endif
				
				float4 _BaseTex_var = tex2D(_BaseTex, baseTwist);
				float3 baseRGBEdit = (_BaseColor.rgb*_BaseTex_var.rgb*_BaseParameter.a);
				float baseAlphaEdit = saturate((_BaseColor.a*_BaseTex_var.a*_BaseParameter.a));

			#ifdef _SECONDTEXTURE_ON//如果激活了Second二号贴图,就怎么样
				float2 sec_Rotate = RotateUV(i.uv0, _SecParameter.z);
				float2 sec_Offset = OffsetUV(_SecParameter.x, _SecParameter.y);
				float2 secUV = TRANSFORM_TEX(sec_Rotate, _SecTex) + sec_Offset - (0.5f * _SecTex_ST.xy - 0.5f) + CustomDataUV_Sec;
				//上行代码里的sec_Offset的作用是控制UV流动。 -(0.5f * _BaseTex_ST.xy - 0.5f)的作用是在用Tiling缩放时以中心为锚点。CustomDataUV_Sec的作用是自定义UV。其它贴图同理。

				float4 _SecTex_var = tex2D(_SecTex, secUV + TwistEffect);
				float3 SecRGB =  baseRGBEdit + (_SecColor.rgb*_SecColor.a*_SecTex_var.rgb*_SecParameter.w);
				float SecAlpha =  baseAlphaEdit*_SecTex_var.a;
			#else
				float3 SecRGB = baseRGBEdit;
				float SecAlpha = baseAlphaEdit;
			#endif
				
			#ifdef _PARTICLEANDSLIDER_ON//Particle的Alpha和Slider控制溶解的切换。
				float particleAlpha_SlinderAA = 1 - i.vertexColor.a;//用1减是为了让Alpha的<255-0>是<先有后无>的效果,否则就是<先无后有>。
			#else
				float particleAlpha_SlinderAA = _DissControl;
			#endif
				

			#ifdef _DISSTEXTURE_ON//如果激活了Diss溶解贴图,就怎么样
				float2 diss_Rotate = RotateUV(i.uv0, _DissParameter.z);
				float2 diss_Offset = OffsetUV(_DissParameter.x, _DissParameter.y);
				float2 dissUV = TRANSFORM_TEX(diss_Rotate, _DissTex) + diss_Offset - (0.5f * _DissTex_ST.xy - 0.5f);

				float4 _DissTex_var = tex2D(_DissTex, dissUV + TwistEffect);
				float dissBlack = saturate(((_DissTex_var.r - (particleAlpha_SlinderAA*2.0 + -1.0))*_DissWidth));
				float3 DissRGB = (_DissIntensity*(1.0 - dissBlack)*_DissColor.rgb) + SecRGB;
				float DisAlpha =  dissBlack*SecAlpha;
			#else
				float3 DissRGB = SecRGB;
				float DisAlpha = SecAlpha;
			#endif
				

			#ifdef _MASKTEXTURE_ON//如果激活了Mask遮罩贴图,就怎么样
				float2 mask_Rotate = RotateUV(i.uv0, _MaskParameter.z);
				float2 mask_Offset = OffsetUV(_MaskParameter.x, _MaskParameter.y);
				float2 maskUV = TRANSFORM_TEX(mask_Rotate, _MaskTex) + mask_Offset - (0.5f * _MaskTex_ST.xy - 0.5f) + CustomDataUV_Mask;

				float4 _MaskTex_var = tex2D(_MaskTex, maskUV + TwistEffect);
				float MaskRGB = DisAlpha*_MaskTex_var.r;
			#else
				float MaskRGB = DisAlpha;
			#endif

			#ifdef _PARTICLEANDSLIDER_ON//激活Particle的Alpha和控制溶解,Particle的Alpha就没有透明效果了。
				float particleAlpha_SlinderBB = MaskRGB;
			#else
				float particleAlpha_SlinderBB = i.vertexColor.a*MaskRGB;
			#endif


			#ifdef _FRESNELALPHA_ON//Alpha通道加一层Fresnel。
				float faceSign = (facing >= 0 ? 1 : -1);
				i.normalDir = normalize(i.normalDir);
				i.normalDir *= faceSign;
				float3 normalDirection = i.normalDir;
				float3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - i.posWorld.xyz);
				float freValue = pow((1.0 - pow(1.0 - max(0, dot(normalDirection, viewDirection)), _FreParameter.x)), _FreParameter.y);
				float3 finalColor = ((DissRGB*i.vertexColor.rgb)*particleAlpha_SlinderBB*freValue);
				return fixed4(finalColor, (particleAlpha_SlinderBB*_BlendMode*freValue));
			#else
				float3 finalColor = ((DissRGB*i.vertexColor.rgb)*particleAlpha_SlinderBB);
				return fixed4(finalColor, (particleAlpha_SlinderBB*_BlendMode));
			#endif

				
			}
			ENDCG
		}
	}
}
