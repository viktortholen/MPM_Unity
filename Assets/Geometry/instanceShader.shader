Shader "Unlit/instanceShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
        {
        Tags { "RenderType"="Opaque" }

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata_t
            {
                float4 vertex : POSITION;
                float4 color : COLOR;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float4 color : COLOR;
            };

            struct MeshProperties {
                float4x4 mat;
                float4 color;
            };

            StructuredBuffer<MeshProperties> _Properties;
            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata_t i, uint instanceID: SV_InstanceID)
            {
                v2f o;
                float4 pos = mul(_Properties[instanceID].mat, i.vertex);
                o.vertex = UnityObjectToClipPos(pos);
                o.color = _Properties[instanceID].color;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                fixed4 col = i.color;

                return col;
            }
            ENDCG
        }
        }
}
