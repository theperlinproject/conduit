[[vk::binding(0, 0)]] cbuffer Uniforms : register(b0, space0) {
    float4 u_tint;
    float4 u_rect;
    float4x4 u_transform;
    float4 u_lights[4];
};

[[vk::binding(0, 1)]] Texture2DArray s_layers : register(t0, space1);
[[vk::binding(1, 1)]] SamplerState s_layers_s : register(s1, space1);
[[vk::binding(2, 1)]] Texture2D s_source : register(t2, space1);
[[vk::binding(3, 1)]] SamplerState s_source_s : register(s3, space1);

struct VsIn {
    [[vk::location(0)]] float4 position : TEXCOORD0;
    [[vk::location(1)]] float4 texcoord : TEXCOORD1;
};

struct VsOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 texcoord : TEXCOORD0;
};

[shader("vertex")]
VsOut vsMain(VsIn input) {
    VsOut output;
    output.position = mul(u_transform, float4(input.position.xy * u_rect.zw + u_rect.xy, 0.0, 1.0));
    output.texcoord = input.texcoord.xy;
    return output;
}

[shader("fragment")]
float4 fsMain(VsOut input) : SV_Target {
    float4 layered = s_layers.Sample(s_layers_s, float3(input.texcoord, 0.0));
    float4 source = s_source.Sample(s_source_s, input.texcoord);
    float3 light = u_lights[0].rgb + u_lights[1].rgb + u_lights[2].rgb + u_lights[3].rgb;
    return layered * source * u_tint * float4(light, 1.0);
}
