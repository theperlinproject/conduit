struct Push {
    float4 tint;
};
struct Frame {
    float2 scale;
    float2 pad;
};

[[vk::push_constant]] ConstantBuffer<Push> push : register(b0, space15);
[[vk::binding(0, 0)]] ConstantBuffer<Frame> frame : register(b0, space0);
[[vk::binding(1, 0)]] Texture2D<float4> tex : register(t1, space0);
[[vk::binding(2, 0)]] SamplerState samp : register(s2, space0);

struct VsIn {
    [[vk::location(0)]] float2 pos : TEXCOORD0;
    [[vk::location(1)]] float2 uv : TEXCOORD1;
    [[vk::location(2)]] float2 offset : TEXCOORD2;
};

struct VsOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOut vsMain(VsIn i) {
    VsOut o;
    o.pos = float4(i.pos * frame.scale + i.offset, 0.5, 1.0);
    o.uv = i.uv;
    return o;
}

[shader("fragment")]
float4 fsMain(VsOut i) : SV_Target {
    return tex.Sample(samp, i.uv) * push.tint;
}
