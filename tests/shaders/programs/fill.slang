[[vk::binding(0, 0)]] cbuffer Uniforms : register(b0, space0) {
    uint4 u_fill;
};

[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output : register(u1, space0);

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < u_fill.y)
        output[id.x] = u_fill.x + id.x;
}
