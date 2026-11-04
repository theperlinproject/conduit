[[vk::binding(0, 0)]] RWStructuredBuffer<uint> args : register(u0, space0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> counter : register(u1, space0);

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= 8u || ((0x25u >> id.x) & 1u) == 0u)
        return;
    uint slot;
    InterlockedAdd(counter[0], 1u, slot);
    args[slot * 5u + 0u] = 3u;
    args[slot * 5u + 1u] = 1u;
    args[slot * 5u + 2u] = 0u;
    args[slot * 5u + 3u] = 0u;
    args[slot * 5u + 4u] = id.x;
}
