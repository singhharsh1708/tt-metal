// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel.h"
#if !defined(ARCH_QUASAR)
#include "ckernel_structs.h" // semaphore indices; Quasar names its own set in ckernel_trisc_common.h
#endif

// Deliberately outside every build guard, so both builds compile the same barrier.

namespace llk_barrier
{

// counters.h includes this before its own build guard, so it also compiles for BRISC, which has no
// thread identity and never rendezvouses.
#if defined(LLK_TRISC_UNPACK) || defined(LLK_TRISC_MATH) || defined(LLK_TRISC_PACK) || defined(LLK_TRISC_ISOLATE_SFPU)
#define LLK_BARRIER_ON_TRISC 1
#endif

#if defined(ARCH_QUASAR)
constexpr std::uint32_t NUM_THREADS = 4; // unpack, math, pack, sfpu
#else
constexpr std::uint32_t NUM_THREADS = 3; // unpack, math, pack
#endif

#if defined(LLK_BARRIER_ON_TRISC)

// Lives here, not in profiler.h, because the barrier needs it first; profiler.h aliases TRISC_ID onto it.
#if defined(LLK_TRISC_UNPACK)
constexpr std::uint32_t THREAD_ID = 0;
#elif defined(LLK_TRISC_MATH)
constexpr std::uint32_t THREAD_ID = 1;
#elif defined(LLK_TRISC_PACK)
constexpr std::uint32_t THREAD_ID = 2;
#else
constexpr std::uint32_t THREAD_ID = 3;
#endif

// Fixed, not per-run-type: letting it vary is how the two builds ended up releasing from different threads.
constexpr bool is_action_thread()
{
#if defined(LLK_TRISC_PACK)
    return true;
#else
    return false;
#endif
}

#if !defined(ARCH_QUASAR)

// RELEASE has no other user in the LLK. ARRIVE does, in tests that are not instrumented, so the count
// is drained each round rather than assumed to start at zero (see issue #54969).
constexpr std::uint8_t ARRIVE_SEM  = ckernel::semaphore::PACK_DONE;
constexpr std::uint8_t RELEASE_SEM = ckernel::semaphore::UNPACK_OPERAND_SYNC;

// The release is a token each peer consumes, not a level it has to observe, so a peer that samples
// late still finds its token. That is what removes the timing assumption from the old barrier.
template <typename Action>
__attribute__((always_inline)) inline void rendezvous(bool is_action_thread, Action action)
{
    ckernel::fence_compiler();

    if (is_action_thread)
    {
        while (ckernel::semaphore_read(ARRIVE_SEM) < NUM_THREADS - 1)
        {
        }
        while (ckernel::semaphore_read(ARRIVE_SEM) != 0)
        {
            ckernel::semaphore_get(ARRIVE_SEM);
        }

        action();

        for (std::uint32_t i = 0; i < NUM_THREADS - 1; ++i)
        {
            ckernel::semaphore_post(RELEASE_SEM);
        }
    }
    else
    {
        ckernel::semaphore_post(ARRIVE_SEM);
        while (ckernel::semaphore_read(RELEASE_SEM) == 0)
        {
        }
        ckernel::semaphore_get(RELEASE_SEM);
    }

    ckernel::fence_compiler();
}

#else // ARCH_QUASAR

// Quasar names only op-owned semaphores, so it gets an L1 rendezvous instead. NUM_THREADS words, set
// by trisc.cpp because barrier.h cannot include the profiler's L1 map.
extern volatile std::uint32_t* barrier_slots;

// Two generation rounds, so peers cannot leave before action() completes. Generations only increase,
// so a late thread still sees the round it missed; hence the < compare rather than an equality.
template <typename Action>
__attribute__((always_inline)) inline void rendezvous(bool is_action_thread, Action action)
{
    ckernel::fence_compiler();

    volatile std::uint32_t* slots = barrier_slots;

    const std::uint32_t arrive_gen = slots[THREAD_ID] + 1;
    slots[THREAD_ID]               = arrive_gen;
    ckernel::invalidate_data_cache();
    for (std::uint32_t i = 0; i < NUM_THREADS; ++i)
    {
        while (i != THREAD_ID && slots[i] < arrive_gen)
        {
            ckernel::invalidate_data_cache();
        }
    }

    if (is_action_thread)
    {
        action();
    }

    const std::uint32_t release_gen = arrive_gen + 1;
    slots[THREAD_ID]                = release_gen;
    ckernel::invalidate_data_cache();
    for (std::uint32_t i = 0; i < NUM_THREADS; ++i)
    {
        while (i != THREAD_ID && slots[i] < release_gen)
        {
            ckernel::invalidate_data_cache();
        }
    }

    ckernel::fence_compiler();
}

#endif // !ARCH_QUASAR

__attribute__((always_inline)) inline void rendezvous(bool is_action_thread)
{
    rendezvous(is_action_thread, [] {});
}

#endif // LLK_BARRIER_ON_TRISC

} // namespace llk_barrier
