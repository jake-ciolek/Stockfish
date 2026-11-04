/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Input features and network structure used in NNUE evaluation function

#ifndef NNUE_ARCHITECTURE_H_INCLUDED
#define NNUE_ARCHITECTURE_H_INCLUDED

#include <cstdint>
#include <iosfwd>

#include "features/k32h3.h"
#include "features/full_threats.h"
#include "features/pp_3wide.h"
#include "layers/affine_transform.h"
#include "layers/affine_transform_sparse_input.h"
#include "layers/clipped_relu.h"
#include "layers/sqr_clipped_relu.h"
#include "nnue_common.h"
#include "nnz_helper.h"

namespace Stockfish::Eval::NNUE {

// Input features used in evaluation function
using ThreatFeatureSet = Features::FullThreats;
using PairFeatureSet   = Features::PP_3Wide;
using PSQFeatureSet    = Features::K32H3;

// Number of input feature dimensions after conversion
constexpr IndexType L1 = 1024;
constexpr int       L2 = 32;
constexpr int       L3 = 16;

constexpr IndexType LayerStacks = 32;

struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       FC_0_OUTPUTS                 = L2;
    static constexpr int       FC_1_OUTPUTS                 = L3;

    Layers::AffineTransformSparseInput<TransformedFeatureDimensions, FC_0_OUTPUTS> fc_0;
    Layers::SqrClippedReLU<FC_0_OUTPUTS, WeightScaleBits + 1>                      ac_sqr_0;
    Layers::ClippedReLU<FC_0_OUTPUTS, WeightScaleBits + 1>                         ac_0;
    Layers::AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS>                        fc_1;
    Layers::SqrClippedReLU<FC_1_OUTPUTS, WeightScaleBits>                          ac_sqr_1;
    Layers::ClippedReLU<FC_1_OUTPUTS, WeightScaleBits>                             ac_1;
    Layers::AffineTransform<FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 1>                fc_2;


#if defined(USE_AVX2)
    alignas(32) i8 fusedOutputWeights[32];

    void prepare_fused_weights() {
        for (int i = 0; i < 32; ++i)
        {
    #if defined(USE_AVX512) && defined(USE_VNNI)
            const int j = (i % 16 / 8) * 16 + i % 8 + (i / 16) * 8;
    #else
            const int j = i % 4 + (i / 4 % 2) * 16 + (i / 8 % 2) * 4 + (i / 16) * 8;
    #endif
            fusedOutputWeights[j] = fc_2.get_weights()[64 + i];
        }
    }

    #if defined(USE_AVXVNNI) && !defined(USE_AVX512)
    i32 propagate_fused(const i32* input) const {
        const auto* p      = reinterpret_cast<const __m256i*>(input);
        const auto  w0     = _mm256_packs_epi32(p[0], p[1]);
        const auto  w1     = _mm256_packs_epi32(p[2], p[3]);
        const auto  square = _mm256_packs_epi16(_mm256_srli_epi16(_mm256_mulhi_epi16(w0, w0), 5),
                                                _mm256_srli_epi16(_mm256_mulhi_epi16(w1, w1), 5));
        const auto  linear =
          _mm256_packs_epi16(_mm256_srli_epi16(_mm256_max_epi16(w0, _mm256_setzero_si256()), 7),
                             _mm256_srli_epi16(_mm256_max_epi16(w1, _mm256_setzero_si256()), 7));
        const auto* output   = reinterpret_cast<const __m256i*>(fc_2.get_weights());
        auto        shortcut = _mm256_setzero_si256();
        SIMD::m256_add_dpbusd_epi32(shortcut, square, output[0]);
        SIMD::m256_add_dpbusd_epi32(shortcut, linear, output[1]);

        __m256i sums[4][2];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 2; ++j)
                sums[i][j] =
                  i == 0
                    ? _mm256_load_si256(reinterpret_cast<const __m256i*>(fc_1.get_biases()) + j)
                    : _mm256_setzero_si256();

        constexpr int order[16] = {5, 14, 0, 11, 2, 10, 3, 9, 15, 6, 1, 4, 13, 7, 12, 8};
        alignas(32) static constexpr int masks[8][8] = {
          {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1, 1}, {2, 2, 2, 2, 2, 2, 2, 2},
          {3, 3, 3, 3, 3, 3, 3, 3}, {4, 4, 4, 4, 4, 4, 4, 4}, {5, 5, 5, 5, 5, 5, 5, 5},
          {6, 6, 6, 6, 6, 6, 6, 6}, {7, 7, 7, 7, 7, 7, 7, 7}};
        #pragma GCC unroll 16
        for (int i = 0; i < 16; ++i)
        {
            const int group = order[i];
            __m256i   mask;
            asm("vmovdqa %1,%0" : "=x"(mask) : "m"(masks[group % 8]));
            const auto  value   = _mm256_permutevar8x32_epi32(group < 8 ? square : linear, mask);
            const auto* weights = reinterpret_cast<const __m256i*>(fc_1.get_weights() + 64 * group);
            SIMD::m256_add_dpbusd_epi32(sums[i % 4][0], value, weights[0]);
            SIMD::m256_add_dpbusd_epi32(sums[i % 4][1], value, weights[1]);
        }
        for (int i = 1; i < 4; ++i)
            for (int j = 0; j < 2; ++j)
                sums[0][j] = _mm256_add_epi32(sums[0][j], sums[i][j]);
        return finish_fused(_mm256_packs_epi32(sums[0][0], sums[0][1]), shortcut);
    }
    #else
    i32 propagate_fused(const u8* input) const {
        #if defined(USE_AVX512) && defined(USE_VNNI)
        auto sum0     = _mm512_load_si512(fc_1.get_biases());
        auto sum1     = _mm512_setzero_si512();
        auto shortcut = _mm512_setzero_si512();
        SIMD::m512_add_dpbusd_epi32(shortcut, _mm512_load_si512(input),
                                    _mm512_load_si512(fc_2.get_weights()));
        for (int i = 0; i < 16; i += 2)
        {
            SIMD::m512_add_dpbusd_epi32(sum0, _mm512_set1_epi32(load_as<i32>(input + 4 * i)),
                                        _mm512_load_si512(fc_1.get_weights() + 64 * i));
            SIMD::m512_add_dpbusd_epi32(sum1, _mm512_set1_epi32(load_as<i32>(input + 4 * (i + 1))),
                                        _mm512_load_si512(fc_1.get_weights() + 64 * (i + 1)));
        }
        const auto words = _mm512_cvtsepi32_epi16(_mm512_add_epi32(sum0, sum1));
        return finish_fused(words, _mm256_setzero_si256()) + SIMD::m512_hadd(shortcut, 0);
        #else
        auto sum0     = _mm256_load_si256(reinterpret_cast<const __m256i*>(fc_1.get_biases()));
        auto sum1     = _mm256_load_si256(reinterpret_cast<const __m256i*>(fc_1.get_biases()) + 1);
        auto shortcut = _mm256_setzero_si256();
        const auto* output = reinterpret_cast<const __m256i*>(fc_2.get_weights());
        SIMD::m256_add_dpbusd_epi32(
          shortcut, _mm256_load_si256(reinterpret_cast<const __m256i*>(input)), output[0]);
        SIMD::m256_add_dpbusd_epi32(
          shortcut, _mm256_load_si256(reinterpret_cast<const __m256i*>(input) + 1), output[1]);
        for (int i = 0; i < 16; ++i)
        {
            const auto  value   = _mm256_set1_epi32(load_as<i32>(input + 4 * i));
            const auto* weights = reinterpret_cast<const __m256i*>(fc_1.get_weights() + 64 * i);
            SIMD::m256_add_dpbusd_epi32(sum0, value, weights[0]);
            SIMD::m256_add_dpbusd_epi32(sum1, value, weights[1]);
        }
        return finish_fused(_mm256_packs_epi32(sum0, sum1), shortcut);
        #endif
    }
    #endif

    i32 finish_fused(__m256i words, __m256i sum) const {
        const auto square = _mm256_srli_epi16(_mm256_mulhi_epi16(words, words), 3);
        const auto linear = _mm256_srli_epi16(_mm256_max_epi16(words, _mm256_setzero_si256()), 6);
        const auto values = _mm256_packs_epi16(square, linear);
        SIMD::m256_add_dpbusd_epi32(
          sum, values, _mm256_load_si256(reinterpret_cast<const __m256i*>(fusedOutputWeights)));
        return SIMD::m256_hadd(sum, fc_2.get_biases()[0]);
    }
#endif

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value() {
        // input slice hash
        u32 hashValue = 0xEC42E90Du;
        hashValue ^= TransformedFeatureDimensions * 2;

        hashValue = decltype(fc_0)::get_hash_value(hashValue);
        // TODO: consider including hash value of ac_sqr_0 in the overall hash value.
        // For now omitted on purpose because hash value is not written by trainer (yet)
        hashValue = decltype(ac_0)::get_hash_value(hashValue);
        hashValue = decltype(fc_1)::get_hash_value(hashValue);
        hashValue = decltype(ac_1)::get_hash_value(hashValue);
        hashValue = decltype(fc_2)::get_hash_value(hashValue);

        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        const bool ok = fc_0.read_parameters(stream) && ac_0.read_parameters(stream)
                     && fc_1.read_parameters(stream) && ac_1.read_parameters(stream)
                     && fc_2.read_parameters(stream);
#if defined(USE_AVX2)
        if (ok)
            prepare_fused_weights();
#endif
        return ok;
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        return fc_0.write_parameters(stream) && ac_0.write_parameters(stream)
            && fc_1.write_parameters(stream) && ac_1.write_parameters(stream)
            && fc_2.write_parameters(stream);
    }

    i32 propagate(const TransformedFeatureType* transformedFeatures,
                  const NNZInfo<L1>&            nnzInfo) const {
        struct alignas(CacheLineSize) Buffer {
            alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
            alignas(CacheLineSize) typename decltype(ac_sqr_0)::OutputType
              concat_buffer[ceil_to_multiple<IndexType>(FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 32)];
            alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
            alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;
        };

        Buffer buffer;

        fc_0.propagate(transformedFeatures, buffer.fc_0_out, nnzInfo);
#if defined(USE_AVXVNNI) && !defined(USE_AVX512)
        i32 fwdOut = propagate_fused(buffer.fc_0_out);
#else
    #if defined(USE_PAIR_ACTIVATIONS)
        ac_sqr_0.propagate_pair(buffer.fc_0_out, buffer.concat_buffer,
                                buffer.concat_buffer + FC_0_OUTPUTS);
    #else
        ac_sqr_0.propagate(buffer.fc_0_out, buffer.concat_buffer);
        ac_0.propagate(buffer.fc_0_out, buffer.concat_buffer + FC_0_OUTPUTS);
    #endif

    #if defined(USE_AVX2)
        i32 fwdOut = propagate_fused(buffer.concat_buffer);
    #else
        fc_1.propagate(buffer.concat_buffer, buffer.fc_1_out);
        #if defined(USE_PAIR_ACTIVATIONS)
        ac_sqr_1.propagate_pair(buffer.fc_1_out, buffer.concat_buffer + FC_0_OUTPUTS * 2,
                                buffer.concat_buffer + FC_0_OUTPUTS * 2 + FC_1_OUTPUTS);
        #else
        ac_sqr_1.propagate(buffer.fc_1_out, buffer.concat_buffer + FC_0_OUTPUTS * 2);
        ac_1.propagate(buffer.fc_1_out, buffer.concat_buffer + FC_0_OUTPUTS * 2 + FC_1_OUTPUTS);
        #endif

        fc_2.propagate(buffer.concat_buffer, buffer.fc_2_out);

        static_assert(FC_0_OUTPUTS >= 2);
        i32 fwdOut = buffer.fc_2_out[0];
    #endif
#endif
        i32 skip_0 = buffer.fc_0_out[FC_0_OUTPUTS - 2] - buffer.fc_0_out[FC_0_OUTPUTS - 1];
        fwdOut += skip_0;

        // fwdOut is such that 1.0 is equal to HiddenOneVal*(1<<WeightScaleBits)*2 in
        // quantized form, but we want 1.0 to be equal to 600*OutputScale
        // to make overflow impossible we cast to int64_t
        constexpr i64 multiplier = 600 * OutputScale;
        constexpr i64 denominator =
          static_cast<i64>(HiddenOneVal) * static_cast<i64>(1U << WeightScaleBits) * 2;

        i32 outputValue = static_cast<i32>((static_cast<i64>(fwdOut) * multiplier) / denominator);
        return outputValue;
    }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, fc_0.get_content_hash());
        hash_combine(h, ac_sqr_0.get_content_hash());
        hash_combine(h, ac_0.get_content_hash());
        hash_combine(h, fc_1.get_content_hash());
        // hash_combine(h, ac_sqr_1.get_content_hash()); TODO
        hash_combine(h, ac_1.get_content_hash());
        hash_combine(h, fc_2.get_content_hash());
        hash_combine(h, get_hash_value());
        return h;
    }
};

}  // namespace Stockfish::Eval::NNUE

template<>
struct std::hash<Stockfish::Eval::NNUE::NetworkArchitecture> {
    Stockfish::usize
    operator()(const Stockfish::Eval::NNUE::NetworkArchitecture& arch) const noexcept {
        return arch.get_content_hash();
    }
};

#endif  // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
