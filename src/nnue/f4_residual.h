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

// Four-neuron quantized residual, serialized as part of each network head.
#ifndef NNUE_F4_RESIDUAL_H
#define NNUE_F4_RESIDUAL_H
#include <algorithm>
#include <array>
#include <cstring>
#include "nnue_common.h"
namespace Stockfish::Eval::NNUE {
struct F4Residual {
    alignas(64) std::array<i8, 4096> dense{};
#if !defined(USE_AVX2) && defined(USE_SSSE3)
    alignas(64) std::array<i8, 4096> blocks{};
#endif
    alignas(16) std::array<i32, 4> bias{}, readout{};
    bool read_parameters(std::istream& stream) {
        read_little_endian(stream, dense.data(), dense.size());
        read_little_endian(stream, bias.data(), bias.size());
        read_little_endian(stream, readout.data(), readout.size());
#if !defined(USE_AVX2) && defined(USE_SSSE3)
        for (unsigned i = 0; i < 1024; ++i)
            for (unsigned k = 0; k < 4; ++k)
                blocks[(i / 4) * 16 + k * 4 + i % 4] = dense[k * 1024 + i];
#endif
        return bool(stream);
    }
    bool write_parameters(std::ostream& stream) const {
        write_little_endian(stream, dense.data(), dense.size());
        write_little_endian(stream, bias.data(), bias.size());
        write_little_endian(stream, readout.data(), readout.size());
        return bool(stream);
    }
    static i32 finish(const i32* values, const i32* weights) {
        i64 sum = 0;
        for (unsigned k = 0; k < 4; ++k)
            sum += i64(std::clamp(values[k] >> 6, 0, 127)) * weights[k];
        // Scale and truncate the residual separately before adding it to the main score.
        return i32(sum * 9600 / 524288);
    }
#if defined(USE_AVX2)
    static __m256i dot256(__m256i a, __m256i x, __m256i w) {
    #if defined(USE_AVXVNNI)
        return _mm256_dpbusd_avx_epi32(a, x, w);
    #elif defined(USE_VNNI)
        return _mm256_dpbusd_epi32(a, x, w);
    #else
        // x <= 127, |w| <= 127: the pairwise i16 sums cannot saturate.
        return _mm256_add_epi32(
          a, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), _mm256_set1_epi16(1)));
    #endif
    }
    static i32 sum8(__m256i a) {
        __m128i v = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        v         = _mm_hadd_epi32(v, v);
        v         = _mm_hadd_epi32(v, v);
        return _mm_cvtsi128_si32(v);
    }
#endif
#if defined(USE_SSSE3)
    static __m128i dot128(__m128i a, __m128i x, __m128i w) {
    #if defined(USE_AVXVNNI)
        return _mm_dpbusd_avx_epi32(a, x, w);
    #elif defined(USE_VNNI)
        return _mm_dpbusd_epi32(a, x, w);
    #else
        return _mm_add_epi32(a, _mm_madd_epi16(_mm_maddubs_epi16(x, w), _mm_set1_epi16(1)));
    #endif
    }
#endif
    i32 propagate(const u8* x) const {
        alignas(32) i32 v[4];
#if defined(USE_AVX2)
        __m256i a = _mm256_setzero_si256(), b = a, c = a, d = a;
        for (unsigned j = 0; j < 1024; j += 32)
        {
            auto in = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + j));
            a =
              dot256(a, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(dense.data() + j)));
            b = dot256(
              b, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(dense.data() + 1024 + j)));
            c = dot256(
              c, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(dense.data() + 2048 + j)));
            d = dot256(
              d, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(dense.data() + 3072 + j)));
        }
        v[0] = sum8(a) + bias[0];
        v[1] = sum8(b) + bias[1];
        v[2] = sum8(c) + bias[2];
        v[3] = sum8(d) + bias[3];
#elif defined(USE_SSSE3)
        __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(bias.data())),
                b = _mm_setzero_si128();
        for (unsigned j = 0; j < 256; j += 2)
        {
            i32 lo, hi;
            std::memcpy(&lo, x + j * 4, 4);
            std::memcpy(&hi, x + j * 4 + 4, 4);
            a = dot128(a, _mm_set1_epi32(lo),
                       _mm_load_si128(reinterpret_cast<const __m128i*>(blocks.data() + j * 16)));
            b =
              dot128(b, _mm_set1_epi32(hi),
                     _mm_load_si128(reinterpret_cast<const __m128i*>(blocks.data() + j * 16 + 16)));
        }
        _mm_store_si128(reinterpret_cast<__m128i*>(v), _mm_add_epi32(a, b));
#else
        std::copy(bias.begin(), bias.end(), v);
        for (unsigned j = 0; j < 1024; ++j)
            for (unsigned k = 0; k < 4; ++k)
                v[k] += i32(x[j]) * dense[k * 1024 + j];
#endif
        return finish(v, readout.data());
    }
    usize get_content_hash() const {
        usize h = 0;
        for (auto value : dense)
            hash_combine(h, value);
        for (auto value : bias)
            hash_combine(h, value);
        for (auto value : readout)
            hash_combine(h, value);
        return h;
    }
};
}
#endif
