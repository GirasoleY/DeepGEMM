#include <cassert>
#include <stdexcept>
#include <string>

#include "../csrc/apis/mega_moe_config.hpp"

using deep_gemm::mega::MegaMoeWorkspaceConfig;
using deep_gemm::mega::MegaMoeWorkspaceConfigTuple;

template <typename F>
void rejects(F&& fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

int main() {
    // Exercise the production common validator without Torch or CUDA. There
    // is deliberately no transport selector: these are the same model rules.
    for (const auto& mma : {"fp8xfp4", "fp8xfp8", "bf16xbf16"}) {
        for (const int ranks : {1, 8, 12, 16}) {
            const MegaMoeWorkspaceConfig config({
                ranks * 17, 1920, 6, 2048, 1024, 2, mma, "swiglu", 37});
            config.validate(ranks);
            for (const int count : {0, 1, 17, 37})
                config.validate_launch(count, 2048, 1024, 2, mma, "swiglu");
            rejects([&] { config.validate_launch(38, 2048, 1024, 2, mma, "swiglu"); });
            rejects([&] { config.validate_launch(-1, 2048, 1024, 2, mma, "swiglu"); });
            rejects([&] { config.validate_launch(1, 4096, 1024, 2, mma, "swiglu"); });
            rejects([&] { config.validate_launch(1, 2048, 3072, 2, mma, "swiglu"); });
            rejects([&] { config.validate_launch(1, 2048, 1024, 0, mma, "swiglu"); });
            const std::string other_mma = std::string(mma) == "bf16xbf16" ? "fp8xfp4" : "bf16xbf16";
            rejects([&] { config.validate_launch(1, 2048, 1024, 2, other_mma, "swiglu"); });
            rejects([&] { config.validate_launch(1, 2048, 1024, 2, mma, "relu"); });
            rejects([&] { config.validate(0); });
            rejects([&] { config.validate(-1); });
        }
    }
    const MegaMoeWorkspaceConfigTuple valid{
        136, 1920, 6, 2048, 1024, 2, "fp8xfp8", "swiglu", 37};
    for (const auto& values : {
            MegaMoeWorkspaceConfigTuple{0, 1920, 6, 2048, 1024, 2, "fp8xfp8", "swiglu", 37},
            MegaMoeWorkspaceConfigTuple{137, 1920, 6, 2048, 1024, 2, "fp8xfp8", "swiglu", 37},
            MegaMoeWorkspaceConfigTuple{136, 32, 6, 2048, 1024, 2, "fp8xfp8", "swiglu", 37},
            MegaMoeWorkspaceConfigTuple{136, 1920, 0, 2048, 1024, 2, "fp8xfp8", "swiglu", 37},
            MegaMoeWorkspaceConfigTuple{136, 1920, 32, 2048, 1024, 2, "fp8xfp8", "swiglu", 37},
            MegaMoeWorkspaceConfigTuple{136, 1920, 6, 0, 1024, 2, "fp8xfp8", "swiglu", 37},
            MegaMoeWorkspaceConfigTuple{136, 1920, 6, 2048, 1024, -1, "fp8xfp8", "swiglu", 37},
            MegaMoeWorkspaceConfigTuple{136, 1920, 6, 2048, 1024, 2, "fp8xfp8", "swiglu", 0},
            MegaMoeWorkspaceConfigTuple{136, 1920, 6, 2048, 1024, 2, "unsupported", "swiglu", 37}}) {
        rejects([&] { MegaMoeWorkspaceConfig(values).validate(8); });
    }
    MegaMoeWorkspaceConfig(valid).validate(8);
}
