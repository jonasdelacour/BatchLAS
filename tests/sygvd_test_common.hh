// SPDX-License-Identifier: MIT
#pragma once
// The fixture base sygvd_tests.cc and masked_batch_tests.cc share: potrf pinned native on GPU backends,
// whether sygvd's vendor family is linked, and the sygvd families a pin must reach.
#include <batchlas/backend_config.h>
#include <batchlas/util/sycl-device-queue.hh>

#include <tests/test_utils.hh>

#include <src/ops/potrf/choice.hh>
#include <src/select/select.hh>

#include <optional>
#include <string>
#include <vector>

namespace sygvd_test {

template <typename Config>
class SygvdFixture : public test_utils::BatchLASTest<Config> {
protected:
    static constexpr batchlas::Backend BK = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        // cuSOLVER's Upper potrf overwrites the lower triangle (agent guide section 8 rule 4), which would
        // wipe B's poison before sygst or the back-transform read it; the native potrf leaves it alone.
        if constexpr (BK != batchlas::Backend::NETLIB) potrf_pin_.emplace("potrf", "native");
    }

    batchlas::Queue& q() { return *this->ctx; }

    /// Whether the build links sygvd's vendor family (and, on the host, sygst's) for this backend.
    static constexpr bool vendor_linked() {
        if constexpr (BK == batchlas::Backend::CUDA) return BATCHLAS_HAS_CUSOLVER;
        if constexpr (BK == batchlas::Backend::NETLIB) return BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS;
        return false;
    }

    /// The sygvd families a pin must reach on this backend.
    static std::vector<std::string> sygvd_families() {
        std::vector<std::string> f{"compose"};
        if (vendor_linked()) f.push_back("vendor");
        return f;
    }

private:
    std::optional<batchlas::select::ScopedPin<batchlas::ops::potrf::PotrfChoice>> potrf_pin_;
};

}  // namespace sygvd_test
