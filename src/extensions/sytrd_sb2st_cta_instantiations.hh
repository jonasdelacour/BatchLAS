#pragma once

#include "../util/template-instantiations.hh"

#define BATCHLAS_FOR_EACH_SYTRD_SB2ST_CTA_DISPATCH_TYPE(INVOKE) \
    INVOKE((float), (float)) \
    INVOKE((double), (double)) \
    INVOKE((std::complex<float>), (float)) \
    INVOKE((std::complex<double>), (double))