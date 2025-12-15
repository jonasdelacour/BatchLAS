#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/mempool.hh>
#include <internal/sort.hh>
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#define DEBUG_STEDC 0

namespace batchlas {

template <typename T>
auto givens_rotation_r(const T& a, const T& b) {
    T r = std::hypot(a, b);
    if (internal::is_numerically_zero(r)) {
        return std::array<T, 3>{T(1), T(0), T(0)};
    }
    return std::array<T, 3>{a / r, b / r, r};
}


template <typename T> 
auto sec_eval(const sycl::group<1>& cta, const VectorView<T>& v, const VectorView<T>& d, const T& x, const int32_t i, const int32_t n, const sycl::local_accessor<T, 1>& psi_buffer) {
    auto tid = cta.get_local_linear_id();
    auto bdim = cta.get_local_range()[0];
    auto bid = cta.get_group_linear_id();

    for (int k = tid; k < n; k += bdim) { psi_buffer[k] = v(k, bid) / (d(k, bid) - x); }
    sycl::group_barrier(cta);
    auto psi1 = sycl::joint_reduce(cta, psi_buffer.get_pointer(), psi_buffer.get_pointer() + i + 1, sycl::plus<T>());
    auto psi2 = (i + 1) < n ? sycl::joint_reduce(cta, psi_buffer.get_pointer() + i + 1, psi_buffer.get_pointer() + n, sycl::plus<T>()) : T(0);
    for (int k = tid; k < n; k += bdim) { psi_buffer[k] = v(k, bid) / ( (d(k, bid) - x) * (d(k, bid) - x)); }
    sycl::group_barrier(cta);
    auto psi1_prime = sycl::joint_reduce(cta, psi_buffer.get_pointer(), psi_buffer.get_pointer() + i + 1, sycl::plus<T>());
    auto psi2_prime = (i + 1) < n ? sycl::joint_reduce(cta, psi_buffer.get_pointer() + i + 1, psi_buffer.get_pointer() + n, sycl::plus<T>()) : T(0);
    return std::array<T, 4>{psi1, psi1_prime, psi2, psi2_prime};
}

template <typename T>
auto sec_eval_roc(const int32_t type, const int32_t k, const int32_t dd, const VectorView<T>& D, const VectorView<T>& z, const T& rho, const T& cor, bool modif){
    int32_t gout, hout;
    T er, fx, gx, hx, fdx, gdx, hdx, zz, tmp;
    // prepare computations
    // if type = 0: evaluate secular equation
    if(type == 0)
    {
        gout = k + 1;
        hout = k;
    }
    // if type = 1: evaluate secular equation without the k-th pole
    else if(type == 1)
    {
        if(modif)
        {
            tmp = D(k) - cor;
            D(k) = tmp;
        }
        gout = k;
        hout = k;
    }
    // if type = 2: evaluate secular equation without the k-th and (k+1)-th poles
    else if(type == 2)
    {
        if(modif)
        {
            tmp = D(k) - cor;
            D(k) = tmp;
            tmp = D(k + 1) - cor;
            D(k + 1) = tmp;
        }
        gout = k;
        hout = k + 1;
    }
    else
    {
        // unexpected value for type, something is wrong
        assert(false);
    }

    // computations
    gx = 0;
    gdx = 0;
    er = 0;
    for(int i = 0; i < gout; ++i)
    {
        tmp = D(i) - cor;
        if(modif)
            D(i) = tmp;
        zz = z(i);
        tmp = zz / tmp;
        gx += zz * tmp;
        gdx += tmp * tmp;
        er += gx;
    }
    er = std::abs(er);

    hx = 0;
    hdx = 0;
    for(int i = dd - 1; i > hout; --i)
    {
        tmp = D(i) - cor;
        if(modif)
            D(i) = tmp;
        zz = z(i);
        tmp = zz / tmp;
        hx += zz * tmp;
        hdx += tmp * tmp;
        er += hx;
    }

    fx = rho + gx + hx;
    fdx = gdx + hdx;

    return std::tuple{fx, fdx, gx, gdx, hx, hdx, er};
}

template <typename T>
SYCL_EXTERNAL T sec_solve_ext_roc(const int32_t dd,
                                  const VectorView<T>& D,
                                  const VectorView<T>& z,
                                  const T p)
{
    bool converged = false;
    T lowb, uppb, aa, bb, cc, x;
    T er, fx, fdx, gx, gdx, hx, hdx;
    T tau, eta;
    T dk, dkm1, ddk, ddkm1;
    int32_t k = dd - 1;
    int32_t km1 = dd - 2;

    // initialize
    dk = D(k);
    dkm1 = D(km1);
    x = dk + p / 2;
    T pinv = 1 / p;

    // find bounds and initial guess
    std::tie(cc, fdx, gx, gdx, hx, hdx, er) = sec_eval_roc(2, km1, dd, D, z, pinv, x, false);
    gdx = z(km1) * z(km1);
    hdx = z(k) * z(k);
    fx = cc + gdx / (dkm1 - x) - 2 * hdx * pinv;
    if(fx > 0)
    {
        // if the secular eq at the midpoint is positive, the root is in between
        // D[k] and the midpoint take D[k] as the origin, i.e. x = D[k] + tau with
        // tau in (0, uppb)
        lowb = 0;
        uppb = p / 2;
        tau = dk - dkm1;
        aa = -cc * tau + gdx + hdx;
        bb = hdx * tau;
        eta = std::sqrt(aa * aa + 4 * bb * cc);
        if(aa < 0)
            tau = 2 * bb / (eta - aa);
        else
            tau = (aa + eta) / (2 * cc);
    }
    else
    {
        // otherwise, the root is in between the midpoint and D[k+1]
        // take D[k+1] as the origin, i.e. x = D[k+1] + tau with tau in (lowb, 0)
        lowb = p / 2;
        uppb = p;
        eta = gdx / (dk - dkm1 + p) + hdx / p;
        if(cc <= eta)
            tau = p;
        else
        {
            tau = dk - dkm1;
            aa = -cc * tau + gdx + hdx;
            bb = hdx * tau;
            eta = std::sqrt(aa * aa + 4 * bb * cc);
            if(aa < 0)
                tau = 2 * bb / (eta - aa);
            else
                tau = (aa + eta) / (2 * cc);
        }
    }
    x = dk + tau; // initial guess

    // evaluate secular eq and get input values to calculate step correction
    std::tie(fx, fdx, gx, gdx, hx, hdx, er) = sec_eval_roc(0, km1, dd, D, z, pinv, dk, true);
    std::tie(fx, fdx, gx, gdx, hx, hdx, er) = sec_eval_roc(0, km1, dd, D, z, pinv, tau, true);

    // calculate tolerance er for convergence test
    er += std::abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;

    // if the value of secular eq is small enough, no point to continue;
    // converged!!!
    if(std::abs(fx) <= std::numeric_limits<T>::epsilon() * er)
        converged = true;

    // otherwise...
    else
    {
        // update bounds
        lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
        uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

        // calculate first step correction with fixed weight method
        ddk = D(k);
        ddkm1 = D(km1);
        cc = std::abs(fx - ddkm1 * gdx - ddk * hdx);
        aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
        bb = ddk * ddkm1 * fx;
        if(cc == 0)
        {
            eta = uppb - tau;
        }
        else
        {
            eta = std::sqrt(std::abs(aa * aa - 4 * bb * cc));
            if(aa >= 0)
                eta = (aa + eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa - eta);
        }

        // verify that the correction eta will get x closer to the root
        // i.e. eta*fx should be negative. If not the case, take a Newton step
        // instead
        if(fx * eta > 0)
            eta = -fx / (gdx + hdx);

        // now verify that applying the correction won't get the process out of
        // bounds if that is the case, bisect the interval instead
        if(tau + eta > uppb || tau + eta < lowb)
        {
            if(fx < 0)
                eta = (uppb - tau) / 2;
            else
                eta = (lowb - tau) / 2;
        }

        // take the step
        tau += eta;
        x = dk + tau;

        // evaluate secular eq and get input values to calculate step correction
        std::tie(fx, fdx, gx, gdx, hx, hdx, er) = sec_eval_roc(0, km1, dd, D, z, pinv, eta, true);

        // calculate tolerance er for convergence test
        er += std::abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;

        // MAIN ITERATION LOOP
        // ==============================================
        for(int i = 1; i < 50; ++i)
        {
            // if the value of secular eq is small enough, no point to continue;
            // converged!!!
            if(std::abs(fx) <= std::numeric_limits<T>::epsilon() * er)
            {
                converged = true;
                break;
            }

            // update bounds
            lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
            uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

                // calculate step correction
            ddk = D(k);
            ddkm1 = D(km1);
            cc = fx - ddkm1 * gdx - ddk * hdx;
            aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
            bb = ddk * ddkm1 * fx;
            eta = std::sqrt(std::abs(aa * aa - 4 * bb * cc));
            if(aa >= 0)
                eta = (aa + eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa - eta);

            // verify that the correction eta will get x closer to the root
            // i.e. eta*fx should be negative. If not the case, take a Newton step
            // instead
            if(fx * eta > 0)
                eta = -fx / (gdx + hdx);

            // now verify that applying the correction won't get the process out of
            // bounds if that is the case, bisect the interval instead
            if(tau + eta > uppb || tau + eta < lowb)
            {
                if(fx < 0)
                    eta = (uppb - tau) / 2;
                else
                    eta = (lowb - tau) / 2;
            }

            // take the step
            tau += eta;
            x = dk + tau;

            // evaluate secular eq and get input values to calculate step correction
            std::tie(fx, fdx, gx, gdx, hx, hdx, er) = sec_eval_roc(0, km1, dd, D, z, pinv, eta, true);

            // calculate tolerance er for convergence test
            er += std::abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;
        }
    }
    return x;
}

template <typename T>
SYCL_EXTERNAL T sec_solve_roc(int32_t dd, const VectorView<T>& d, const VectorView<T>& z, const T& rho, const int32_t k){
    bool converged = false;
    bool up, fixed;
    T lowb, uppb, aa, bb, cc, x;
    T nx, er, fx, fdx, gx, gdx, hx, hdx, oldfx;
    T tau, eta;
    T dk, dk1, ddk, ddk1;
    int32_t kk;
    int32_t k1 = k + 1;
    // initialize
    dk = d(k);
    dk1 = d(k1);
    x = (dk + dk1) / 2; // midpoint of interval
    tau = (dk1 - dk);
    T pinv = 1 / rho;
    // find bounds and initial guess; translate origin
    std::tie(cc, fdx, gx, gdx, hx, hdx, er) = sec_eval_roc(2, k, dd, d, z, pinv, x, false);
    gdx = z(k) * z(k);
    hdx = z(k1) * z(k1);
    fx = cc + 2 * (hdx - gdx) / tau;
    if(fx > 0)
    {
        // if the secular eq at the midpoint is positive, the root is in between
        // D(k) and the midpoint take D(k) as the origin, i.e. x = D(k) + tau with
        // tau in (0, uppb)
        lowb = 0;
        uppb = tau / 2;
        up = true;
        kk = k; // origin remains the same
        aa = cc * tau + gdx + hdx;
        bb = gdx * tau;
        eta = std::sqrt(std::abs(aa * aa - 4 * bb * cc));
        if(aa > 0)
            tau = 2 * bb / (aa + eta);
        else
            tau = (aa - eta) / (2 * cc);
        x = dk + tau; // initial guess
    }
    else
    {
        // otherwise, the root is in between the midpoint and D(k+1)
        // take D(k+1) as the origin, i.e. x = D(k+1) + tau with tau in (lowb, 0)
        lowb = -tau / 2;
        uppb = 0;
        up = false;
        kk = k + 1; // translate the origin
        aa = cc * tau - gdx - hdx;
        bb = hdx * tau;
        eta = std::sqrt(std::abs(aa * aa + 4 * bb * cc));
        if(aa < 0)
            tau = 2 * bb / (aa - eta);
        else
            tau = -(aa + eta) / (2 * cc);
        x = dk1 + tau; // initial guess
    }

    // evaluate secular eq and get input values to calculate step correction
    std::tie(fx, fdx, gx, gdx, hx, hdx, er) = sec_eval_roc(0, kk, dd, d, z, pinv, (up ? dk : dk1), true);
    std::tie(fx, fdx, gx, gdx, hx, hdx, er) = sec_eval_roc(1, kk, dd, d, z, pinv, tau, true);
    bb = z(kk);
    aa = bb / d(kk);
    fdx += aa * aa;
    bb *= aa;
    fx += bb;

    // calculate tolerance er for convergence test
    er += 8 * (hx - gx) + 2 * pinv + 3 * std::abs(bb) + std::abs(tau) * fdx;

    // if the value of secular eq is small enough, no point to continue;
    // converged!!!
    if(std::abs(fx) <= std::numeric_limits<T>::epsilon() * er)
        converged = true;

    // otherwise...
    else
    {
        // update bounds
        lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
        uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

        // calculate first step correction with fixed weight method
        ddk = d(k);
        ddk1 = d(k1);
        if(up)
            cc = fx - ddk1 * fdx - (dk - dk1) * z(k) * z(k) / ddk / ddk;
        else
            cc = fx - ddk * fdx - (dk1 - dk) * z(k1) * z(k1) / ddk1 / ddk1;
        aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
        bb = ddk * ddk1 * fx;
        if(cc == 0)
        {
            if(aa == 0)
            {
                if(up)
                    aa = z(k) * z(k) + ddk1 * ddk1 * (gdx + hdx);
                else
                    aa = z(k1) * z(k1) + ddk * ddk * (gdx + hdx);
            }
            eta = bb / aa;
        }
        else
        {
            eta = std::sqrt(std::abs(aa * aa - 4 * bb * cc));
            if(aa <= 0)
                eta = (aa - eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa + eta);
        }

        // verify that the correction eta will get x closer to the root
        // i.e. eta*fx should be negative. If not the case, take a Newton step
        // instead
        if(fx * eta >= 0)
            eta = -fx / fdx;

        // now verify that applying the correction won't get the process out of
        // bounds if that is the case, bisect the interval instead
        if(tau + eta > uppb || tau + eta < lowb)
        {
            if(fx < 0)
                eta = (uppb - tau) / 2;
            else
                eta = (lowb - tau) / 2;
        }

        // take the step
        tau += eta;
        x = (up ? dk : dk1) + tau;

        // evaluate secular eq and get input values to calculate step correction
        oldfx = fx;
        std::tie(fx, fdx, gx, gdx, hx, hdx, er) = sec_eval_roc(1, kk, dd, d, z, pinv, eta, true);
        bb = z(kk);
        aa = bb / d(kk);
        fdx += aa * aa;
        bb *= aa;
        fx += bb;

        // calculate tolerance er for convergence test
        er += 8 * (hx - gx) + 2 * pinv + 3 * std::abs(bb) + std::abs(tau) * fdx;

        // from now on, further step corrections will be calculated either with
        // fixed weights method or with normal interpolation depending on the value
        // of boolean fixed
        cc = up ? -1 : 1;
        fixed = (cc * fx) > (std::abs(oldfx) / 10);

        // MAIN ITERATION LOOP
        // ==============================================
        for(int i = 1; i < 50; ++i)
        {
            // if the value of secular eq is small enough, no point to continue;
            // converged!!!
            if(std::abs(fx) <= std::numeric_limits<T>::epsilon() * er)
            {
                converged = true;
                break;
            }

            // update bounds
            lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
            uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

            // calculate next step correction with either fixed weight method or
            // simple interpolation
            ddk = d(k);
            ddk1 = d(k1);
            if(fixed)
            {
                if(up)
                    cc = fx - ddk1 * fdx - (dk - dk1) * z(k) * z(k) / ddk / ddk;
                else
                    cc = fx - ddk * fdx - (dk1 - dk) * z(k1) * z(k1) / ddk1 / ddk1;
            }
            else
            {
                if(up)
                    gdx += aa * aa;
                else
                    hdx += aa * aa;
                cc = fx - ddk * gdx - ddk1 * hdx;
            }
            aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
            bb = ddk * ddk1 * fx;
            if(cc == 0)
            {
                if(aa == 0)
                {
                    if(fixed)
                    {
                        if(up)
                            aa = z(k) * z(k) + ddk1 * ddk1 * (gdx + hdx);
                        else
                            aa = z(k1) * z(k1) + ddk * ddk * (gdx + hdx);
                    }
                    else
                        aa = ddk * ddk * gdx + ddk1 * ddk1 * hdx;
                }
                eta = bb / aa;
            }
            else
            {
                eta = std::sqrt(std::abs(aa * aa - 4 * bb * cc));
                if(aa <= 0)
                    eta = (aa - eta) / (2 * cc);
                else
                    eta = (2 * bb) / (aa + eta);
            }

            // verify that the correction eta will get x closer to the root
            // i.e. eta*fx should be negative. If not the case, take a Newton step
            // instead
            if(fx * eta >= 0)
                eta = -fx / fdx;

            // now verify that applying the correction won't get the process out of
            // bounds if that is the case, bisect the interval instead
            if(tau + eta > uppb || tau + eta < lowb)
            {
                if(fx < 0)
                    eta = (uppb - tau) / 2;
                else
                    eta = (lowb - tau) / 2;
            }

            // take the step
            tau += eta;
            x = (up ? dk : dk1) + tau;

            // evaluate secular eq and get input values to calculate step correction
            oldfx = fx;
            std::tie(fx, fdx, gx, gdx, hx, hdx, er) = sec_eval_roc(1, kk, dd, d, z, pinv, eta, true);
            bb = z(kk);
            aa = bb / d(kk);
            fdx += aa * aa;
            bb *= aa;
            fx += bb;

            // calculate tolerance er for convergence test
            er += 8 * (hx - gx) + 2 * pinv + 3 * std::abs(bb) + std::abs(tau) * fdx;

            // update boolean fixed if necessary
            if(fx * oldfx > 0 && std::abs(fx) > std::abs(oldfx) / 10)
                fixed = !fixed;
        }
    }
    assert(converged && "sec_solve_roc did not converge!");
    return x; // return the computed root (k^{th} eigenvalue)
}

// Explicit instantiations to ensure device symbols are emitted for SYCL_EXTERNAL calls
template float sec_solve_ext_roc<float>(const int32_t dd,
                                        const VectorView<float>& D,
                                        const VectorView<float>& z,
                                        const float p);
template double sec_solve_ext_roc<double>(const int32_t dd,
                                          const VectorView<double>& D,
                                          const VectorView<double>& z,
                                          const double p);

template float sec_solve_roc<float>(int32_t dd,
                                    const VectorView<float>& d,
                                    const VectorView<float>& z,
                                    const float& rho,
                                    const int32_t k);
template double sec_solve_roc<double>(int32_t dd,
                                      const VectorView<double>& d,
                                      const VectorView<double>& z,
                                      const double& rho,
                                      const int32_t k);

template <typename T>
Event secular_solver(Queue& ctx, const VectorView<T>& d, const VectorView<T>& v, const MatrixView<T, MatrixFormat::Dense>& Qprime, const VectorView<T>& lambdas, const Span<int32_t>& n_reduced, const Span<T> rho, const T& tol_factor = 10.0) {
    //Solve the secular equation for each row in d and v
    auto N_max = d.size();
    const T safe_min = std::numeric_limits<T>::min();
    const T safe_denorm = std::numeric_limits<T>::denorm_min() > T(0) ? std::numeric_limits<T>::denorm_min() : safe_min;
    const T log_upper_bound = static_cast<T>(std::log(static_cast<double>(std::numeric_limits<T>::max())));
    const T log_lower_bound = static_cast<T>(std::log(static_cast<double>(safe_denorm)));
    ctx -> submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        auto shared_mem = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto vhat = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto vsign = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto log_numerators = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        auto log_denominators = sycl::local_accessor<T, 1>(sycl::range<1>(N_max), h);
        h.parallel_for(sycl::nd_range<1>(d.batch_size()*128, 128), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range(0);
            auto tid = item.get_local_linear_id();
            auto cta = item.get_group();
            auto n = n_reduced[bid];
            for (int k = tid; k < n; k += bdim) {
                log_numerators[k] = T(0);
                log_denominators[k] = T(0);
            }
            sycl::group_barrier(cta);
            for (int k = tid; k < n; k += bdim) { vsign[k] = (v(k, bid) >= T(0)) ? T(1) : T(-1); }
            for (int k = tid; k < n; k += bdim) { auto v_temp = v(k, bid); v(k, bid) *= v_temp; }
            
            auto v_norm2 = sycl::joint_reduce(cta, v.batch_item(bid).data_ptr(), v.batch_item(bid).data_ptr() + n, sycl::plus<T>());

            for (int i = 0; i < n; i++) {
                auto di = d(i, bid);
                auto di1 = i < n - 1 ? d(i + 1, bid) : d(n - 1, bid) + v_norm2;
                auto lam = i < n - 1 ? T(0.5) * (di + di1) : di1;
                auto [psi1, psi1_prime, psi2, psi2_prime] = sec_eval(cta, v, d, lam, i, n, shared_mem);

                auto condition = T(1) + psi1 + psi2 > T(0);
                auto shift = condition ? di : di1;
                lam  -= shift;
                if (condition) {di1 -= di;} else {di -= di1;}
                auto iter = 0;
                while (true) {
                    auto [psi1, psi1_prime, psi2, psi2_prime] = sec_eval(cta, v, d, lam + shift, i, n, shared_mem);
                    auto Di = condition ? -lam : di - lam;
                    auto Di1 = condition ? di1 - lam : -lam;
                    auto a = (Di + Di1) * (T(1) + psi1 + psi2) - Di * Di1 * (psi1_prime + psi2_prime);
                    auto b = Di * Di1 * (T(1) + psi1 + psi2);
                    auto c = (T(1) + psi1 + psi2) - Di * psi1_prime - Di1 * psi2_prime;
                    auto disc = a * a - T(4) * b * c;
                    disc = (disc < T(0) && disc < std::numeric_limits<T>::epsilon()) ? T(0) : disc;
                    auto s = std::sqrt(disc);
                    auto eta = (a > T(0)) ? (T(2) * b) / (a + s) : (a - s) / (T(2) * c);
                    //assert( (std::isfinite(eta) && (iter > 0)) || (iter == 0));
                    auto new_lam = lam + eta;
                    // Gu - Eisenstat stopping criterion: 

                    // $$ | g(u) | \leq \eta n (1 + |\psi_1(u)| + |\psi_2(u)|) $$

                    auto stop_threshold = T(4.0) * std::numeric_limits<T>::epsilon() * n * (1 + std::abs(psi1) + std::abs(psi2));
                    if (std::abs(eta) <= stop_threshold || iter >= 100) {
                        lam = new_lam;
                        break;
                    } else if (!std::isfinite(new_lam)) {
                        //Fall back to Newton's method
                        eta = -(1 + psi1 + psi2) / (psi1_prime + psi2_prime);
                        new_lam = lam + eta;
                    }
                        
                    lam = new_lam;
                    iter++;
                }
                
                if(tid == 0) lambdas(i,bid) = lam;

                for (int k = tid; k < n; k += bdim) {
                    auto sign = d(k, bid) - (lam + shift) >= T(0) ? T(1) : T(-1);
                    auto clamped_diff = std::max(sycl::fabs(d(k, bid) - (lam + shift)), std::numeric_limits<T>::epsilon()*T(1e-5));
                    Qview(k, i, bid) =  sign * clamped_diff;
                }


                for (int k = tid; k < n; k += bdim) {
                    auto diff = sycl::fabs(Qview(k, i, bid));
                    diff = sycl::fmax(diff, safe_denorm);
                    auto log_term = sycl::log(diff);
                    if (i == 0) {
                        log_numerators[k] = log_term;
                    } else {
                        log_numerators[k] += log_term;
                    }
                }

                for (int k = tid; k < i; k += bdim) {
                    auto diff = sycl::fabs(d(i, bid) - d(k, bid));
                    diff = sycl::fmax(diff, safe_denorm);
                    shared_mem[k] = sycl::log(diff);
                }
                sycl::group_barrier(cta);
                auto log_den2 = i > 0 ? sycl::joint_reduce(cta, shared_mem.get_pointer(), shared_mem.get_pointer() + i, sycl::plus<T>()) : T(0);

                for (int k = tid; k < n - i - 1; k += bdim) {
                    auto diff = sycl::fabs(d(k + i + 1, bid) - d(i, bid));
                    diff = sycl::fmax(diff, safe_denorm);
                    shared_mem[k] = sycl::log(diff);
                }
                sycl::group_barrier(cta);
                auto log_den1 = (i + 1) < n ? sycl::joint_reduce(cta, shared_mem.get_pointer(), shared_mem.get_pointer() + (n - i - 1), sycl::plus<T>()) : T(0);

                if (tid == 0) log_denominators[i] = log_den1 + log_den2;
                
                
                // $$v_i =  \frac{ \prod_{j=1}^n |d_j - \lambda_i| }{ \prod_{j=1}^{i-1} (d_j - d_i) \prod_{j=i+1}^{n} (d_j - d_i) } $$
                
                

            }

            sycl::group_barrier(cta);
            for (int k = tid; k < n; k += bdim) {
                auto exponent = T(0.5) * (log_numerators[k] - log_denominators[k]);
                exponent = sycl::fmax(log_lower_bound, sycl::fmin(log_upper_bound, exponent));
                auto magnitude = sycl::exp(exponent);
                vhat[k] = vsign[k] * magnitude;
            }

            
            
            // $$ q_i = \frac{(\lambda_i \mathtt{I} - D)^{-1} \vec{\hat{v}}}{||(\lambda_i \mathtt{I} - D)^{-1} \vec{\hat{v}}||} $$

            
            

            for (int i = 0; i < n; i++) {
                for (int k = tid; k < n; k += bdim) {
                    Qview(k, i, bid) = vhat[k] / Qview(k, i, bid);
                }
            }


            for (int i = 0; i < n; i++) {
                //for (int k = tid; k < n; k += bdim) shared_mem[k] = Qview(k, i, bid) * Qview(k, i, bid);

                //auto norm = std::sqrt(sycl::joint_reduce(cta, shared_mem.get_pointer(), shared_mem.get_pointer() + n, sycl::plus<T>()));
                auto norm = internal::nrm2<T>(cta, VectorView<T>(Qview.data() + i * Qview.ld(), n, 1, Qview.stride(), Qview.batch_size()));

                for (int k = tid; k < n; k += bdim) Qview(k, i, bid) = Qview(k, i, bid) / norm;
            }

            for (int k = tid; k < n; k += bdim) {
                auto d_term = lambdas(k, bid) > 0 ? d(k, bid) : (k + 1 < n ? d(k + 1, bid) : d(k, bid) + v_norm2);
                lambdas(k, bid) += d_term;
            }
        });
    });
    
    return ctx.get_event();
}

template <Backend B, typename T>
Event stedc_impl(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects, const MatrixView<T, MatrixFormat::Dense>& temp_Q)
{
    constexpr auto steqr_params = SteqrParams<T>{32, 10, std::numeric_limits<T>::epsilon(), false, false, false};
    auto n = d.size();
    auto batch_size = d.batch_size();
    if (n <= params.recursion_threshold){
        return steqr<B, T>(ctx, d, e, eigenvalues, ws, jobz, steqr_params, eigvects);
    }

    //Split the matrix into two halves
    int64_t m = n / 2;
    
    //When uneven the first half has size m x m and the second (m+1) x (m+1)
    auto d1 = d(Slice(0, m));
    auto e1 = e(Slice(0, m - 1));
    auto d2 = d(Slice(m, SliceEnd()));
    auto e2 = e(Slice(m, SliceEnd()));
    auto E1 = eigvects(Slice{0, m}, Slice(0, m));
    auto E2 = eigvects(Slice{m, SliceEnd()}, Slice(m, SliceEnd()));
    auto Q1 = temp_Q(Slice{0, m}, Slice(0, m));
    auto Q2 = temp_Q(Slice{m, SliceEnd()}, Slice(m, SliceEnd()));
    auto lambda1 = eigenvalues(Slice(0, m));
    auto lambda2 = eigenvalues(Slice(m, SliceEnd()));

    auto pool = BumpAllocator(ws);
    auto rho = pool.allocate<T>(ctx, batch_size);

    ctx -> parallel_for(sycl::range(batch_size), [=](sycl::id<1> idx) {
        //Modify the two diagonal entries adjacent to the split
        auto ix = idx[0];
        rho[ix] = e(m - 1, ix);
        d1(m - 1, ix) -= std::abs(rho[ix]);
        d2(0, ix) -= std::abs(rho[ix]);
    });

    //Scope this section: after the child recursions return, their workspace memory can be reused
    {
        auto pool = BumpAllocator(ws.subspan(BumpAllocator::allocation_size<T>(ctx, batch_size)));
        auto ws1 = pool.allocate<std::byte>(ctx, stedc_internal_workspace_size<B, T>(ctx, m, batch_size, jobz, params));
        auto ws2 = pool.allocate<std::byte>(ctx, stedc_internal_workspace_size<B, T>(ctx, n - m, batch_size, jobz, params));
        stedc_impl<B, T>(ctx, d1, e1, lambda1, ws1, jobz, params, E1, Q1);
        stedc_impl<B, T>(ctx, d2, e2, lambda2, ws2, jobz, params, E2, Q2);
    }
    
    //Once permutations are done we can free the memory once again
    auto permutation = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, 1, n, batch_size);
    auto v = VectorView<T>(pool.allocate<T>(ctx, n * batch_size), n, 1, n, batch_size);
    ctx -> submit([&](sycl::handler& h) {
        auto E1view = E1.kernel_view();
        auto E2view = E2.kernel_view();
        h.parallel_for(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range(0);
            auto tid = item.get_local_linear_id();
            auto sign = (e(m - 1, bid) >= 0) ? 1 : -1;
            //Normalized v through division by sqrt(2)
            for (int i = tid; i < m; i += bdim) {
                v(i, bid) = E1view(m - 1, i, bid) / std::sqrt(T(2));
            }
            for (int i = tid; i < n - m; i += bdim) {
                v(i + m, bid) = E2view(0, i, bid) / std::sqrt(T(2));
            }
        });
    });
    argsort(ctx, eigenvalues, permutation, SortOrder::Ascending, true);
    permute(ctx, eigenvalues, permutation);
    permute(ctx, v, permutation);
    permuted_copy(ctx, eigvects, temp_Q, permutation);

    auto keep_indices = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, 1, n, batch_size);
    auto n_reduced = pool.allocate<int32_t>(ctx, batch_size);
    //Deflation scheme
    T reltol = T(64.0) * std::numeric_limits<T>::epsilon();
    ctx -> submit([&](sycl::handler& h) {
        auto Q = temp_Q.kernel_view();
        auto scan_mem_include = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto scan_mem_exclude = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
        auto norm_mem = sycl::local_accessor<T, 1>(sycl::range<1>(n), h);
        h.parallel_for(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
        auto bid = item.get_group_linear_id();
        auto bdim = item.get_local_range(0);
        auto tid = item.get_local_linear_id();
        auto cta = item.get_group();

        for (int k = tid; k < n; k += bdim){
            keep_indices(k, bid) = 0;
            scan_mem_exclude[k] = 0;
            permutation(k, bid) = -1;
        }

        sycl::group_barrier(cta);
        for (int j = 0; j < n - 1; j++) {
            if(std::abs(eigenvalues(j + 1, bid) - eigenvalues(j, bid)) <= reltol * std::max(T(1), std::max(std::abs(eigenvalues(j + 1, bid)), std::abs(eigenvalues(j, bid))))) {
                auto f = v(j + 1, bid);
                auto g = v(j, bid);
                auto [c, s, r] = internal::lartg(f, g);
                sycl::group_barrier(cta);
                if (tid == 0) {
                    v(j, bid) = T(0.0);
                    v(j + 1, bid) = r;
                }
                for (int k = tid; k < n; k += bdim) {
                    auto Qi = Q(k,j,bid), Qj = Q(k,j + 1,bid);
                    Q(k,j,bid) = Qi*c - Qj*s;
                    Q(k,j + 1,bid) = Qj*c + Qi*s;
                }
            }
        }

        sycl::group_barrier(cta);
        //auto v_norm = std::sqrt(sycl::joint_reduce(cta, norm_mem.get_pointer(), norm_mem.get_pointer() + n, sycl::plus<T>()));
        //LAPACK LAED8 based tolerance:
        
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(eigenvalues(k, bid)); }
        auto eig_max = sycl::joint_reduce(cta,
                          norm_mem.template get_multi_ptr<sycl::access::decorated::no>().get(),
                          norm_mem.template get_multi_ptr<sycl::access::decorated::no>().get() + n,
                          sycl::maximum<T>());

        auto v_norm = internal::nrm2<T>(cta, v);
        for (int k = tid; k < n; k += bdim) { norm_mem[k] = std::abs(v(k, bid) / v_norm); }
        auto v_max = sycl::joint_reduce(cta,
                        norm_mem.template get_multi_ptr<sycl::access::decorated::no>().get(),
                        norm_mem.template get_multi_ptr<sycl::access::decorated::no>().get() + n,
                        sycl::maximum<T>());
        auto tol = T(8.0) * std::numeric_limits<T>::epsilon() * std::max(eig_max, v_max);

        //if(tid == 0) sycl::ext::oneapi::experimental::printf("Tolerance for deflation: %e\n", tol);
        for (int k = tid; k < n; k += bdim) {
            //sycl::ext::oneapi::experimental::printf("|v[%d]| * 2 * |rho| = %e\n", k, std::abs(T(2) * rho[bid]) * norm_mem[k]);
            if (std::abs(rho[bid] * norm_mem[k]) > tol ) {
                keep_indices(k, bid) = 1;
            } else {
                scan_mem_exclude[k] = 1;
            }
        }

        sycl::group_barrier(cta);

        //Exclusive scan to determine the indices to keep
        sycl::joint_exclusive_scan(cta,
                       keep_indices.batch_item(bid).data_ptr(),
                       keep_indices.batch_item(bid).data_ptr() + n,
                       scan_mem_include.template get_multi_ptr<sycl::access::decorated::no>().get(),
                       0,
                       sycl::plus<int32_t>());
        sycl::joint_exclusive_scan(cta,
                       scan_mem_exclude.template get_multi_ptr<sycl::access::decorated::no>().get(),
                       scan_mem_exclude.template get_multi_ptr<sycl::access::decorated::no>().get() + n,
                       scan_mem_exclude.template get_multi_ptr<sycl::access::decorated::no>().get(),
                       0,
                       sycl::plus<int32_t>());

        //sycl::group_barrier(cta);
        for (int k = tid; k < n; k += bdim) {
            if (keep_indices(k, bid) == 1) {
                permutation(scan_mem_include[k], bid) = k;
            } else {
                permutation(n - 1 - scan_mem_exclude[k], bid) = k;
            }
        }

        for (int k = tid; k < n; k += bdim) {
            //v(k, bid) *= std::sqrt(rho[bid]);
        }

        if (tid == 0) {
            n_reduced[bid] = scan_mem_include[n - 1] + keep_indices(n - 1, bid);
        }
        
        });
    });
    
    permute(ctx, temp_Q, eigvects, permutation);
    permute(ctx, eigenvalues, permutation);
    permute(ctx, v, permutation);

    auto temp_lambdas = VectorView<T>(pool.allocate<T>(ctx, n * batch_size), n, 1, n, batch_size);
    MatrixView<T> Qprime = MatrixView<T>(pool.allocate<T>(ctx, n * n * batch_size).data(), n, n, n, n * n, batch_size);
    Qprime.fill_identity(ctx);
    //Problem: We ultimately need to compute Q1 ⨂ Q2 * Qprime, however since we are deflating the columns of Q1 ⨂ Q2 we need to be careful about how we form Qprime.
    //Idea: As long as the columns of Qprime are the euclidean basis vectors, multiplying by Qprime is just a permutation of the columns of Q1 ⨂ Q2
    ctx -> submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        h.parallel_for(sycl::nd_range<1>(batch_size*128, 128), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto bdim = item.get_local_range(0);
            auto tid = item.get_local_linear_id();
            auto cta = item.get_group();
            auto Q_bid = Qview.batch_item(bid);
            auto sign = (e(m - 1, bid) >= 0) ? 1 : -1;
            auto n = n_reduced[bid];
            for (int k = tid; k < n * n; k += bdim) {
                auto i = k % n;
                auto j = k / n;
                Q_bid(i, j) = eigenvalues(i, bid);
            }
            sycl::group_barrier(cta);
            for (int k = tid; k < n; k += bdim) {
                //Get the k^th column of the bid^th batch item of Qprime
                auto dview = Q_bid(Slice{}, k);
                if (k == n - 1){
                    temp_lambdas(k, bid) = sec_solve_ext_roc(n, dview, v.batch_item(bid), std::abs(2 * rho[bid])) * sign;
                } else {
                    temp_lambdas(k, bid) = sec_solve_roc(n, dview, v.batch_item(bid), std::abs(2 * rho[bid]), k) * sign;
                }
            }
            sycl::group_barrier(cta);
        });
    });
    // Rescale v (secular vector) to avoid bad numerics when an eigenvalue
    // is too close to a pole. This mirrors ROCm's stedc_mergeValues_Rescale_kernel
    // but uses SYCL group collectives for the product reduction.
    ctx -> submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        h.parallel_for(
            sycl::nd_range<1>(d.batch_size() * 128, 128),
            [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id(); // 0 .. (batch_size*n - 1)

                auto g   = item.get_group();
                auto tid = item.get_local_linear_id();
                auto bdim = item.get_local_range(0);
                auto Qbid = Qview.batch_item(bid);

                auto dd = n_reduced[bid]; // number of non-deflated eigenvalues in this batch

                // Old poles D_j and new eigenvalues lambda_i
                for (int eid = 0; eid < dd; ++eid)
                {
                    // Compute partial product over j for this thread:
                    // prod_j (D_j - lambda_i)/(D_i - D_j), except j==i where we take (D_i - lambda_i).
                    auto Di = eigenvalues(eid, bid);
                    T partial = T(1);
                    for(int j = tid; j < dd; j += static_cast<int>(bdim))
                    {
                        partial *= (j == eid) ? Qbid(eid, j) : Qbid(eid, j) / (Di - eigenvalues(j, bid));
                    }

                    // Reduce the product across the work-group
                    T valf = sycl::reduce_over_group(g, partial, sycl::multiplies<T>());

                    if(tid == 0)
                    {
                        T mag  = sycl::sqrt(sycl::fabs(valf));
                        T sign = v(eid, bid) >= T(0) ? T(1) : T(-1);
                        v(eid, bid) = sign * mag;
                    }
                }
            });
    });

    ctx -> submit([&](sycl::handler& h) {
        auto Qview = Qprime.kernel_view();
        h.parallel_for(
            sycl::nd_range<1>(batch_size * 128, 128),
            [=](sycl::nd_item<1> item) {
                auto bid  = item.get_group_linear_id();   // batch index
                auto cta    = item.get_group();
                auto tid  = item.get_local_linear_id();
                auto bdim = item.get_local_range(0);

                const int dd = n_reduced[bid];
                auto Qbid = Qview.batch_item(bid);
                // For each non-deflated eigenvalue, build the corresponding
                // merge vector in the diagonal basis:
                //
                //   w_i(j) ∝ v_j / (D_j - lambda_i)
                //
                // normalized so ||w_i||_2 = 1.
                for(int eig = 0; eig < dd; ++eig)
                {
                    for(int i = tid; i < dd; i += static_cast<int>(bdim))
                    {
                        Qbid(i, eig) = v(i, bid) / Qbid(i, eig);
                    }

                    // Reduce to get the squared norm.
                    auto nrm2 = internal::nrm2(cta, Qview(Slice{0, dd}, eig));

                    // Normalize the column.
                    for(int i = tid; i < dd; i += static_cast<int>(bdim))
                    {
                        Qbid(i, eig) /= nrm2;
                    }
                }
            });
    });
    //secular_solver(ctx, eigenvalues, v, Qprime, temp_lambdas, n_reduced, rho, T(10.0));

    gemm<B>(ctx, temp_Q, Qprime, eigvects, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);

    ctx -> submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
        auto bid = item.get_group_linear_id();
        auto bdim = item.get_local_range(0);
        auto tid = item.get_local_linear_id();
        auto cta = item.get_group();

        for (int k = tid; k < n_reduced[bid]; k += bdim) {
            eigenvalues(k, bid) = temp_lambdas(k, bid);
        }
        });
    });
    argsort(ctx, eigenvalues, permutation, SortOrder::Ascending, true);
    permute(ctx, eigenvalues, permutation);
    permute(ctx, eigvects, temp_Q, permutation);

    return ctx.get_event();
}

template <Backend B, typename T>
Event stedc(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects) 
{
    if (d.size() != e.size() + 1) {
        throw std::runtime_error("The size of e must be one less than the size of d.");
    }
    if (d.size() != eigenvalues.size()) {
        throw std::runtime_error("The size of eigenvalues must match the size of d.");
    }
    if (d.batch_size() != e.batch_size() || d.batch_size() != eigenvalues.batch_size()) {
        throw std::runtime_error("The batch sizes of d, e, and eigenvalues must match.");
    }
    if (jobz == JobType::EigenVectors) {
        if (eigvects.rows() != d.size() || eigvects.cols() != d.size() || eigvects.batch_size() != d.batch_size()) {
            throw std::runtime_error("The dimensions of eigvects must match the size of d and its batch size.");
        }
    }
    //Clean the output matrix before we begin.
    eigvects.fill_zeros(ctx);
    auto pool = BumpAllocator(ws);
    auto n = d.size();
    auto alloc_size = BumpAllocator::allocation_size<T>(ctx, n * n * d.batch_size());
    auto temp_Q = MatrixView<T>(pool.allocate<T>(ctx, n * n * d.batch_size()).data(), n, n, n, n * n, d.batch_size());
    return stedc_impl<B, T>(ctx, d, e, eigenvalues, ws.subspan(alloc_size), jobz, params, eigvects, temp_Q);

}

template <Backend B, typename T>
size_t stedc_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
    if (n <= 0 || batch_size <= 0) {
        return 0;
    }

    size_t size = 0;
    auto d = VectorView<T>(nullptr, params.recursion_threshold, 1, 0, batch_size);
    auto e = VectorView<T>(nullptr, params.recursion_threshold - 1, 1, 0, batch_size);
    auto eigenvalues = VectorView<T>(nullptr, params.recursion_threshold, 1, 0, batch_size);
    // How many recursions do we need?
    auto n_rec = (n + params.recursion_threshold - 1) / params.recursion_threshold;
    auto m = (n + n_rec - 1) / n_rec; // Size of each subproblem
    
    // Compute the workspace size based on the job type
    switch (jobz) {
        case JobType::NoEigenVectors:
            size = steqr_buffer_size<T>(ctx, d, e, eigenvalues);
            break;
        case JobType::EigenVectors:
            size = steqr_buffer_size<T>(ctx, d, e, eigenvalues, jobz);
            break;
        default:
            throw std::runtime_error("Invalid job type");
    }
    size += 2 * BumpAllocator::allocation_size<int32_t>(ctx, 2 * (m + 1) * batch_size) + BumpAllocator::allocation_size<T>(ctx, batch_size); // For permutation array and rho storage

    return (size * n_rec) + 2 * BumpAllocator::allocation_size<T>(ctx, n * n * batch_size); // Multiply by number of recursions needed
}


template <Backend B, typename T>
size_t stedc_internal_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
    if (n <= 0 || batch_size <= 0) {
        return 0;
    }

    size_t size = 0;
    auto d = VectorView<T>(nullptr, params.recursion_threshold, 1, 0, batch_size);
    auto e = VectorView<T>(nullptr, params.recursion_threshold - 1, 1, 0, batch_size);
    auto eigenvalues = VectorView<T>(nullptr, params.recursion_threshold, 1, 0, batch_size);
    // How many recursions do we need?
    auto n_rec = (n + params.recursion_threshold - 1) / params.recursion_threshold;
    auto m = (n + n_rec - 1) / n_rec; // Size of each subproblem
    
    // Compute the workspace size based on the job type
    switch (jobz) {
        case JobType::NoEigenVectors:
            size = steqr_buffer_size<T>(ctx, d, e, eigenvalues);
            break;
        case JobType::EigenVectors:
            size = steqr_buffer_size<T>(ctx, d, e, eigenvalues, jobz);
            break;
        default:
            throw std::runtime_error("Invalid job type");
    }
    size += 2 * BumpAllocator::allocation_size<int32_t>(ctx, 2 * (m + 1) * batch_size) + BumpAllocator::allocation_size<T>(ctx, batch_size); // For permutation array and rho storage

    return (size * n_rec) + BumpAllocator::allocation_size<T>(ctx, n * n * batch_size); // Multiply by number of recursions needed
}

#if BATCHLAS_HAS_HOST_BACKEND
template Event stedc<Backend::NETLIB, float>(Queue& ctx, const VectorView<float>& d, const VectorView<float>& e, const VectorView<float>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<float> params, const MatrixView<float, MatrixFormat::Dense>& eigvects);
template Event stedc<Backend::NETLIB, double>(Queue& ctx, const VectorView<double>& d, const VectorView<double>& e, const VectorView<double>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<double> params, const MatrixView<double, MatrixFormat::Dense>& eigvects);

template size_t stedc_workspace_size<Backend::NETLIB, float>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<float> params);
template size_t stedc_workspace_size<Backend::NETLIB, double>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<double> params);
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
template Event stedc<Backend::CUDA, float>(Queue& ctx, const VectorView<float>& d, const VectorView<float>& e, const VectorView<float>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<float> params, const MatrixView<float, MatrixFormat::Dense>& eigvects);
template Event stedc<Backend::CUDA, double>(Queue& ctx, const VectorView<double>& d, const VectorView<double>& e, const VectorView<double>& eigenvalues, const Span<std::byte>& ws, JobType jobz, StedcParams<double> params, const MatrixView<double, MatrixFormat::Dense>& eigvects);

template size_t stedc_workspace_size<Backend::CUDA, float>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<float> params);
template size_t stedc_workspace_size<Backend::CUDA, double>(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<double> params);
#endif


}
