#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/mempool.hh>
#include <util/sycl-local-accessor-helpers.hh>
#include <internal/sort.hh>
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "stedc_secular.hh"

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
    auto psi1 = sycl::joint_reduce(cta, util::get_raw_ptr(psi_buffer), util::get_raw_ptr(psi_buffer) + i + 1, sycl::plus<T>());
    auto psi2 = (i + 1) < n ? sycl::joint_reduce(cta, util::get_raw_ptr(psi_buffer) + i + 1, util::get_raw_ptr(psi_buffer) + n, sycl::plus<T>()) : T(0);
    for (int k = tid; k < n; k += bdim) { psi_buffer[k] = v(k, bid) / ( (d(k, bid) - x) * (d(k, bid) - x)); }
    sycl::group_barrier(cta);
    auto psi1_prime = sycl::joint_reduce(cta, util::get_raw_ptr(psi_buffer), util::get_raw_ptr(psi_buffer) + i + 1, sycl::plus<T>());
    auto psi2_prime = (i + 1) < n ? sycl::joint_reduce(cta, util::get_raw_ptr(psi_buffer) + i + 1, util::get_raw_ptr(psi_buffer) + n, sycl::plus<T>()) : T(0);
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

template <typename T>
Event secular_solver(Queue& ctx, const VectorView<T>& d, const VectorView<T>& v, const MatrixView<T, MatrixFormat::Dense>& Qprime, const VectorView<T>& lambdas, const Span<int32_t>& n_reduced, const Span<T> rho, const T& tol_factor) {
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
                    auto new_lam = lam + eta;

                    auto stop_threshold = tol_factor * std::numeric_limits<T>::epsilon() * n * (1 + std::abs(psi1) + std::abs(psi2));
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
                auto log_den2 = i > 0 ? sycl::joint_reduce(cta, util::get_raw_ptr(shared_mem), util::get_raw_ptr(shared_mem) + i, sycl::plus<T>()) : T(0);

                for (int k = tid; k < n - i - 1; k += bdim) {
                    auto diff = sycl::fabs(d(k + i + 1, bid) - d(i, bid));
                    diff = sycl::fmax(diff, safe_denorm);
                    shared_mem[k] = sycl::log(diff);
                }
                sycl::group_barrier(cta);
                auto log_den1 = (i + 1) < n ? sycl::joint_reduce(cta, util::get_raw_ptr(shared_mem), util::get_raw_ptr(shared_mem) + (n - i - 1), sycl::plus<T>()) : T(0);

                if (tid == 0) log_denominators[i] = log_den1 + log_den2;
            }

            sycl::group_barrier(cta);
            for (int k = tid; k < n; k += bdim) {
                auto exponent = T(0.5) * (log_numerators[k] - log_denominators[k]);
                exponent = sycl::fmax(log_lower_bound, sycl::fmin(log_upper_bound, exponent));
                auto magnitude = sycl::exp(exponent);
                vhat[k] = vsign[k] * magnitude;
            }

            for (int i = 0; i < n; i++) {
                for (int k = tid; k < n; k += bdim) {
                    Qview(k, i, bid) = vhat[k] / Qview(k, i, bid);
                }
            }


            for (int i = 0; i < n; i++) {
                auto norm = internal::nrm2<T>(cta, VectorView<T>(Qview.data() + i * Qview.ld(), n, Qview.batch_size(), 1, Qview.stride()));

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

template Event secular_solver<float>(Queue& ctx,
                                     const VectorView<float>& d,
                                     const VectorView<float>& v,
                                     const MatrixView<float, MatrixFormat::Dense>& Qprime,
                                     const VectorView<float>& lambdas,
                                     const Span<int32_t>& n_reduced,
                                     const Span<float> rho,
                                     const float& tol_factor);

template Event secular_solver<double>(Queue& ctx,
                                      const VectorView<double>& d,
                                      const VectorView<double>& v,
                                      const MatrixView<double, MatrixFormat::Dense>& Qprime,
                                      const VectorView<double>& lambdas,
                                      const Span<int32_t>& n_reduced,
                                      const Span<double> rho,
                                      const double& tol_factor);

} // namespace batchlas
