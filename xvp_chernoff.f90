program verify_chernoff
  !---------------------------------------------------------------------
  ! Verify Chernoff (MGF) tail bounds and the Vysochanskiĭ–Petunin bound
  ! for |X| ≥ k:
  !   VP bound:          P(|X| ≥ k) ≤ 4/(9 k^2)
  !   Chernoff bound:    P(|X| ≥ k) ≤ 2 inf_{t>0} M_X(t) e^{-t k}
  ! Distributions: Normal(0,1), Laplace(0,b), Logistic(0,1), Sech(0,1).
  ! k ranges from k_min in steps of k_inc for nk points.
  ! Columns: k, VP_bound, Bound_N, P_N, Bound_L, P_L,
  !          Bound_Log, P_Log, Bound_Sech, P_Sech
  !---------------------------------------------------------------------
  implicit none

  integer, parameter :: dp = kind(1.0d0), nk = 15
  integer, parameter :: nint_l = 500, nint_s = 500
  real(kind=dp), parameter :: k_min = 1.0_dp, k_inc = 1.0_dp, pi = acos(-1.0_dp)
  real(kind=dp), parameter :: b = 1.0_dp/sqrt(2.0_dp)  ! Laplace scale

  real(kind=dp) :: k, vp_bound, p_norm, bound_norm, p_lap, bound_lap, t_star, &
                   p_log, bound_log, p_sech, bound_sech, t, dt, cur
  integer :: i, j

  print "(A6,2X,A10,2X,9(A12,2X))", "   k", " VP_bound", "Bound_N", " P_N", &
       "Bound_L"," P_L","Bound_Log"," P_Log","Bound_Sech"," P_Sech"

  do i = 1, nk
    k = k_min + (i-1)*k_inc
    ! VP bound
    vp_bound = 4.0_dp / (9.0_dp * k**2)

    ! --- Normal(0,1) ---
    p_norm     = 2.0_dp*(1.0_dp - 0.5_dp*(1.0_dp + erf(k/sqrt(2.0_dp))))
    bound_norm = 2.0_dp * exp(-k*k/2.0_dp)

    ! --- Laplace(0,b) ---
    p_lap      = exp(-k/b)
    t_star     = (sqrt(b*b + k*k) - b) / (b*k)
    bound_lap  = 2.0_dp * exp(-t_star*k) / (1.0_dp - b*b*t_star*t_star)

    ! --- Logistic(0,1) via grid search over t in (0,1) ---
    dt         = 0.99_dp / nint_l
    bound_log  = huge(1.0_dp)
    do j = 1, nint_l-1
      t   = j * dt
      cur = 2.0_dp * (pi*t / sin(pi*t)) * exp(-t*k)
      if (cur < bound_log) bound_log = cur
    end do
    p_log = 2.0_dp * exp(-k) / (1.0_dp + exp(-k))

    ! --- Hyperbolic secant (MGF = sec(t)) via grid search over t in (0,pi/2) ---
    dt          = (pi/2.0_dp - 0.001_dp) / nint_s
    bound_sech  = huge(1.0_dp)
    do j = 1, nint_s-1
      t   = j * dt
      cur = 2.0_dp * (1.0_dp / cos(t)) * exp(-t*k)
      if (cur < bound_sech) bound_sech = cur
    end do
    p_sech = 1.0_dp - (4.0_dp/pi) * atan(tanh(pi*k/4.0_dp))

    print "(F6.3,2X,9(F12.8,2X))", k, vp_bound, bound_norm, p_norm, bound_lap, &
          p_lap, bound_log, p_log, bound_sech, p_sech
  end do

end program verify_chernoff
