import numpy as np
import matplotlib.pyplot as plt


# ./helmholtz_1d -ksp_converged_reason -ksp_view -pc_type none -ksp_rtol 1e-10

dx = np.array([1/8, 1/16, 1/32, 1/64, 1/128, 1/256])  # grid spacings

# NO PRECONDITIONING - same norms as we should expect
errors_helmholtz_1d = np.array([0.146125, 0.0390549, 0.00963906, 0.00240207, 0.000600039, 0.00014998])


n_ksp_iters = []

slope, intercept = np.polyfit(np.log(dx), np.log(errors_helmholtz_1d), 1)

plt.figure(figsize=(6, 5))
plt.loglog(dx, errors, 'o-', label=f'Rate ≈ {abs(slope):.2f}')
plt.xlabel('Grid spacing (dx)')
plt.ylabel('Error')
plt.title('Error Convergence Plot')
plt.grid(True, which='both', ls='--')
plt.legend()

plt.tight_layout()
plt.savefig('convergence.png', dpi=300, transparent=True)
plt.show()
